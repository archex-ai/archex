"""
SPECTRA — Synthetic EMG Generator
archex-ai/archex · spectra/generators/emg.py

Generates physiologically realistic surface EMG signals for 6 prosthetic
hand gesture classes without hardware, participants, or ethics approval.

Built on published spectral profiles:
  De Luca CJ (1997) — "The use of surface electromyography in biomechanics"
  Phinyomark A et al (2012) — "Feature reduction and selection for EMG signal classification"

Output format: CSV with columns [sample_id, gesture_label, gesture_name, ch_0..ch_7, split]
Each row = one 250ms window at 250Hz = 62 samples per channel, flattened.

Usage:
  python -m spectra.generators.emg --gestures 6 --samples 1000 --output emg_data.csv
  python -m spectra.generators.emg --gestures 6 --samples 30000 --output spectra_v1.csv --seed 42

Gesture classes:
  0 — rest
  1 — open hand
  2 — close (power grip)
  3 — pinch (fine grip)
  4 — point (index extension)
  5 — wrist flexion
"""

import argparse
import csv
import os
import sys
import time
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# PHYSIOLOGICAL PARAMETERS
# Per-gesture muscle activation profiles for 8-channel forearm EMG array.
# Values are mean activation (0-1) per channel derived from published literature.
# Channel layout: [FCR, FCU, ECR, ECU, FDS, EDC, FPL, APB]
# FCR=Flexor Carpi Radialis, FCU=Flexor Carpi Ulnaris,
# ECR=Extensor Carpi Radialis, ECU=Extensor Carpi Ulnaris,
# FDS=Flexor Digitorum Superficialis, EDC=Extensor Digitorum Communis,
# FPL=Flexor Pollicis Longus, APB=Abductor Pollicis Brevis
# ─────────────────────────────────────────────────────────────────────────────

GESTURE_PROFILES = {
    0: {
        "name": "rest",
        "activations": np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01]),
        "activation_noise": 0.01,
        "dominant_freq_hz": 15.0,
        "freq_bandwidth_hz": 10.0,
    },
    1: {
        "name": "open_hand",
        "activations": np.array([0.10, 0.08, 0.75, 0.70, 0.05, 0.80, 0.05, 0.10]),
        "activation_noise": 0.08,
        "dominant_freq_hz": 80.0,
        "freq_bandwidth_hz": 50.0,
    },
    2: {
        "name": "close_power_grip",
        "activations": np.array([0.60, 0.65, 0.10, 0.12, 0.85, 0.05, 0.70, 0.20]),
        "activation_noise": 0.10,
        "dominant_freq_hz": 95.0,
        "freq_bandwidth_hz": 60.0,
    },
    3: {
        "name": "pinch_fine_grip",
        "activations": np.array([0.40, 0.30, 0.15, 0.10, 0.50, 0.08, 0.75, 0.80]),
        "activation_noise": 0.09,
        "dominant_freq_hz": 75.0,
        "freq_bandwidth_hz": 45.0,
    },
    4: {
        "name": "point_index_extension",
        "activations": np.array([0.08, 0.06, 0.30, 0.25, 0.20, 0.65, 0.05, 0.10]),
        "activation_noise": 0.08,
        "dominant_freq_hz": 70.0,
        "freq_bandwidth_hz": 40.0,
    },
    5: {
        "name": "wrist_flexion",
        "activations": np.array([0.80, 0.75, 0.05, 0.08, 0.35, 0.04, 0.10, 0.05]),
        "activation_noise": 0.09,
        "dominant_freq_hz": 85.0,
        "freq_bandwidth_hz": 55.0,
    },
}

SAMPLE_RATE_HZ = 250          # Standard surface EMG sample rate
WINDOW_MS = 250               # Window length in milliseconds
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_MS / 1000)   # = 62 samples
N_CHANNELS = 8
AMPLITUDE_UV = 500.0          # Max amplitude in microvolts
NOISE_FLOOR_UV = 8.0          # Baseline noise floor


def _band_limited_noise(
    rng: np.random.Generator,
    n_samples: int,
    sample_rate: float,
    center_freq: float,
    bandwidth: float,
    amplitude: float,
) -> np.ndarray:
    """
    Generate bandpass-limited noise resembling EMG power spectral density.

    Uses sum-of-sinusoids method: physiologically realistic, deterministic
    given a fixed RNG, and fast without requiring scipy.
    """
    t = np.arange(n_samples) / sample_rate
    signal = np.zeros(n_samples)

    # Number of frequency components — more = smoother spectral shape
    n_components = 40
    freq_low = max(1.0, center_freq - bandwidth / 2)
    freq_high = center_freq + bandwidth / 2
    freqs = np.linspace(freq_low, freq_high, n_components)

    # Gaussian amplitude weighting centred on dominant frequency
    weights = np.exp(-0.5 * ((freqs - center_freq) / (bandwidth / 4)) ** 2)
    weights /= weights.sum()

    for freq, weight in zip(freqs, weights):
        phase = rng.uniform(0, 2 * np.pi)
        signal += weight * np.sin(2 * np.pi * freq * t + phase)

    # Scale to amplitude
    peak = np.abs(signal).max()
    if peak > 0:
        signal = signal / peak * amplitude

    return signal


def generate_window(
    rng: np.random.Generator,
    gesture_id: int,
) -> np.ndarray:
    """
    Generate one 250ms EMG window for a given gesture class.

    Returns shape (N_CHANNELS, WINDOW_SAMPLES) in microvolts.
    """
    profile = GESTURE_PROFILES[gesture_id]
    activations = profile["activations"]
    act_noise = profile["activation_noise"]
    center_freq = profile["dominant_freq_hz"]
    bandwidth = profile["freq_bandwidth_hz"]

    window = np.zeros((N_CHANNELS, WINDOW_SAMPLES))

    for ch in range(N_CHANNELS):
        # Per-channel activation level with inter-trial variability
        act = np.clip(
            activations[ch] + rng.normal(0, act_noise),
            0.0, 1.0
        )

        if act < 0.03:
            # Near-silent channel: just noise floor
            window[ch] = rng.normal(0, NOISE_FLOOR_UV, WINDOW_SAMPLES)
            continue

        # EMG-like bandpass signal
        emg_amplitude = act * AMPLITUDE_UV
        emg_signal = _band_limited_noise(
            rng, WINDOW_SAMPLES, SAMPLE_RATE_HZ,
            center_freq, bandwidth, emg_amplitude
        )

        # Full-wave rectification characteristic of real EMG envelope
        # Add noise floor
        baseline_noise = rng.normal(0, NOISE_FLOOR_UV, WINDOW_SAMPLES)

        # Simulate muscle onset: ramp up in first 20ms, stable through window
        onset_samples = int(0.020 * SAMPLE_RATE_HZ)  # 5 samples
        ramp = np.ones(WINDOW_SAMPLES)
        ramp[:onset_samples] = np.linspace(0.3, 1.0, onset_samples)

        window[ch] = emg_signal * ramp + baseline_noise

    return window


def generate_dataset(
    n_samples: int,
    n_gesture_classes: int = 6,
    seed: int = 42,
    val_split: float = 0.15,
    verbose: bool = True,
) -> list[dict]:
    """
    Generate n_samples EMG windows balanced across gesture classes.

    Args:
        n_samples: Total number of windows to generate.
        n_gesture_classes: Number of gesture classes (2–6).
        seed: Random seed for reproducibility.
        val_split: Fraction held out for validation split.
        verbose: Print progress.

    Returns:
        List of dicts with keys: sample_id, gesture_label, gesture_name,
        ch_0..ch_7 (comma-separated sample values), split.
    """
    assert 2 <= n_gesture_classes <= 6, "n_gesture_classes must be 2–6"
    rng = np.random.default_rng(seed)
    gesture_ids = list(range(n_gesture_classes))

    samples_per_class = n_samples // n_gesture_classes
    remainder = n_samples % n_gesture_classes

    rows = []
    sample_id = 0
    t0 = time.time()

    for gesture_id in gesture_ids:
        count = samples_per_class + (1 if gesture_id < remainder else 0)
        if verbose:
            print(
                f"  Generating {count:,} windows for class {gesture_id} "
                f"({GESTURE_PROFILES[gesture_id]['name']})...",
                flush=True
            )

        for i in range(count):
            window = generate_window(rng, gesture_id)
            row = {
                "sample_id": sample_id,
                "gesture_label": gesture_id,
                "gesture_name": GESTURE_PROFILES[gesture_id]["name"],
            }
            for ch in range(N_CHANNELS):
                # Store as rounded microvolts, space-efficient
                row[f"ch_{ch}"] = " ".join(
                    f"{v:.2f}" for v in window[ch]
                )
            row["split"] = "val" if rng.random() < val_split else "train"
            rows.append(row)
            sample_id += 1

        if verbose and (gesture_id + 1) % 2 == 0:
            elapsed = time.time() - t0
            done = sample_id / n_samples
            eta = elapsed / done * (1 - done) if done > 0 else 0
            print(f"    [{done*100:.0f}%] elapsed {elapsed:.1f}s, ETA {eta:.1f}s")

    # Shuffle so classes are interleaved
    indices = rng.permutation(len(rows))
    rows = [rows[i] for i in indices]

    # Re-assign sample_ids after shuffle
    for i, row in enumerate(rows):
        row["sample_id"] = i

    if verbose:
        total = time.time() - t0
        train_n = sum(1 for r in rows if r["split"] == "train")
        val_n = len(rows) - train_n
        print(f"\n  Done. {len(rows):,} windows in {total:.1f}s")
        print(f"  Train: {train_n:,} | Val: {val_n:,}")

    return rows


def save_csv(rows: list[dict], output_path: str) -> None:
    """Save generated dataset to CSV."""
    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    size_mb = path.stat().st_size / 1_048_576
    print(f"  Saved → {path} ({size_mb:.1f} MB, {len(rows):,} rows)")


def quick_stats(rows: list[dict]) -> None:
    """Print basic sanity-check stats."""
    from collections import Counter
    labels = Counter(r["gesture_label"] for r in rows)
    splits = Counter(r["split"] for r in rows)
    print("\n  Class distribution:")
    for label, count in sorted(labels.items()):
        name = GESTURE_PROFILES[label]["name"]
        print(f"    {label} {name:<25} {count:>6,}")
    print(f"\n  Splits: train={splits['train']:,}  val={splits['val']:,}")

    # Spot-check one channel of one sample
    sample = rows[0]
    ch0 = np.array([float(x) for x in sample["ch_0"].split()])
    print(f"\n  Sample 0 (class={sample['gesture_label']}, "
          f"name={sample['gesture_name']})")
    print(f"    ch_0: mean={ch0.mean():.2f}µV  std={ch0.std():.2f}µV  "
          f"peak={np.abs(ch0).max():.2f}µV  shape=({len(ch0)},)")


def main():
    parser = argparse.ArgumentParser(
        description="SPECTRA synthetic EMG generator — ARCHEX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gestures", type=int, default=6, choices=range(2, 7),
                        help="Number of gesture classes (2–6)")
    parser.add_argument("--samples", type=int, default=1000,
                        help="Total number of 250ms windows to generate")
    parser.add_argument("--output", type=str, default="spectra_emg.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data for validation split")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress progress output")
    args = parser.parse_args()

    print(f"\nSPECTRA EMG Generator — ARCHEX")
    print(f"  Gestures: {args.gestures} classes")
    print(f"  Samples:  {args.samples:,} windows × {N_CHANNELS} channels × {WINDOW_SAMPLES} timesteps")
    print(f"  Seed:     {args.seed}")
    print(f"  Output:   {args.output}\n")

    rows = generate_dataset(
        n_samples=args.samples,
        n_gesture_classes=args.gestures,
        seed=args.seed,
        val_split=args.val_split,
        verbose=not args.quiet,
    )

    quick_stats(rows)
    save_csv(rows, args.output)
    print("\nDone. Run with --samples 30000 to generate the full SPECTRA v1 dataset.")


if __name__ == "__main__":
    main()