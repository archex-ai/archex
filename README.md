<div align="center">

# ARCHEX

**The open-source intelligence layer for human augmentation.**

*Origin. Command. Evolution.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/🤗_Model-MedLM_v0-yellow)](https://huggingface.co/archex-ai/archex-medlm-v0)
[![Demo](https://img.shields.io/badge/🤗_Demo-Live-green)](https://huggingface.co/spaces/archex-ai/archex-medlm-demo)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-7289DA)](https://discord.gg/FeMCJZJTry)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-meridian--v1-blue)](https://huggingface.co/datasets/archex-ai/meridian-v1)

---

*What CUDA did for GPU computing — ARCHEX does for the human body.*

</div>

---

## What is ARCHEX?

ARCHEX is an open-source medical AI stack built for **human augmentation**.

We are not building another chatbot. We are building the intelligence layer that will power the next generation of prosthetic limbs, neural interfaces, cochlear implants, and human augmentation devices — the same way NVIDIA's CUDA became the invisible backbone of every AI system on earth.

**Few people. One mission. The window is open right now.**

OpenAI is building for chat. Google is building for search. Nobody is building the AI brain for the 500 million people who live with limb loss, paralysis, hearing loss, or vision impairment.

That is what ARCHEX is for.

---

## The Stack

```
ARCHEX
├── core/                  # Model training pipeline
│   └── archex-medlm/      # Medical language model (open weights)
│
├── spectra/               # Synthetic biosignal environment
│   ├── generators/        # EMG/EEG synthetic signal generation
│   └── sim2real/          # Sim-to-real transfer utilities
│
├── meridian/              # Medical knowledge ingestion pipeline
│   ├── pubmed/            # PubMed Open Access ingestion
│   └── clinical/          # Clinical notes pipeline
│
├── signal-encoder/        # 1D-CNN biosignal → LLM embedding bridge
│   ├── models/            # Pretrained signal encoders
│   └── tokenizer/         # BioSignal tokenization
│
└── sdk/                   # ARCHEX SDK for bionic device integration
    ├── python/
    ├── cpp/               # For embedded/real-time systems
    └── examples/
```

---

## Why now?

| Signal | What it means |
|--------|--------------|
| Prosthetic limb market growing 6.8% YoY | Demand is accelerating |
| EMG-controlled prosthetics still use 1990s pattern matching | The AI gap is enormous |
| Open-source LLMs now match GPT-3.5 at 7B params | We can afford to train this |
| Synthetic biosignal generation is finally viable | Data is no longer the bottleneck |
| No foundation model exists for bionic control signals | The category is unclaimed |

---

## The BioSignal Tokenization Problem

Every LLM processes text as tokens. EMG signals, EEG recordings, and neural spike trains are not text — they are continuous time-series with temporal dependencies, amplitude distributions, and frequency-domain features that existing tokenizers destroy.

**ARCHEX solves this with a learnable 1D-CNN signal encoder** that maps biosignal windows directly into the LLM's embedding space. A 250ms EMG window becomes a vector the transformer can reason about — alongside clinical notes, patient history, and movement intent labels.

This is the technical foundation for a bionic limb that doesn't just detect "open hand" vs "close hand" — but understands *why* the user wants to move, in the context of their clinical history and rehabilitation progress.

---

## SPECTRA — Synthetic Biosignal Environment

SPECTRA generates physiologically realistic biosignal training data without hardware, participants, or cost. Built on published EMG spectral profiles, it produces unlimited synthetic training samples across 6 prosthetic hand gesture classes.

No hardware. No participants. No ethics approval required. Unlimited data at ₹0.

This is the sim-to-real strategy — the same approach Boston Dynamics uses for robot locomotion. Train entirely in simulation, deploy to real hardware.

---

## MERIDIAN — Medical Knowledge Pipeline

MERIDIAN ingests, cleans, and prepares open-license medical knowledge for training. Sources include PubMed Open Access (34M papers), PhysioNet biosignal collections, and clinical note datasets. All open-license, all verifiable, all reproducible.

---

## Phases

### Phase 0 — Foundation ✓ Complete
- SPECTRA synthetic EMG generator — 6 gesture classes, ₹0 cost
- ARCHEX MedLM v0 trained and live on HuggingFace
- Public demo running at huggingface.co/spaces/archex-ai/archex-medlm-demo
- Training pipeline validated on free hardware

### Phase 1 — Medical Intelligence (Now → Month 3)
- [x] MERIDIAN v1 — 49,000 PubMed abstracts ingested and published
- Model becomes genuinely medically literate
- REST API launch — medical consultation at scale
- First paying customer

### Phase 2 — Signal Intelligence (Month 4–6)
- SPECTRA v2 — adds EEG, ECG synthetic generation
- ARCHEX Signal Model — multimodal text + biosignals
- Synthetic dataset sales to research labs
- arXiv preprint: *"BioSignal Tokenization for Multimodal Medical LLMs"*

### Phase 3 — Platform (Month 7–9)
- ARCHEX SDK v1.0 — bionic device integration in 3 lines of code
- Enterprise API for medical device companies
- On-prem deployment for hospitals
- First commercial bionic device integration

---

## The CUDA Parallel

NVIDIA did not win because their GPU was the best. They won because they gave developers CUDA for free — and once developers wrote CUDA code, the hardware lock-in was automatic and invisible.

**ARCHEX replicates this at the AI + bionics layer.**

```python
# This is what we are building toward.
# A prosthetic hand manufacturer writes this once.
# Then they cannot switch without rewriting everything.

from archex.sdk import BionicInterface, SignalEncoder

interface = BionicInterface(device="prosthetic_hand_v2")
encoder = SignalEncoder.from_pretrained("archex-ai/signal-encoder-emg-v1")

# Real-time EMG → intent → motor command
# 200ms end-to-end latency
# Running on a Raspberry Pi 4

for signal_window in interface.stream(sample_rate=250):
    intent = encoder.classify(signal_window)
    interface.actuate(intent)
```

---

## Getting Started

```bash
git clone https://github.com/archex-ai/archex
cd archex
pip install -e ".[all]"

# Run the medical QA demo
python -m archex.demo.medqa

# Start SPECTRA — synthetic biosignal generation
python -m archex.spectra.generators.emg --gestures 6 --samples 500
```

---

## Data Sources

All data used in ARCHEX training is open-license:

| Dataset | Domain | Size | License |
|---------|--------|------|---------|
| PubMed Open Access | Medical literature | 34M papers | NLM Open Access |
| MIMIC-IV | Clinical notes | 40K patients | PhysioNet Credentialed |
| PhysioNet Collections | Biosignals (ECG, EEG, EMG) | 4TB | Open Data Commons |
| BCI Competition IV | Motor imagery EEG | 9 subjects | Free research use |

---

## Roadmap

- [x] SPECTRA synthetic biosignal generator
- [x] ARCHEX MedLM v0 — open weights on HuggingFace
- [x] Live demo — huggingface.co/spaces/archex-ai/archex-medlm-demo
- [x] Training pipeline on free hardware
- [ ] MERIDIAN — PubMed ingestion pipeline
- [ ] ARCHEX MedLM v1 — medically literate
- [ ] Signal encoder pretrained weights
- [ ] ARCHEX Signal Model — multimodal
- [ ] SDK v1.0
- [ ] arXiv paper
- [ ] First bionic device integration

---

## Contributing

ARCHEX is built for the long term. If you are a developer, ML researcher, biomedical engineer, or someone who uses or builds bionic devices — you belong here.

```
Good first issues:  [good first issue]
Research bounties:  [research]
Medical domain:     [biomedical]
```

Join the [Discord](https://discord.gg/FeMCJZJTry) — introduce yourself.

---

## License

Apache 2.0 — use it, build on it, sell with it. The only thing we ask: if you extend the BioSignal Tokenization spec, contribute it back. We are building a standard, not a walled garden.

---

<div align="center">

**Built by humans, for humans who need more than what they were given.**

*ARCHEX — archex.ai*

</div>
