
<div align="center">

# ARCHEX

**The open-source intelligence layer for human augmentation.**

*Origin. Command. Evolution.*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-ARCHEX-yellow)](https://huggingface.co/archex-ai)
[![Discord](https://img.shields.io/badge/Discord-Join%20the%20community-7289DA)](https://discord.gg/FeMCJZJTry)
[![arXiv](https://img.shields.io/badge/arXiv-BioSignal%20Tokenization-red)](https://arxiv.org)
[![HuggingFace](https://img.shields.io/badge/🤗-archex--medlm--v0-yellow)](https://huggingface.co/archex-ai/archex-medlm-v0)

---

*What CUDA did for GPU computing — ARCHEX does for the human body.*

</div>

---

## What is ARCHEX?

ARCHEX is an open-source AI stack purpose-built for **medical intelligence and bionic systems**.

We are not building another chatbot. We are building the infrastructure layer that will power the next generation of prosthetic limbs, neural interfaces, cochlear implants, and human augmentation devices — the same way NVIDIA's CUDA became the invisible backbone of every AI system on earth.

**Few people. One mission. The window is open right now.**

OpenAI is building for chat. Google is building for search. Nobody is building the AI brain for the 500 million people who live with limb loss, paralysis, hearing loss, or vision impairment.

That is what ARCHEX is for.

---

## The Stack

```
ARCHEX
├── archex-medlm/          # Medical language model (open weights)
│   ├── 7B-instruct/       # Fine-tuned on PubMed + MIMIC-IV
│   └── training/          # QLoRA recipes, data pipelines
│
├── biosim/                # Virtual environment for bionic AI training
│   ├── envs/              # MuJoCo + Gymnasium bionic environments
│   ├── generators/        # Synthetic EMG/EEG signal generation
│   └── sim2real/          # Sim-to-real transfer utilities
│
├── signal-encoder/        # 1D-CNN biosignal → LLM embedding bridge
│   ├── models/            # Pretrained signal encoders
│   └── tokenizer/         # BioSignal tokenization (our core innovation)
│
└── sdk/                   # ARCHEX SDK for bionic device integration
    ├── python/
    ├── cpp/               # For embedded/real-time systems
    └── examples/
```

---

## Why now?

The convergence is happening whether we build for it or not:

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

## Phases

### Phase 1 — ARCHEX MedLM (Now → Month 3)
- Phi-3-mini fine-tuned on 34M PubMed papers + MIMIC-IV clinical notes
- Open weights on HuggingFace (Apache 2.0)
- REST API at `api.archex.ai` — ₹0.001/1K tokens
- Benchmark target: beat GPT-3.5 on MedQA, PubMedQA, BioASQ

### Phase 2 — BioSim + Signal Model (Month 4–6)
- BioSim: open-source virtual environment for bionic AI training
- ARCHEX-Signal-13B: first multimodal model understanding text + biosignals
- Synthetic EMG/EEG dataset generation for FDA-compliant training data
- arXiv preprint: *"BioSignal Tokenization for Multimodal Medical LLMs"*

### Phase 3 — ARCHEX Platform (Month 7–9)
- ARCHEX SDK v1.0 — integrate bionic AI in 3 lines of code
- Enterprise API for medical device companies
- On-prem deployment package for hospitals (patient data never leaves)
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
encoder = SignalEncoder.from_pretrained("archex/signal-encoder-emg-v1")

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
git clone https://github.com/kephh/archex
cd archex
pip install -e ".[all]"

# Run the medical QA demo
python -m archex.demo.medqa

# Start the BioSim environment
python -m archex.biosim.envs.prosthetic_hand --render
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
| The Stack v2 | Code (Python, C, MATLAB) | 3TB | Stack Exchange ODbL |

---

## Roadmap

- [x] Repository structure
- [x] Data pipeline (PubMed + MIMIC ingestion)
- [ ] ARCHEX-MedLM-7B release
- [ ] BioSim v0.1 (EMG prosthetic hand environment)
- [ ] Signal encoder pretrained weights
- [ ] ARCHEX-Signal-13B release
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
