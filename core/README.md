# CORE

Model training pipeline for ARCHEX.

## Models
- **archex-medlm-v0** — TinyLlama 1.1B, synthetic EMG data, Phase 0
- **archex-medlm-v1** — Phi-2 2.7B, MERIDIAN 49K medical abstracts, Phase 1 (training)

## Training notebooks
- `archex_phase0_kaggle_v6.ipynb` — v0 training pipeline
- `archex_medlm_v1_training.ipynb` — v1 training pipeline (Phi-2 + MERIDIAN)

## Models on HuggingFace
- huggingface.co/archex-ai/archex-medlm-v0
- huggingface.co/archex-ai/archex-medlm-v1 (coming after TPU run)
