# Project 4 Reverse Engineering Report: EditLens

* **Project Name:** EditLens
* **Repository:** https://github.com/pangramlabs/EditLens
* **Project Category:** AI Training & Optimisation
* **Deadline:** April 3rd, 2026

## 1. Project Overview and Key Components

EditLens (ICLR 2026) quantifies how much AI editing is present in a piece of text. The repository provides training and inference pipelines that turn continuous edit-intensity scores into bucketed classification targets, then decode predictions back into smooth scores. Two reference setups are included: a RoBERTa-Large classifier and a Llama-3.2-3B QLoRA variant. Utilities cover text cleaning, bucket assignment, evaluation scripts, and standalone scoring metrics that mirror the paper’s cosine-distance and soft n-gram formulations. Example datasets and ready-to-run configs make it possible to reproduce the paper’s results or adapt the approach to new corpora.

### Repository Analysis Summary

```
EditLens/
|-- README.md                 - Primary README with paper link, setup, training, and inference commands.
|-- requirements.txt          - Python dependencies for training, inference, and scoring.
|-- configs/
|   |-- roberta.yaml          - Hydra config for the RoBERTa-Large bucketed classifier.
|   |-- llama.yaml            - Hydra config for the Llama-3.2-3B QLoRA classifier (4-bit, LoRA targets).
|-- scripts/
|   |-- train.py              - Hydra entry point to fine-tune models; supports LoRA/QLoRA, W&B logging, and metric computation.
|   |-- inference.py          - Runs inference on HuggingFace datasets, auto-inferring bucket count, and writes bucket/score columns.
|   |-- preprocess.py         - Text normalization (emoji, boilerplate removal), word counts, and score-to-bucket mapping utilities.
|   |-- eval/
|   |   |-- binary_eval.py    - Evaluation helpers for binary detection settings.
|   |   |-- ternary_eval.py   - Evaluation helpers for three-class setups.
|   |   |-- threshold.py      - Threshold sweeping utilities for calibration.
|   |-- scoring/
|       |-- cosine_distance.py - Standalone cosine-distance metric using sentence embeddings.
|       |-- soft_ngrams.py     - Soft n-gram overlap metric capturing novel phrase content.
|-- data/
|   |-- raid_10k.csv          - Sample RAID corpus slice for experiments.
|   |-- val.csv               - Validation split referenced by configs.
|   |-- test.csv              - General test set.
|   |-- test_enron.csv        - Enron-specific test subset.
|   |-- test_llama.csv        - Llama-generated test subset.
|   |-- human_detectors.csv   - Human-produced detector comparisons.
|   |-- nonnative_english.csv - Non-native English subset for robustness checks.
```

All ten project questions are answered separately in `Q1.md` through `Q10.md`; `final answers proj4.md` aggregates them in one place.

