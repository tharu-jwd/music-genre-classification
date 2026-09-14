# Research constraints

The deep-learning contribution is the proposed shared encoder, supervised concept branches, joint objective, and gated fusion—not the data preparation process.

The experimental report must include:

- the direct CNN baseline;
- the descriptor-fusion baseline;
- the proposed model;
- controlled ablations;
- compute and latency comparisons;
- honest explainability validation;
- limitations and failed experiments where relevant.

Do not present handcrafted descriptors as learned concept embeddings. Do not present attention or gates alone as proof of explanation. Do not claim that the proposed model exists until its code runs and produces recorded results.

Do not present an LLM, agent workflow, or data-cleaning pipeline as the research contribution.

All reported results must use official `split-0`, validation-only model selection, and reproducible configurations suitable for accessible Colab or Kaggle hardware.
