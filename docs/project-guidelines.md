# DNN Research Project — Project Brief

*Compiled from the course briefing session transcript*

## 1. Overview

This semester's project follows the same team-based research model as last semester, but shifts focus away from the data science pipeline and onto **deep neural network (DNN) design**. The end goal is unchanged: produce a research contribution strong enough to be accepted at a conference.

**Why this format:** the course staff introduced this stricter, hands-on requirement specifically because prior cohorts leaned on downloaded code or senior students' work. The project is structured so that genuine, individual engagement with the internals of neural networks is unavoidable.

**Team structure:** groups of 5 students. Teams may carry over from last semester, or be reformed if a team isn't working well — the choice is up to the students.

## 2. Core Focus: Contribution Must Be in the DNN, Not the Data Science

Unlike the previous project, the data science process (cleaning, EDA, correlation analysis, etc.) is **not** where the contribution should happen. The required contribution is on the deep learning side, in at least one of the following areas:

- **A new neural network architecture/framework** — combining, customizing, or ensembling existing architectures for a new problem.
- **A new training/learning objective** — modifying what the network is optimized for (beyond plain accuracy), e.g. adding new loss terms.
- **A new feature-learning or representation-learning mechanism** — e.g. an unsupervised pretraining step for domains (like medical data) where labeled data is scarce but unlabeled data is abundant.
- **Efficiency/compression contributions** — reducing memory or compute (e.g. producing a smaller model for edge/mobile deployment, or for cases where data can't leave the device for privacy reasons).

At minimum, one of these dimensions must be addressed; addressing more than one is welcomed.

## 3. Dataset Requirements

- Reusing last semester's dataset is allowed and encouraged if convenient, since the team already understands it, has done correlation analysis, and has preprocessing in place — but it is **not mandatory**.
- Options for datasets:
  - Use the previous dataset as-is.
  - Expand it (more instances — e.g. more months of data — or more features/modalities).
  - Use a fresh dataset from a source like Kaggle or Hugging Face.
- **Size guidance:** ideally a few thousand instances, more is better. Real-world / self-collected datasets are often smaller, and that's acceptable — it's normal even at PhD level for domains like medical data (often under 10,000 instances). Working well with a smaller dataset is itself a valid contribution angle.
- **Target variable:** every dataset needs one. If it isn't obvious, teams should think creatively — e.g. a "tourist arrivals count" was used as a regression target for a group with that kind of data.
- A survey/quiz has already been assigned for teams to submit dataset details, since staff and PAs need this to judge dataset viability for a DNN project up front — teams should make sure this was completed.
- **Do not spend significant time on dataset creation.** Dataset work is not the graded focus this semester.

## 4. Step 1: Build a Baseline Model

For any dataset, after identifying the target variable, build a baseline model:

- **Match architecture to data type:**
  - Tabular data → MLP (multi-layer perceptron)
  - Images → CNN or Vision Transformer
  - Time series → Transformer-based models (LSTMs are usable but less common now; occasionally still competitive)
  - Graph data (e.g. social networks) → Graph Neural Network
  - Text → Transformer
- **Multi-modal data:** if the dataset spans multiple types (e.g. tabular + image + text), extract feature vectors from each modality using the appropriate specialized network, then combine the feature vectors — by concatenation, addition, or averaging — and feed the combined representation into a small decision network (a couple of thin layers). Combining modalities is preferable when possible, since it creates more room for later innovation.
- Baseline development, including basic hyperparameter tuning (choice of CNN/transformer variant, number of layers, etc.), should only take a couple of hours of actual training time once the code is in place.
- **Sanity check the dataset via the baseline:** if the baseline already achieves ~99% accuracy, the dataset is likely too easy to be worth using — pick something with more headroom for contribution.
- The baseline results are a required part of the **project proposal** deliverable.

## 5. Step 2: Architectural Contribution

Starting from the baseline (e.g. 80% accuracy on a classification task), the goal is to improve on it through a **deliberate architectural or methodological change** — not just hyperparameter tuning. This might mean changing:

- How input is processed
- How features are extracted or combined
- How the network is trained

This combination of existing architectures, deliberately modified, is what the course refers to as a "deep neural network framework." Inventing an entirely new architecture from scratch is explicitly **not** expected — that's PhD-level work. The expectation is to combine and adapt existing components in a way that has a real, measurable impact.

The resulting model should be compared against:
- The baseline
- Other existing solutions to similar problems
- Ablation studies (to be covered later in the course)

## 6. Explicit Constraints — What NOT to Do

- **No agentic / multi-agent solutions.** The objective is to learn the fundamentals of deep learning by working with the internals of neural networks directly — not to wrap an LLM in an orchestration layer.
- **No LLM fine-tuning.** LLMs may only be used as a **preprocessing step** — e.g. generating text embeddings to use as input features. The LLM itself is not trained or fine-tuned as part of the project.
- **No dependence on heavy or restricted compute.** The project should be scoped to run on freely available resources (Google Colab, Kaggle's free GPU tier, or personal hardware). No department cloud/GPU cluster access will be provided. If compute is a genuine constraint, addressing it (e.g. via a smaller, more efficient model) is itself a valid contribution — not an excuse.
- **No last-minute excuses about training time.** Training deep models can take hours or days, and problems like non-converging or oscillating loss are expected. This is flagged explicitly upfront so there's no room for "we ran out of time" as an excuse later.

## 7. Research Paper & Experiments

The paper structure mirrors last semester's, with one key difference: the **methodology** section describes the proposed framework or training modification (instead of a data science process). The **experimental section carries more weight** this time and should include:

- Baseline vs. proposed model comparison
- Testing across multiple similar datasets where possible (to demonstrate generalizability)
- Application studies
- Comparative analysis against other existing solutions
- Hyperparameter tuning experiments
- Ablation studies
- Any other experiments that help demonstrate the value of the proposed approach

Thoroughness here directly affects acceptance chances — papers in this space are commonly rejected for missing standard experiments.

## 8. Deliverables & Timeline

| Deliverable | Requirement |
|---|---|
| **Project Proposal** | Dataset and problem finalized; baseline model implemented with results; proposed contribution and experimental plan described |
| **Short Paper** | Proposed approach implemented with initial results (worse than baseline is acceptable at this stage — the point is having *something* implemented) |
| **Final Research Paper** | All planned experiments completed; target outcome is conference acceptance |

*(Exact calendar deadlines were referenced in the session as being communicated separately — confirm specific dates with the course announcement/schedule.)*

Grading/marks allocation follows the same scheme as last semester's project.

## 9. Notes on Support

- This is the first time this specific project format has been run, so there's no senior/prior-cohort reference project to draw from — this cohort's output will become the reference for future years.
- Slide decks for concepts not yet formally covered in lectures (e.g. attention mechanisms) will be released early so teams aren't blocked waiting for the lecture schedule to catch up.
- Course staff explicitly invited ongoing feedback on the project's difficulty and are open to adjusting the process if legitimate issues come up.

## 10. Quick Contribution Checklist

- [ ] Dataset finalized (reused, expanded, or new) and target variable identified
- [ ] Baseline model built and evaluated (confirms dataset isn't trivially easy)
- [ ] Contribution area chosen (architecture / learning objective / representation learning / efficiency)
- [ ] Architectural or methodological change implemented on top of baseline
- [ ] Comparison plan defined: baseline vs. proposed vs. other existing solutions vs. ablations
- [ ] Compute plan confirmed to run on free-tier resources (Colab/Kaggle/local)
- [ ] Any LLM usage scoped strictly to preprocessing (embeddings), not fine-tuning
- [ ] No agentic/multi-agent framing used as the "contribution"
