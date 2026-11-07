# Coreset Selection for Claim Verification

**M.Tech Thesis · Phase 1 · IIT Madras Zanzibar**

How much labelled training data can we remove before claim verification performance drops? And does a more expensive selection method earn back its cost during fine-tuning?

This project compares **stratified random sampling** with a **HealthVer adaptation of TAGCOS** across three language models and four training-data budgets. Each model has its own full-data LoRA reference, and all runs are evaluated on the same HealthVer test split.

**1 public dataset · 3 models · 4 subset budgets · 27 training runs**

## Notebooks

| Model | Full-data LoRA | Stratified random | TAGCOS adaptation |
| :--- | :---: | :---: | :---: |
| Qwen2.5-0.5B-Instruct | [m1_1](Notebooks/m1_1.ipynb) | [m1_2](Notebooks/m1_2.ipynb) | [m1_3](Notebooks/m1_3.ipynb) |
| SmolLM2-360M-Instruct | [m2_1](Notebooks/m2_1.ipynb) | [m2_2](Notebooks/m2_2.ipynb) | [m2_3](Notebooks/m2_3.ipynb) |
| TinyLlama-1.1B-Chat-v1.0 | [m3_1](Notebooks/m3_1.ipynb) | [m3_2](Notebooks/m3_2.ipynb) | [m3_3](Notebooks/m3_3.ipynb) |

## Experimental setup

The [HealthVer dataset](https://github.com/sarrouti/HealthVer) contains labelled claim–evidence pairs. Given a claim and an evidence passage, the model predicts one of three verdicts: **Supports**, **Refutes**, or **Neutral**.

| Split | Claim–evidence pairs | Role |
| :--- | ---: | :--- |
| Train | 10,590 | Subset selection and LoRA fine-tuning |
| Validation | 1,917 | Checkpoint selection by macro-F1 |
| Test | 1,823 | Final evaluation on a fixed split |

- **Full-data reference:** Fine-tune on all 10,590 training pairs.
- **Stratified random:** Sample within verdict classes to approximately preserve the training label distribution.
- **TAGCOS adaptation:** Use a short LoRA warm-up to estimate per-example update features, cluster examples by these features, and select representatives using orthogonal matching pursuit (OMP).

Within each model, the final runs start from the same LoRA initialization and train for three epochs. The four subset budgets retain **530**, **1,059**, **2,648**, and **5,295** training pairs, respectively.

## Phase 1 results

**HealthVer test macro-F1 (%)**. Bold marks the better subset method at each model and budget. The full-data column is a separate reference for each model.

| Model | Data retained | Stratified random | TAGCOS adaptation | Full-data reference |
| :--- | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B | 5% | 54.85 | **55.76** | 79.16 |
| Qwen2.5-0.5B | 10% | 61.58 | **64.02** | 79.16 |
| Qwen2.5-0.5B | 25% | 67.04 | **70.64** | 79.16 |
| Qwen2.5-0.5B | 50% | **75.02** | 73.19 | 79.16 |
| SmolLM2-360M | 5% | **30.26** | 27.36 | 68.62 |
| SmolLM2-360M | 10% | **38.33** | 29.14 | 68.62 |
| SmolLM2-360M | 25% | 43.39 | **47.02** | 68.62 |
| SmolLM2-360M | 50% | **59.45** | 58.70 | 68.62 |
| TinyLlama-1.1B | 5% | **44.32** | 42.51 | 79.67 |
| TinyLlama-1.1B | 10% | **53.26** | 49.89 | 79.67 |
| TinyLlama-1.1B | 25% | **65.64** | 65.47 | 79.67 |
| TinyLlama-1.1B | 50% | **75.51** | 73.78 | 79.67 |

**What we found**

- No evaluated subset matched its model's full-data macro-F1. Stratified random sampling performed better at the 50% budget for all three models.
- TAGCOS achieved higher macro-F1 in **4 of 12** comparisons: Qwen2.5 at 5%, 10%, and 25%, and SmolLM2 at 25%.
- Per-example feature extraction accounted for most of TAGCOS selection time. Shorter final fine-tuning did not establish a saving in total selection-plus-training time.

These are preliminary results from **one dataset and one training seed (42)**. They show where the methods differed under this setup; they do not establish which method is generally better.

## Reproduce on Kaggle

1. **Start with [`m1_1.ipynb`](Notebooks/m1_1.ipynb).** It prepares HealthVer from a pinned source revision and runs the Qwen full-data reference. The notebooks were designed for a Kaggle GPU environment.
2. **Save and attach upstream outputs.** The notebooks write artifacts under `/kaggle/working` and load saved inputs under `/kaggle/input`. Attach the required outputs as Kaggle inputs and update hard-coded paths where they differ in your workspace.
3. **Run the Qwen subset notebooks before their SmolLM2 and TinyLlama counterparts.** The later stratified random runs reuse the saved subset IDs, and the later TAGCOS notebooks depend on artifacts from the Qwen TAGCOS run. For each model, run its full-data notebook before its two subset notebooks.

When comparing runs, retain the split definitions, subset IDs, model revisions, seeds, and timing records. Report selection time separately from final fine-tuning time.

## Next research question

At the same training-pair budget, is it more useful to keep evidence passages that lead to **different verdicts for the same claim**, or to cover **more distinct claims**? The next phase will test this on annotated evidence contrasts and measure both verification performance and total compute time.

## References

- Sarrouti et al. (2021). [Evidence-based Fact-Checking of Health-related Claims](https://aclanthology.org/2021.findings-emnlp.297/).
- Hu et al. (2022). [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
- Zhang et al. (2025). [TAGCOS: Task-agnostic Gradient Clustered Coreset Selection for Instruction Tuning Data](https://aclanthology.org/2025.findings-naacl.264/).
