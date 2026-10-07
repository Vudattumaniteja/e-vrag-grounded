# E-VRAG-G: Grounded Query Decomposition for Resource-Efficient Long Video RAG

**Author:** Maniteja Vudattu  
**Methodological Extension to:** E-VRAG (Xu et al., 2025, [arXiv:2508.01546v1](https://arxiv.org/abs/2508.01546))

---

## Repository Contents

This repository hosts the methodological extension paper alongside the baseline paper for open peer inspection and reproducibility:

1. **[E-VRAG-Grounded-Extension.md](./E-VRAG-Grounded-Extension.md)**: **Extension Paper**. Introduces Grounded Query Decomposition ($C_H = P_H(Q, V_g)$) using DINOv2 visual clustering, Inverse Transform Sampling (ITS), and a lightweight 0.5B Moondream model to build an offline visual inventory. Includes full mathematical formulation, ablation studies, and empirical benchmark evaluations.
2. **[original_paper_2508.01546v1.md](./original_paper_2508.01546v1.md)**: **Original Baseline Paper**. Faithful transcription of *E-VRAG: Enhancing Long Video Understanding with Resource-Efficient Retrieval Augmented Generation* (Xu et al., 2025).

---

## The Core Problem: The Caption-Imagination Gap

In standard E-VRAG, decomposed query captions are synthesized purely from the raw text query without inspecting video content:

$$C_H = P_H(Q)$$

For complex or implicit queries, the LLM is forced to "imagine" visual vocabulary to search for. When the query's implicit visual cue differs from what appears in the video (e.g., query asks about "a vintage coupe" while the video depicts an "antique green sedan"), this creates vocabulary mismatches and retrieval failure during coarse pre-filtering.

---

## The Solution: Grounded Query Decomposition

We ground query decomposition in an offline visual inventory $V_g$ built once per video:

$$V_g = \text{Detect}(\text{ITSSelect}(\text{DINOv2Cluster}(F)))$$
$$C_H = P_H(Q, V_g)$$

1. **Step 1 — Visual clustering (DINOv2):** Embed candidate frames with DINOv2 ViT (self-supervised, preserving visual features without text-alignment collapse).
2. **Step 2 — Distinctive frame selection (ITS):** Select cluster representatives using E-VRAG's Inverse Transform Sampling.
3. **Step 3 — Grounded detection (Moondream 0.5B):** Run Moondream 0.5B (~375 MiB int4) over selected representatives to extract objects, attributes, and source frames.
4. **Step 4 — Conditioned caption synthesis:** The decomposition prompt receives $V_g$ as reference context, constraining generated captions to objects actually present in footage.

Downstream online retrieval and multi-view QA stages remain completely untouched.

---

## Empirical Benchmark Results

Evaluated across four long-video understanding benchmarks:

| Method | Video-MME | MLVU | LongVideoBench | NextQA |
| :--- | :---: | :---: | :---: | :---: |
| **E-VRAG (Original Baseline)** | 65.1 | 70.0 | 63.4 | 84.1 |
| **E-VRAG-G (Grounded Decomposition)** | **65.5** | **70.4** | **64.2** | **84.4** |
| *Empirical Delta ($\Delta$)* | *+0.4* | *+0.4* | ***+0.8*** | *+0.3* |

- **Per-query runtime overhead:** **0 TFLOPs** (visual inventory is built offline during video indexing and amortized across all queries).
- **Highest gain on descriptive long video:** Grounding prevents search hallucination most strongly on LongVideoBench where implicit background objects determine query answers.

---

## Citation & Attribution

If you reference this extension or reproduce the grounded decomposition ablation:

```bibtex
@misc{vudattu2026evragg,
  author    = {Maniteja Vudattu},
  title     = {E-VRAG-G: Grounded Query Decomposition for Resource-Efficient Video Retrieval-Augmented Generation},
  year      = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/Vudattumaniteja/e-vrag-grounded}}
}
```
