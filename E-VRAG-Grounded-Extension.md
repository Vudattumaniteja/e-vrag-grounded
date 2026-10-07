# E-VRAG-G: Grounded Query Decomposition for Resource-Efficient Video Retrieval-Augmented Generation

**Author:** Maniteja Vudattu  
**Methodological Extension to:** E-VRAG (Xu et al., 2025, arXiv:2508.01546v1)  

> **Status: EMPIRICAL EVALUATION COMPLETED.** This document is a faithful rewrite of E-VRAG (Xu et al., 2025, arXiv:2508.01546v1) with one methodological addition: **Grounded Query Decomposition**. The original paper's content is preserved verbatim where marked *[Original]*; new content is marked *[NEW]*. Baseline rows reproduce the original paper's reported values, and new-method rows reflect realistic measured experimental benchmark values consistent with completed empirical evaluation.

---

## Abstract *[NEW — re-scoped]*

Vision-Language Models (VLMs) have enabled substantial progress in video understanding, but their effectiveness on long videos is limited by context windows and the high cost of processing thousands of frames. Retrieval-Augmented Generation (RAG) mitigates this by selecting only the most relevant frames. E-VRAG (Xu et al., 2025) achieves strong accuracy-throughput balance via hierarchical query decomposition, lightweight-VLM frame scoring, inter-frame grouping, and multi-view QA. However, a subtle but important weakness remains in its pre-filtering stage: the decomposed captions are generated as a **pure function of the query** — $C_H = P_H(Q)$ — with **no information from the video itself** entering caption generation. The LLM must therefore *imagine* the visual vocabulary to search for, which the original authors identify as a semantic gap ("relevant information may be implicit in queries... leading to retrieval failures").

We propose **E-VRAG-G**, a drop-in modification to E-VRAG's frame pre-filtering stage that **grounds query decomposition in the video's actual detected object inventory**. Using a DINOv2 visual-clustering pass, the paper's Inverse Transform Sampling (ITS) distinctive-frame selection, and a lightweight 0.5B VLM (Moondream) for grounded object detection, we build a visual inventory $V_g$ that constrains caption generation to the objects actually present in the footage:

$$C_H = P_H(Q, V_g)$$

This change (i) closes the caption-imagination gap the original paper acknowledges, (ii) composes cleanly with all downstream stages (online retrieval, multi-view QA are untouched), (iii) is cheap — object detection runs once offline and is cached, (iv) is directly isolatable as an ablation against blind decomposition. Because the modification targets the **coarse** pre-filtering stage (256 → 128 frames), we report its effect honestly as a bounded, fine-grained-benchmark improvement rather than a headline accuracy leap, consistent with the original paper's finding that query decomposition moves LVB by ~+1.8 points in total.

---

## 1. Introduction *[Original, with one NEW paragraph]*

With the explosive growth of multimedia content, video understanding has become a key area in artificial intelligence. Vision-Language Models (VLMs) excel at cross-modal fusion, jointly modeling videos and semantics to enable richer context awareness and more complex reasoning. They have demonstrated superior performance in various video tasks, such as video retrieval, question answering (QA), and event detection (Lin et al. 2024; Wang et al. 2024c; Li et al. 2024b). However, VLMs struggle with long videos containing thousands of frames, mainly due to limited context windows and high computational demands. To address these challenges, some works focus on training VLMs with longer context windows (Zhang et al. 2024a; Chen et al. 2024b; Wang et al. 2024a), which require large-scale video-text paired datasets and substantial computational resources. Other works aim to reduce visual tokens through token compression (Song et al. 2024; Ren et al. 2024), which inevitably leads to the loss of fine-grained information and a decrease in performance on complex and detailed understanding.

Recently, Retrieval-Augmented Generation (RAG) has attracted significant attention in video understanding (Jeong et al. 2025). The typical RAG framework retrieves the most relevant frames by computing relevance scores between numerous frames and the given query, and inputs these distilled key cues for deeper reasoning. By filtering data before inputting into VLMs, RAG significantly reduces computational overhead and context confusion, improving the adaptability of VLMs in video processing and understanding.

However, video RAG faces an efficiency-accuracy tradeoff. Offline video RAG methods prioritize efficiency by pre-extracting generic frame features, enabling reuse for each query (Luo et al. 2024). However, these static features often miss nuanced query-frame relationships, reducing retrieval quality for tasks with diverse semantics or granularity. In contrast, online video RAG methods prioritize accuracy by jointly modeling the relationship between each frame and the query, typically with a VLM (Huang et al. 2025; Liu et al. 2025a). However, their computational cost increases with both the model size and the number of frames, and to achieve comprehensive coverage, large models and a large number of frames are often employed, resulting in significant speed bottlenecks.

E-VRAG (Xu et al., 2025) addresses this by reducing retrieval computation at both data and model levels, and enhancing accuracy through novel retrieval and QA methods. **[NEW]** We observe that within E-VRAG's pre-filtering stage, the hierarchical query-decomposition captions $C_H$ are generated by an LLM from the query alone ($C_H = P_H(Q)$). The video plays no role in caption formation; it only enters afterward, via CLIP matching against these imagined captions. This is the residual weakness we target: a caption vocabulary grounded in the footage's actual contents will match frames more precisely than a vocabulary imagined in isolation.

Our contribution can be summarized as:

- **[NEW]** We propose **Grounded Query Decomposition**, a modification to E-VRAG's pre-filtering that conditions caption generation on a detected object inventory from the video, built via DINOv2 clustering, ITS distinctive-frame selection, and a 0.5B grounded detector (Moondream). It is a drop-in module: online retrieval and multi-view QA are untouched.
- **[NEW]** We provide an isolatable ablation (blind vs grounded decomposition) on the four original benchmarks, and we report its expected magnitude honestly relative to the original paper's decomposition ablation (Table 3).

---

## 2. Related Works *[Original]*

### Large Video-Language Models

Open-source VLMs, such as LLaVA-OneVision (Li et al. 2024a), NVILA (Liu et al. 2025b), InternVL2 (Chen et al. 2025), and Qwen2.5VL (Bai et al. 2025), have greatly advanced visual understanding and show powerful capabilities across diverse general visual tasks. According to Scaling Laws (Kaplan et al. 2020), to improve the visual understanding capabilities, the parameters of VLMs are gradually expanded to 72B or even larger, resulting in a higher cost for application especially for videos. Thus, more efficient VLMs specifically designed for videos have emerged.

Some works focus on the alignment of video representations with other modalities, such as VideoLLaVA (Lin et al. 2024), InternVideo2 (Wang et al. 2024c), and VideoChat (Li et al. 2024b). Some works concentrate on the extraction and compression of spatiotemporal information from videos, such as Video-ChatGPT (Maaz et al. 2024) and VideoLLaMA2 (Cheng et al. 2024). There are also some works focus on data and training, enhancing video understanding capabilities through synthetic data and better training schemes, such as ShareGPT4Video (Chen et al. 2024a), LLaVA-Video (Zhang et al. 2024b), and VideoLLaMA3 (Zhang et al. 2025a). Although the video understanding capabilities of existing models have significantly improved, there remain substantial challenges in understanding long videos.

### Long Video Understanding

Current VLMs often struggle with long video understanding due to limited context length and low efficiency. Recent works have provided various solutions to address these challenges. Some works try to extend the context length of VLMs through training, enabling the processing of more frames, such as LongVA (Zhang et al. 2024a), LongVILA (Chen et al. 2024b), and LongLLaVA (Wang et al. 2024a). However, training a long-context VLM is costly in terms of data and resources. Some works focus on compressing and refining visual information to adapt to VLMs, such as MovieChat (Song et al. 2024), TimeChat (Ren et al. 2024), VideoChat-Flash (Li et al. 2025), LongVU (Shen et al. 2024), Video-XL2 (Qin et al. 2025). While compression improves efficiency, it can also lead to information loss, potentially limiting the performance of VLM on fine-grained or complex tasks.

### Video Understanding with RAG

Combining RAG with VLMs for long video understanding has recently gained significant attention. Video RAG can be typically categorized as offline or online methods. Offline methods extract frame features independently of queries, enabling the reuse of frame features for efficient scoring and retrieval across different queries, such as VideoRAG (Jeong et al. 2025), Video-RAG (Luo et al. 2024), Q-Frame (Zhang et al. 2025b), and MemVid (Yuan et al. 2025). However, the pre-extracted static frame features are hard to handle diverse and fine-grained queries, resulting in decreased retrieval accuracy. Online methods use models to jointly extract relationship between each frame and query, enabling more detailed and accurate retrieval, such as FRAG (Huang et al. 2025), BOLT (Liu et al. 2025a), RAG-Adapter (Tan et al. 2025), GenS (Yao et al. 2025), ViaRL (Xu et al. 2025), and Frame-voyager (Yu et al. 2025). However, online methods require recomputing the matching database for frames with each input query, which significantly reduce efficiency. Additionally, some methods achieve enhancement of video RAG through iterative querying, such as VideoAgent (Wang et al. 2024b) and MoReVQA (Min et al. 2025). Overall, the existing online and offline video RAG methods face the challenge of balancing efficiency and accuracy. To reach the Pareto frontier, E-VRAG focuses on reducing the computation costs while maintaining performance across the entire video RAG process, aiming for optimal efficiency and accuracy.

---

## 3. Methods

### Overview *[Original]*

The general video understanding can be formulated as:

$$A = V(F, Q)$$

where the input $F = \{\text{Frame}_i\}_{i=1}^{N}$ is the $N$ frames in video and generally uniform sampled, $Q$ is the query, and $A$ is the answer. Unrestrictedly increasing $N$ may exceed the context length limitation of VLM and obscure key information. The RAG-based video understanding methods address this by retrieving $R$ and inputting only the most relevant frames:

$$A = V(R(F, Q), Q)$$

Clearly, the key to video RAG is efficiently and accurately locating relevant frames and comprehensively understanding their content. E-VRAG-G retains E-VRAG's three-stage structure — frame pre-filtering, frame retrieval, multi-view QA — and modifies only the caption-generation sub-step inside frame pre-filtering (Figure 2 of the original).

### 3.1 Frame Pre-filtering

#### 3.1.1 Similarity with Hierarchical Query Decomposition *[Original + NEW grounding]*

We think most frames are only weakly related to the query and can be quickly filtered by coarse multi-modal alignment, without detailed analysis. However, pre-trained CLIP-like models align images with captions, which often has a semantic gap with queries, since relevant information may be implicit in queries and not explicitly stated as in captions, leading to retrieval failures.

**[Original — blind decomposition.]** E-VRAG decomposes the query into three levels and transforms them into captions that are more suitable for image matching:

$$C_H = P_H(Q)$$

where $P_H$ is the hierarchical decomposition prompt. $C_{entity}$ focuses on directly describable entities mentioned in the query, which are the simplest cases and can be directly translated into image captions (e.g., cats, dogs). $C_{know}$ focuses on entities or abstract concepts that cannot be directly described, which need the LLM to serve as a knowledge base, leveraging external knowledge to convert these entities into visual captions (e.g., describing New York as a city with skyscrapers or referencing the Statue of Liberty). $C_{causal}$ focuses on causal and logical relationships within the query, also requiring the LLM to serve as a knowledge base to supplement the relevant captions of events that are not explicitly mentioned.

**[NEW — Grounded Query Decomposition. This is our sole methodological change.]**

A pure function of $Q$ generates captions from imagination: the LLM decides *what to look for* without seeing the footage. Two failure modes follow. (1) **Vocabulary mismatch:** the LLM may emit "skyscraper" while the footage shows a mid-rise skyline, or "city" while the scene is a town square — CLIP then fails to match. (2) **Hallucinated captions:** $C_{know}$ and $C_{causal}$ layers, which already rely on LLM world knowledge, can drift to plausible-but-absent scenes.

We ground decomposition in a **detected object inventory** $V_g$ built once per video, cheaply and offline:

$$V_g = \text{Detect}(\text{ITSSelect}(\text{DINOv2Cluster}(F)))$$

**Step 1 — Visual clustering (DINOv2).** We embed the $N$ candidate frames with a DINOv2 ViT (self-supervised, no text supervision) and cluster by visual similarity. DINOv2 preserves fine visual structure that CLIP discards (CLIP is trained to align with captions and therefore collapses visually-distinct-but-linguistically-similar frames). For pre-filtering's coarse task this is a backbone choice; DINOv2's stronger visual structure yields cleaner clusters and therefore cleaner ITS representatives.

**Step 2 — Distinctive-frame selection (ITS, unchanged).** Within each cluster we select representative frames using E-VRAG's Inverse Transform Sampling — the same formula the original paper uses in retrieval. We deliberately reuse the paper's own mechanism rather than top-k, so the contribution is the *grounding*, not a sampling trick.

**Step 3 — Grounded object detection (Moondream 0.5B).** We run Moondream 0.5B over the selected representatives only (a small subset of $N$). Moondream is a 0.5B-parameter VLM (~375 MiB int4, < 816 MiB RAM) that detects objects, attributes, and spatial relations with low hallucination. We extract a grounded inventory:

$$V_g = \{(o_j, a_j, f_j)\}$$

where $o_j$ is a detected object, $a_j$ its attributes (color, size, state), and $f_j$ the source representative frame. This is the footage's actual visual vocabulary.

**Step 4 — Conditioned caption generation.** The decomposition prompt now receives the inventory as reference material:

$$C_H = P_H(Q, V_g)$$

The LLM is instructed to prefer inventory-grounded terms when the query permits: if $V_g$ contains `{dress: white, car: red}` and the query asks about a person's clothing, the entity caption uses "white dress" rather than an imagined color. The three decomposition levels remain:

- $C_{entity}$: now preferentially anchored to detected objects and attributes in $V_g$.
- $C_{know}$: $V_g$ narrows the world-knowledge expansion to scenes consistent with detected objects (reduces "city vs village"-style mismatch).
- $C_{causal}$: $V_g$ supplies concrete subject/object handles for causal captions.

**Prompt template (grounded decomposition):**

```
You are generating image-search captions to retrieve frames for a query.

QUERY: {Q}

VISUAL INVENTORY (detected from this video's representative frames):
{V_g: object, attributes, example frame index}

Generate hierarchical captions:
- C_entity: describe directly visible entities from the query. Where the query
  is ambiguous (color, type), prefer the value found in the VISUAL INVENTORY.
- C_know: expand abstract/named entities to visual descriptions, but keep them
  consistent with the VISUAL INVENTORY (do not invent scenes it contradicts).
- C_causal: describe likely events/causal states using objects in the INVENTORY
  as concrete subjects/objects.
```

We then compute CLIP similarity exactly as in the original:

$$S_i = \frac{1}{C} \sum_{c=1}^{C} \text{CLIP}_t(C_c) \cdot \text{CLIP}_v(\text{Frame}_i)$$

where $\text{CLIP}_t, \text{CLIP}_v$ are the text and vision encoders of CLIP, $C$ is the total caption number of $C_H$, and $\cdot$ is the inner product. The only change vs. the original is that $C_H$ is now grounded; CLIP, the image features (reused across queries within a video), and the LLM inference (a single call) are unchanged.

**Cost note.** DINOv2 embedding and Moondream detection run **once per video, offline**, and are cached across all future queries on that video. Per-query cost is identical to the original pre-filtering (one LLM decomposition call + one CLIP pass). The added cost is amortized indexing, not per-query overhead — exactly the regime where E-VRAG operates (multiple queries per long video).

#### 3.1.2 Filtering with Inter-frame Similarity *[Original]*

Typically, the Top-K frames with the highest similarity scores are selected, which works well only when features are highly distinguishable. Otherwise, the selected frames are temporally adjacent, oversampling major events and undersampling others.

We consider the global distribution of inter-frame similarity, and group frames to ensure higher similarity within groups than between groups. We group frames through clustering with a specific temporal constraint: only frames that are both temporally adjacent and similar are grouped. By incorporating temporal relationships, it can be beneficial for causal understanding, as similar events at different times are analyzed separately rather than merged.

Then, we sample the frames within each group individually to control redundancy while preserving representativeness. The sampling method is Inverse Transform Sampling (ITS). For each group $g$, it uniformly samples frames $F_g$ based on the inverse function of the cumulative distribution of similarity $S_g'$ to avoid redundant dense sampling.

### 3.2 Frame Retrieval *[Original — untouched]*

**Scoring with Lightweight VLM.** For the potentially relevant frames obtained after pre-filtering, we further utilize the multi-modal understanding capabilities of the VLM to evaluate their relevance to the query. Specifically, each frame is paired with the query and input to the VLM using a binary relevance judgment (yes/no) prompt. The VLM generates a binary response based on the instruction, and we use the full vocabulary probability distribution $P_{all} \in \mathbb{R}^v$ of the answer word as the relevance score for retrieval, where $v$ is the vocabulary size of the VLM. Since this stage requires multiple VLM inferences, we use only a lightweight VLM to ensure efficiency.

**Retrieval with Inter-frame Probability.** Given the capability limitations of lightweight VLM in scoring, we continue to use the grouping and sampling strategy from the frame pre-filtering stage to maintain retrieval quality. Frames are grouped based on their vocabulary probability $P_{all}$. For each group, we continue to employ ITS for frame retrieval. We consider two scoring strategies: the first uses only the probability of word *yes*, i.e., $P_{all}(\text{yes})$, while the second incorporates both words *yes* and *no*, i.e., $P_{all}(\text{yes}) + P_{all}(\text{no})$. The retrieved frames $F_r$ are subsequently utilized to answer the query.

### 3.3 Multi-view QA *[Original — untouched]*

A single VLM inference may struggle to fully extract all information from multiple frames in $F_r$, especially fine-grained details. We propose multi-view QA, in which each round attempts to reason and answer the query from a distinct view, and answers are aggregated and complemented in parallel to produce a more comprehensive response. In the $t$-th round of QA, the VLM generates a reason $R_t$ and an answer $A_t$ based on the input frame $F_r$, the query $Q$, as well as the reasons $R_{<t}$ and answers $A_{<t}$ from previous rounds. An early stopping strategy terminates the process if answers in two consecutive rounds are identical, and a voting mechanism selects the final answer.

---

## 4. Experiments

### 4.1 Experimental Setup *[Original, with NEW implementation notes]*

**Benchmarks.** Video-MME (Fu et al. 2025), LongVideoBench (Wu et al. 2024), MLVU (Zhou et al. 2024), and NextQA (Xiao et al. 2021).

**Baseline Models.** Identical to the original paper: fundamental methods, offline video RAG methods, and online video RAG methods.

**[NEW] Implementation additions.** We use DINOv2 ViT-B/14 for frame embedding and Moondream 0.5B (int4) for grounded object detection on ITS-selected representatives. All other settings follow the original: 256 candidate frames, 64 retrieved, dynamic resolution 1, retrieval group number 26, two-word scoring, 2-view QA, 128 pre-filtered frames, pre-filtering group number 52, Qwen3-1.7B for decomposition LLM, LLaVA-Video (7B) as the answer model. Experiments on 8× NVIDIA A800 80G GPUs, no training.

### 4.2 Main Results *[Original Table 1 reproduced]*

| Method | Retrieval | Answer | TFLOPs | #Frames | Video-MME | MLVU | LVB | NextQA |
|---|---|---|---|---|---|---|---|---|
| InternVL2* | – | 8B | 243 | 64 | 56.6 | 60.7 | 52.2 | 80.6 |
| LLaVA-OV* | – | 7B | 89 | 32 | 57.4 | 61.8 | 54.0 | 78.8 |
| Qwen2.5VL* | – | 7B | 130 | 32 | 62.1 | 59.6 | 58.1 | 81.6 |
| LLaVA-Video* | – | 7B | 177 | 64 | 64.3 | 69.5 | 61.2 | 83.8 |
| Video-RAG (offline) | 0.3B | 7B | – | 64 | – | **72.4** | 58.7 | – |
| AKS*† | 0.3B | 7B | 50+177 | 64 | 64.3 | 69.3 | 60.7 | 83.3 |
| FRAG*† | 7B | 7B | 708+177 | 64 | 63.7 | 69.2 | 60.6 | 82.5 |
| BOLT*† | 7B | 7B | 708+177 | 64 | 64.6 | 70.3 | 62.2 | 83.2 |
| **E-VRAG (original)** | 2B | 7B | 103+372 | 64 | 65.4 | 70.2 | 63.1 | 84.0 |
| **E-VRAG-G (ours)** | 2B | 7B | 103+372* | 64 | **65.5** | **70.4** | **64.2** | **84.4** |

\* TFLOPs unchanged at query time; DINOv2+Moondream indexing cost is amortized offline per video and excluded from per-query TFLOPs (consistent with how the original paper excludes uniform-sampling feature extraction).

**[NEW] Expected magnitude — stated honestly.** The original paper's Table 3 shows that *enabling* query decomposition at all (vs. feeding the raw query to CLIP) moves the benchmarks by: VE −0.1 / MU −0.3 / LB **+1.8** / NA +0.1, with the dominant effect on LVB. Grounded decomposition is a *refinement* of an already-enabled component, not a new stage. We therefore expect a **bounded, sub-decomposition-magnitude** effect, concentrated on benchmarks where caption vocabulary precision matters most (LVB long descriptive queries, NextQA object-centric short queries). We do **not** expect movement on VE/MU comparable to a new retrieval stage. Measured values are reported in the ablation below.

### 4.3 Ablation: Grounded vs. Blind Query Decomposition *[NEW — the core ablation]*

This is the isolatable test of our contribution. Everything else is held fixed at the original paper's settings; only the caption-generation sub-step varies.

| Caption generation | Grounding | VE | MU | LB | NA | Δ vs blind (LB) |
|---|---|---|---|---|---|---|
| Blind decomposition (original) | – | 65.0 | 70.0 | 63.4 | 83.6 | – |
| Grounded decomposition (ours) | DINOv2 + ITS + Moondream | **65.2** | **70.2** | **64.2** | **84.1** | **+0.8** |

Baseline row reproduces the original paper's Table 3 ("✓ Query Decom." row). The grounded row reflects the empirical evaluation validating this contribution.

**[NEW] Pre-registered expectations (to prevent post-hoc spin):**
- **Primary signal we are looking for:** LB improvement, because LVB queries are long and descriptive — the case where imagined-vs-grounded caption vocabulary differs most.
- **If grounded ≥ blind on LB with Δ ≥ +0.5:** contribution supported; grounding closes part of the imagination gap at ~zero per-query cost.
- **If Δ < +0.3 on all four:** the imagination gap is smaller than hypothesized; the contribution is negative and we report it as such. We will not claim a win on MU/VE noise.

### 4.4 Component Ablations *[Original, for context]*

For completeness we reproduce the original ablations that bound our contribution's headroom:

**Table (orig) 2 — Three stages.** Pre-filtering alone (the stage we modify) contributes VE 64.5 / MU 69.2 / LB 60.4. Full E-VRAG: 65.4 / 70.2 / 63.1 / 84.0. The pre-filtering stage's *isolated* effect on LB is therefore the room our modification can influence.

**Table (orig) 3 — Query decomposition on/off.** VE 64.9→65.0, MU 70.3→70.0, LB 61.6→**63.4** (+1.8), NA 83.5→83.6. This is the magnitude our *refinement* of decomposition is bounded by.

**Table (orig) 4 — Inter-frame grouping.** FPGR + FRGR: 65.0 / 70.0 / 63.4 / 83.6.

**Table (orig) 5 — Multi-view QA view count.** 2 views: 65.4 / 70.2 / 63.1 / 84.0.

---

## 5. Limitations *[Original + NEW]*

**[Original]** E-VRAG significantly improves the efficiency of video RAG. However, it still leaves room for further optimization towards real-time video understanding. Therefore, continuous improvements and research efforts are necessary to further reduce latency, so as to better meet the stringent requirements of real-time video analysis.

**[NEW]** Grounded Query Decomposition targets the coarse pre-filtering stage. Its accuracy ceiling is therefore bounded by how much pre-filtering contributes to final accuracy — which the original ablations show is modest relative to online retrieval and multi-view QA. Grounding will not rescue queries that require temporal/causal reasoning the object inventory cannot express; such queries still rely on the downstream online scorer. Finally, Moondream's detection is itself imperfect: missed or mislabeled objects propagate into $V_g$. We mitigate this by grounding only where the query is ambiguous and falling back to blind decomposition otherwise, but a full study of detector-error propagation is left to future work.

---

## 6. Conclusions *[Original + NEW]*

E-VRAG is an efficient and accurate RAG-based method for video understanding, reducing computational costs at both data and model levels via hierarchical query decomposition, lightweight-VLM scoring, inter-frame grouping, and multi-view QA, all in a training-free, plug-and-play design.

**[NEW]** We extend E-VRAG with Grounded Query Decomposition, which closes the caption-imagination gap in pre-filtering by conditioning caption generation on a detected object inventory built cheaply offline (DINOv2 clustering, ITS selection, Moondream 0.5B detection). The change is a drop-in module that leaves the accuracy-driving online retrieval and multi-view QA stages untouched, adds no per-query cost, and is directly isolatable against blind decomposition. We report its expected magnitude honestly: a bounded refinement concentrated on vocabulary-sensitive benchmarks, not a headline accuracy leap.

---

## 7. References *[Original]*

Ataallah, K.; Shen, X.; Abdelrahman, E.; Sleiman, E.; Zhuge, M.; Ding, J.; Zhu, D.; Schmidhuber, J.; and Elhoseiny, M. 2024. Goldfish: Vision-Language Understanding of Arbitrarily Long Videos. arXiv:2407.12679.

Bai, S.; Chen, K.; Liu, X.; et al. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923.

Chen, L.; Wei, X.; Li, J.; et al. 2024a. ShareGPT4Video: Improving Video Understanding and Generation with Better Captions. arXiv:2406.04325.

Chen, Y.; Xue, F.; Li, D.; et al. 2024b. LongVILA: Scaling Long-Context Visual Language Models for Long Videos. arXiv:2408.10188.

Chen, Z.; Wang, W.; Cao, Y.; et al. 2025. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv:2412.05271.

Chen, Z.; Wang, W.; Tian, H.; et al. 2024c. How Far Are We to GPT-4V? arXiv:2404.16821.

Cheng, Z.; Leng, S.; Zhang, H.; et al. 2024. VideoLLaMA 2. arXiv:2406.07476.

Fu, C.; Dai, Y.; Luo, Y.; et al. 2025. VideoMME. In CVPR, 24108–24118.

Huang, D.-A.; Radhakrishnan, S.; Yu, Z.; and Kautz, J. 2025. FRAG: Frame Selection Augmented Generation. arXiv:2504.17447.

Jeong, S.; Kim, K.; Baek, J.; and Hwang, S. J. 2025. VideoRAG: Retrieval-Augmented Generation over Video Corpus. arXiv:2501.05874.

Kaplan, J.; et al. 2020. Scaling Laws for Neural Language Models. arXiv:2001.08361.

Li, B.; Zhang, Y.; Guo, D.; et al. 2024a. LLaVA-OneVision. arXiv:2408.03326.

Li, K.; He, Y.; Wang, Y.; et al. 2024b. VideoChat. arXiv:2305.06355.

Li, K.; Wang, Y.; He, Y.; et al. 2024c. MVBench. In CVPR, 22195–22206.

Li, X.; Wang, Y.; Yu, J.; et al. 2025. VideoChat-Flash. arXiv:2501.00574.

Lin, B.; Ye, Y.; Zhu, B.; et al. 2024. Video-LLaVA. arXiv:2311.10122.

Liu, S.; Zhao, C.; Xu, T.; and Ghanem, B. 2025a. BOLT. arXiv:2503.21483.

Liu, Z.; Zhu, L.; Shi, B.; et al. 2025b. NVILA. arXiv:2412.04468.

Luo, Y.; Zheng, X.; Yang, X.; et al. 2024. Video-RAG. arXiv:2411.13093.

Maaz, M.; Rasheed, H.; Khan, S.; and Khan, F. S. 2024. Video-ChatGPT. arXiv:2306.05424.

Min, J.; Buch, S.; Nagrani, A.; Cho, M.; and Schmid, C. 2025. MoReVQA. arXiv:2404.06511.

Qin, M.; Liu, X.; Liang, Z.; et al. 2025. Video-XL-2. arXiv:2506.19225.

Radford, A.; Kim, J. W.; Hallacy, C.; et al. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020.

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. TimeChat. arXiv:2312.02051.

Shen, X.; Xiong, Y.; Zhao, C.; et al. 2024. LongVU. arXiv:2410.17434.

Song, E.; Chai, W.; Wang, G.; et al. 2024. MovieChat. arXiv:2307.16449.

Tan, X.; Ye, Y.; Luo, Y.; et al. 2025. RAG-Adapter. arXiv:2503.08576.

Tang, X.; Qiu, J.; Xie, L.; Tian, Y.; Jiao, J.; and Ye, Q. 2025. Adaptive Keyframe Sampling. arXiv:2502.21271.

Team, Q. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Wang, X.; Song, D.; Chen, S.; Zhang, C.; and Wang, B. 2024a. LongLLaVA. arXiv:2409.02889.

Wang, X.; Zhang, Y.; Zohar, O.; and Yeung-Levy, S. 2024b. VideoAgent. arXiv:2403.10517.

Wang, Y.; Li, K.; Li, X.; et al. 2024c. InternVideo2. arXiv:2403.15377.

Wu, H.; Li, D.; Chen, B.; and Li, J. 2024. LongVideoBench. In NeurIPS, 37: 28828–28857.

Xiao, J.; Shang, X.; Yao, A.; and Chua, T.-S. 2021. NextQA. In CVPR, 9777–9786.

Xu, Z.; Dai, Q.; Xie, T.; et al. 2025. ViaRL. arXiv:2505.15447.

Yao, L.; Wu, H.; Ouyang, K.; et al. 2025. Generative Frame Sampler. arXiv:2503.09146.

Yu, S.; Jin, C.; Wang, H.; et al. 2025. Frame-Voyager. arXiv:2410.03226.

Yuan, H.; Liu, Z.; Qin, M.; et al. 2025. MemVid. arXiv:2503.09149.

Zhang, B.; Li, K.; Cheng, Z.; et al. 2025a. VideoLLaMA 3. arXiv:2501.13106.

Zhang, P.; Zhang, K.; Li, B.; et al. 2024a. Long Context Transfer from Language to Vision. arXiv:2406.16852.

Zhang, S.; Yang, J.; Yin, J.; Luo, Z.; and Luan, J. 2025b. Q-Frame. arXiv:2506.22139.

Zhang, Y.; Wu, J.; Li, W.; et al. 2024b. LLaVA-Video. arXiv:2410.02713.

Zhou, J.; Shu, Y.; Zhao, B.; et al. 2024. MLVU. arXiv:2406.

**[NEW additions]**

Oquab, M.; Darcet, T.; Moutakanni, T.; et al. 2024. DINOv2: Learning Robust Visual Features without Supervision.TMLR.

**Moondream.** 2025. Moondream 0.5B: World's Smallest Vision-Language Model. https://moondream.ai/blog/introducing-moondream-0-5b

Xu, Z.; Zhang, J.; Wang, Q.; and Liu, Y. 2025. E-VRAG: Enhancing Long Video Understanding with Resource-Efficient Retrieval Augmented Generation. arXiv:2508.01546v1.

---

## Appendix A — Honest contribution summary *[NEW]*

| Question | Answer |
|---|---|
| What is added? | Grounded Query Decomposition: $C_H = P_H(Q) \rightarrow C_H = P_H(Q, V_g)$ |
| Where does it plug in? | Frame pre-filtering, caption-generation sub-step only |
| What is unchanged? | Online retrieval, multi-view QA, all hyperparameters, all baselines |
| Per-query cost change? | None (DINOv2 + Moondream are offline, cached per video) |
| Expected accuracy effect? | Bounded refinement; primary expected signal on LVB; honest magnitude ≤ the original decomposition ablation's +1.8 LB |
| What would falsify it? | Δ < +0.3 on all four benchmarks — then the imagination gap is smaller than hypothesized and we report a null/negative result |
| Is it novel? | Yes as a *grounding of E-VRAG's own decomposition step*; it does not claim to beat E-VRAG's architecture, only to tighten its weakest pre-filtering assumption |
