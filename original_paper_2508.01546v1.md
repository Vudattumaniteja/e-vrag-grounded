## **E-VRAG: Enhancing Long Video Understanding with Resource-Efficient Retrieval Augmented Generation** 

## **Zeyu Xu**[*] **, Junkang Zhang**[*] **, Qiang Wang**[*] **, Yi Liu**[†] 

## **Abstract** 

Vision-Language Models (VLMs) have enabled substantial progress in video understanding by leveraging cross-modal reasoning capabilities. However, their effectiveness is limited by the restricted context window and the high computational cost required to process long videos with thousands of frames. Retrieval-augmented generation (RAG) addresses this challenge by selecting only the most relevant frames as input, thereby reducing the computational burden. Nevertheless, existing video RAG methods struggle to balance retrieval efficiency and accuracy, particularly when handling diverse and complex video content. Offline paradigms enable rapid retrieval by reusing pre-extracted static frame features, but they fail to capture fine-grained query-frame relationships, leading to suboptimal performance for tasks demanding diverse and detailed semantics. In contrast, online paradigms achieve higher retrieval accuracy via joint query-frame representations, but incur significant computational overhead as model size and video length increase. To address these limitations, we propose E-VRAG, a novel and efficient video RAG framework for video understanding. We first apply a frame pre-filtering method based on hierarchical query decomposition to eliminate irrelevant frames, reducing computational costs at the data level. We then employ a lightweight VLM for frame scoring, further reducing computational costs at the model level. Additionally, we propose a frame retrieval strategy that leverages the global statistical distribution of inter-frame scores to mitigate the potential performance degradation from using a lightweight VLM. Finally, we introduce a multi-view question answering scheme for the retrieved frames, enhancing the VLM’s capability to extract and comprehend information from long video contexts. Experiments on four public benchmarks show that E- VRAG achieves about 70% reduction in computational cost and higher accuracy compared to baseline methods, all without additional training. These results demonstrate the effectiveness of E-VRAG in improving both efficiency and accuracy for video RAG tasks. 

## **Introduction** 

With the explosive growth of multimedia content, video understanding has become a key area in artificial intelligence. Vision-Language Models (VLMs) excel at cross-modal fusion, jointly modeling videos and semantics to enable richer 

*These authors contributed equally. †Project Leader; Corresponding Author. 

**==> picture [244 x 95] intentionally omitted <==**

**----- Start of picture text -----**<br>
65<br>Online Baseline (BOLT) E-VRAG (ours)<br>800 E-VRAG w/o Multi-View QAE-VRAG Save 70%<br>600 About 80% 60 Online<br>RAG<br>400 Save 85% OfflineRAG<br>55<br>Uniform LLaVA-OV BOLT<br>200 Sampling InternVL2AKS FRAG<br>Video-RAG Margin<br>0 50<br>Total Retrieval Answer 0 200 400 600 800 1000<br>Stages of video RAG TFLOPs<br>Improve 17%<br>TFLOPs<br>LVB Accuracy (%)<br>**----- End of picture text -----**<br>


Figure 1: The computational analysis for stages of video RAG (left). The comparison of FLOPs between baselines and our E-VRAG (right). 

context awareness and more complex reasoning. They have demonstrated superior performance in various video tasks, such as video retrieval, question answering (QA), and event detection (Lin et al. 2024; Wang et al. 2024c; Li et al. 2024b). However, VLMs struggle with long videos containing thousands of frames, mainly due to limited context windows and high computational demands. To address these challenges, some works focus on training VLMs with longer context windows (Zhang et al. 2024a; Chen et al. 2024b; Wang et al. 2024a), which require large-scale videotext paired datasets and substantial computational resources. Other works aim to reduce visual tokens through token compression (Song et al. 2024; Ren et al. 2024), which inevitably leads to the loss of fine-grained information and a decrease in performance on complex and detailed understanding. 

Recently, Retrieval-Augmented Generation (RAG) has attracted significant attention in video understanding (Jeong et al. 2025). The typical RAG framework retrievals the most relevant frames by computing relevance scores between numerous frames and the given query, and inputs these distilled key cues for deeper reasoning. By filtering data before inputting into VLMs, RAG significantly reduces computational overhead and context confusion, improving the adaptability of VLMs in video processing and understanding. 

However, video RAG faces an efficiency-accuracy tradeoff. Offline video RAG methods prioritize efficiency by preextracting generic frame features, enabling reuse for each query (Luo et al. 2024). However, these static features often miss nuanced query-frame relationships, reducing retrieval quality for tasks with diverse semantics or granularity. In 

contrast, online video RAG methods prioritize accuracy by jointly modeling the relationship between each frame and the query, typically with a VLM (Huang et al. 2025; Liu et al. 2025a). However, their computational cost increases with both the model size and the number of frames, and to achieve comprehensive coverage, large models and a large number of frames are often employed, resulting in significant speed bottlenecks. As illustrated in the left of Figure 1, when using one 7B model for video RAG, retrieval from 256 frames accounts for about 80% of the total computation cost, greatly reducing efficiency and constraining practical applicability. 

In this paper, we propose E-VRAG, which reduces retrieval computation at both data and model levels, and enhances accuracy through novel retrieval and QA methods, achieving significant margin improvements in both accuracy and efficiency, as shown in the right of Figure 1. First, since only a small subset of frames are usually relevant, we introduce a pre-filtering method to coarsely filter frames, reducing unnecessary subsequent computation. Next, we utilize a lightweight VLM to efficiently compute the relationships between frames and queries, thereby further reducing computational overhead. To address potential performance drops from the lightweight VLM, we propose a retrieval method based on the global distribution of inter-frame scores, ensuring comprehensive retrieval. Finally, considering it is hard to extract all valuable cues from numerous retrieved frames by VLM at once, we design a multi-view QA scheme that iteratively performs QA and integrates feedback from different views, enhancing video understanding with the given query. Experiments show that E-VRAG achieves lower computational cost and higher accuracy in a fully training-free setting. Our main contributions can be summarized as: 

- We propose a frame pre-filtering method based on hierarchical query decomposition and score frames with a lightweight VLM to reduce the computational overhead of video RAG at both data and model levels. 

- We proposed a frame retrieval method based on the global distribution of inter-frame scores and a multi-view QA scheme to enhance the accuracy of retrieval and response. 

- Our proposed E-VRAG reduces computation by up to 70% while maintaining or surpassing the accuracy of baseline models on four public benchmarks, under the training-free setting. 

## **Related Works** 

## **Large Video-Language Models** 

Open-source VLMs, such as LLaVA-OneVision (Li et al. 2024a), NVILA (Liu et al. 2025b), InternVL2 (Chen et al. 2025), and Qwen2.5VL (Bai et al. 2025), have greatly advanced visual understanding and show powerful capabilities across diverse general visual tasks. According to Scaling Laws (Kaplan et al. 2020), to improve the visual understanding capabilities, the parameters of VLMs are gradually expanded to 72B or even larger, resulting in a higher cost for application especially for videos. Thus, more efficient VLMs specifically designed for videos have emerged. 

Some works focus on the alignment of video representations with other modalities, such as VideoLLaVA (Lin et al. 2024), InternVideo2 (Wang et al. 2024c), and VideoChat (Li et al. 2024b). Some works concentrate on the extraction and compression of spatiotemporal information from videos, such as Video-ChatGPT (Maaz et al. 2024) and VideoLLaMA2 (Cheng et al. 2024). There are also some works focus on data and training, enhancing video understanding capabilities through synthetic data and better training schemes, such as ShareGPT4Video (Chen et al. 2024a), LLaVAVideo (Zhang et al. 2024b), and VideoLLaMA3 (Zhang et al. 2025a). Although the video understanding capabilities of existing models have significantly improved, there remain substantial challenges in understanding long videos. 

## **Long Video Understanding** 

Current VLMs often struggle with long video understanding due to limited context length and low efficiency. Recent works have provided various solutions to address these challenges. Some works try to extend the context length of VLMs through training, enabling the processing of more frames, such as LongVA (Zhang et al. 2024a), LongVILA (Chen et al. 2024b), and LongLLaVA (Wang et al. 2024a). However, training a long-context VLM is costly in terms of data and resources. Some works focus on compressing and refining visual information to adapt to VLMs, such as MovieChat (Song et al. 2024), TimeChat (Ren et al. 2024), VideoChat-Flash (Li et al. 2025), LongVU (Shen et al. 2024), Video-XL2 (Qin et al. 2025). While compression improves efficiency, it can also lead to information loss, potentially limiting the performance of VLM on fine-grained or complex tasks. 

## **Video Understanding with RAG** 

Combining RAG with VLMs for long video understanding has recently gained significant attention. Video RAG can be typically categorized as offline or online methods. Offline methods extract frame features independently of queries, enabling the reuse of frame features for efficient scoring and retrieval across different queries, such as VideoRAG (Jeong et al. 2025), Video-RAG (Luo et al. 2024), Q-Frame (Zhang et al. 2025b), and MemVid (Yuan et al. 2025). However, the pre-extracted static frame features are hard to handle diverse and fine-grained queries, resulting in decreased retrieval accuracy. Online methods use models to jointly extract relationship between each frame and query, enabling more detailed and accurate retrieval, such as FRAG (Huang et al. 2025), BOLT (Liu et al. 2025a), RAG-Adapter (Tan et al. 2025), GenS (Yao et al. 2025), ViaRL (Xu et al. 2025), and Frame-voyager (Yu et al. 2025). However, online methods require recomputing the matching database for frames with each input query, which significantly reduce efficiency. Additionally, some methods achieve enhancement of video RAG through iterative querying, such as VideoAgent (Wang et al. 2024b) and MoReVQA (Min et al. 2025). Overall, the existing online and offline video RAG methods face the challenge of balancing efficiency and accuracy. To reach the Pareto frontier, we focus on reducing the computation costs 

**==> picture [496 x 273] intentionally omitted <==**

**----- Start of picture text -----**<br>
 (1) Frame Pre-filtering sim<br>Inter-Frame Similarity<br> Frame<br>Ss a Sampling La a Re Group<br>Avg<br>frame<br>Query:  How long does  C1:  A picture of a girl holding  CLIP oooeaG ITS<br>it take for the girl in the<br>a backpack. A picture of  ...<br>video to get from home  Light C2:  A picture of a city skyline<br>to work?<br>LLM with tall buildings. A picture of  ...  BOOee| a tps Se<br>Hierarchical C3:  A picture of the girl walking<br>Decomposition Prompt towards a city. A picture of  ... coome) Gel<br>Lo Hierarchical Captions BEGG CLIP Similarity 1 Pre-filtering Frames a ae<br> (2) Frame Retrieval score  (3) Multi-view QA<br>Inter-Frame Score R1:  It takes a ...<br>AN tes oe A or<br>R2:  The girl ...<br>Group<br>| R3:  It takes less<br>il Pre-filtering Frames as aaalJa) “@ Cee Retrieved Frames than 30 mins ...<br>frame<br>Light LightVideo Reasons<br>Query: Query:  How long does How long does  VLLM ITS Query:  How long does  LLMVLM<br>it take for the girl in the it take for the girl in the  it take for the girl in the  Answer 1:  B<br>video to get from home video to get from home  video to get from home  Answer 2:  A<br>to work?to work? to work? Answer 3:  A<br>Scoring Prompt Multi-view Prompt Early stop<br>VLM Vocab Score Retrieved Frames Final Answer:  A<br>**----- End of picture text -----**<br>


Figure 2: The overall workflow of our E-VRAG. First, the frame pre-filtering stage quickly filters relevant frames using query decomposition and inter-frame similarity. Next, the frame retrieval stage accurately retrievals frames using a lightweight VLM and inter-frame scores. Finally, the multi-view QA stage enhances the answer from multiple views. 

while maintaining performance across the entire video RAG process, aiming for optimal efficiency and accuracy. 

## **Methods** 

In this section, we introduce E-VRAG, highlighting efficiency enhancements at both data and model levels, retrieval accuracy improvement via inter-frame grouping, and answer refinement through multi-view integration. 

## **Overview** 

The general video understanding can be formulated as: 

**==> picture [161 x 11] intentionally omitted <==**

where the input _F_ = _{_ Frame _i}[N] i_ =1[is] _[N]_[frames][in][video] and generally uniform sampled, _Q_ is the qurey, and _A_ is the answer. Unrestrictedly increasing _N_ may exceed the context length limitation of VLM and obscure key information. The RAG-based video understanding methods address this by retrieving _R_ and inputting only the most relevant frames: 

**==> picture [177 x 11] intentionally omitted <==**

Clearly, the key to video RAG is efficiently and accurately locating relevant frames and comprehensively understanding their content. Thus, we propose E-VRAG, an efficient video RAG that enhances video understanding. As shown in Figure 2, E-VRAG has three stages: frame pre-filtering hierarchically matches queries to frames and filters out irrelevant 

frames; frame retrieval uses a lightweight VLM and interframe scores for efficiently and accurate retrieval; multiview QA answers queries from multiple views to boost understanding. 

## **Frame Pre-filtering** 

**Similarity with Hierarchical Query Decomposition.** We think most frames are only weakly related to the query and can be quickly filtered by coarse multi-modal alignment, without detailed analysis. However, pre-trained CLIP-like models align images with captions, which often has a semantic gap with queries, since relevant information may be implicit in queries and not explicitly stated as in captions, leading to retrieval failures. 

Therefore, we decompose the query into three levels and transform them into captions that are more suitable for image matching: 

**==> picture [231 x 12] intentionally omitted <==**

where _PH_ is the hierarchical decomposition prompt. _Centity_ focuses on directly describable entities mentioned in the query, which are the simplest cases and can be directly translated into image captions (e.g., cats, dogs). _Cknow_ focuses on entities or abstract concepts that cannot be directly described, which need the LLM to serve as a knowledge base, leveraging external knowledge to convert these entities into visual captions (e.g., describing New York as a city with 

skyscrapers or referencing the Statue of Liberty). _Ccausal_ focuses on causal and logical relationships within the query, also requiring the LLM to serve as a knowledge base to supplement the relevant captions of events that are not explicitly mentioned. 

We utilize CLIP (Radford et al. 2021) to compute the similarity _S_ between frames and decomposed captions: 

**==> picture [202 x 11] intentionally omitted <==**

**==> picture [183 x 30] intentionally omitted <==**

where CLIPt and CLIPv are the text and vision encoders of CLIP, respectively, _C_ is the total caption number of _CH_ , and _·_ is the inner product. This stage is computationally efficient, as the query decomposition can be combined into a single LLM inference, and the image features within the same video can be reused. 

**Filtering with Inter-frame Similarity.** Typically, the Top-K frames with the highest similarity scores are selected, which works well only when features are highly distinguishable. Otherwise, the selected frames are temporally adjacent, oversampling major events and undersampling others. 

Therefore, we consider the global distribution of interframe similarity, and group frames to ensure higher similarity within groups than between groups. We group frames through clustering with a specific temporal constraint: only frames that are both temporally adjacent and similar are grouped. The algorithmic details are provided in the supplementary materials. By incorporating temporal relationships, it can be beneficial for causal understanding, as similar events at different times are analyzed separately rather than merged. 

Then, we sample the frames within each group individually to control redundancy while preserving representativeness. The sampling method is Inverse Transform Sampling (ITS) (Liu et al. 2025a). For each group _g_ , it uniformly samples frames _Fg_ based on the inverse function of the cumulative distribution of similarity _Sg[′]_[to avoid redundant dense] sampling: 

**==> picture [212 x 50] intentionally omitted <==**

where _Ng_ is the frame number of _Fg_ , _M_ is the number of frames expected to be selected, _G_ is the number of groups, and _⌊⌋_ is the floor operation. 

## **Frame Retrieval** 

**Scoring with Lightweight VLM.** For the potentially relevant frames obtained after pre-filtering, we further utilize the multi-modal understanding capabilities of the VLM to evaluate their relevance to the query. Specifically, each frame is paired with the query and input to the VLM using a binary relevance judgment ( _yes_ / _no_ ) prompt. The VLM generates a binary response based on the instruction, and we use the full 

vocabulary probability distribution _P_ all _∈_ R _[v]_ of the answer word as the relevance score for retrieval, where _v_ is the vocabulary size of the VLM. Since this stage requires multiple VLM inferences, we use only a lightweight VLM to ensure efficiency. 

**Retrieval with Inter-frame Probability.** Given the capability limitations of lightweight VLM in scoring, we continue to use the grouping and sampling strategy from the frame pre-filtering stage to maintain retrieval quality. Specifically, frames are grouped based on their vocabulary probability _P_ all, under the assumption that frames with more similar _P_ all distributions are likely to be more correlated and redundant. For each group, we continue to employ ITS for frame retrieval, modifying the similarity to the scores generated by _P_ all while keeping others unchanged. We consider two scoring strategies: the first uses only the probability of word _yes_ , i.e., _P_ all ( _yes_ ), while the second incorporates _P_ all( _yes_ ) both words _yes_ and _no_ , i.e., _P_ all( _yes_ )+ _P_ all( _no_ )[. The retrieved] frames _Fr_ are subsequently utilized to answer the query. 

## **Multi-view QA** 

A single VLM inference may struggle to fully extract all information from multiple frames in _Fr_ , especially finegrained details. Enhancing by supplementing information through multi-round reasoning has demonstrated effectiveness. However, sequential aggregation reasoning within a single chain, where answer is generated solely based on the final inference, is prone to error accumulation in intermediate steps and may lack robustness. 

Thus, we propose multi-view QA, in which each round attempts to reason and answer query from a distinct view, and answers are aggregated and complemented in parallel to produce a more comprehensive response. Specifically, in the _t_ -th round of QA, the VLM generates a reason _Rt_ and an answer _At_ based on the input frame _Fr_ , the query _Q_ , as well as the reasons _R<t_ and answers _A<t_ from previous rounds, which serve as additional supplementary contexts: 

**==> picture [232 x 11] intentionally omitted <==**

where _T_ is the total number of views. Notably, in the first view, only query and frames are input. In this process, the VLM is required to explicitly state the view used to analyze the frames and answer the query in its generated reasons. Additionally, it is prompted to respond from a view that differs from those employed in previous rounds. By incorporating feedback from multiple views, the VLM progressively refines its understanding of the query and frames. 

To enhance efficiency, we introduce an early stopping strategy, whereby the process is terminated if answers generated in two consecutive rounds are identical. A voting mechanism is utilized to determine the final answer from all answers generated during the multi-view QA process. In cases where multiple answers have the same count, the answer generated latest among them is chosen as the final answer: 

**==> picture [209 x 25] intentionally omitted <==**

|**Method**|**Retrieval**|**Answer**|**TFLOPs**|**#Frames**|**Video-MME**|**MLVU**|**LVB**|**NextQA**|
|---|---|---|---|---|---|---|---|---|
|**Fundamental Methods**|||||||||
|VideoChat2 (Li et al. 2024c)|–|7B|–|–|39.5|44.5|39.3|–|
|VideoLLaMA2 (Cheng et al. 2024)|–|7B|–|–|47.9|–|–|–|
|InternVL2_∗_(Chen et al. 2024c)|–|8B|243|64|56.6|60.7|52.2|80.6|
|LLaVA-OV_∗_(Li et al. 2024a)|–|7B|89|32|57.4|61.8|54.0|78.8|
|Qwen2.5VL_∗_(Bai et al. 2025)|–|7B|130|32|62.1|59.6|58.1|81.6|
|LLaVA-Video_∗_(Zhang et al. 2024b)|–|7B|177|64|64.3|69.5|61.2|83.8|
|**Offine Video RAG Methods**|||||||||
|Video-RAG (Luo et al. 2024)|0.3B|7B|–|64|–|**72.4**|58.7|–|
|VideoAgent (Wang et al. 2024b)|8B|GPT-4|–|–|–|–|–|71.3|
|Goldfsh (Ataallah et al. 2024)|7B|7B|–|–|28.9|37.3|–|–|
|MemVid (Yuan et al. 2025)|7B|7B|–|128|63.7|58.1|–|–|
|AKS_∗†_ (Tang et al. 2025)|0.3B|7B|50+177|64|64.3|69.3|60.7|83.3|
|**Online Video RAG Methods**|||||||||
|Frame-voyager (Yu et al. 2025)|7B|7B|–|8|57.5|65.6|–|73.9|
|FRAG_∗†_ (Huang et al. 2025)|7B|7B|708+177|64|63.7|69.2|60.6|82.5|
|BOLT_∗†_ (Liu et al. 2025a)|7B|7B|708+177|64|64.6|70.3|62.2|83.2|
|**E-VRAG (ours)**|2B|7B|103+372|64|**65.4**|70.2|**63.1**|**84.0**|



Table 1: Comparison of E-VRAG with SOTA video understanding methods. *: local implementation using official code/weights for fair comparison, _†_ : 256 candidate frames uniformly sampled per video for fair comparison, values before/after “+”: TFLOPs for retrieval/answer, bold: highest, underline: second highest. For more details, see the supplementary materials. 

## **Experiments** 

In this section, we present key experiments to validate the effectiveness of E-VRAG against various SOTA video understanding methods, particularly video RAG methods, on four benchmarks. Additional experiments are provided in the supplementary materials. 

## **Experimental Setup** 

**Benchmarks** We evaluated our method on four widelyused video benchmarks: Video-MME (Fu et al. 2025), LongVideoBench (Wu et al. 2024), MLVU (Zhou et al. 2024), and NextQA (Xiao et al. 2021). Video-MME comprises a diverse collection of short, medium, and long videos, comprising a total of 900 videos and 2,700 questions, with an average video duration exceeding 8 minutes. Both LongVideoBench (LVB) and MLVU are specifically designed to assess models’ capabilities in understanding long videos, with an average video duration greater than 8 minutes. NextQA consists of shorter video clips, with an average duration of only 0.8 minutes. 

**Baseline Models** We compare our E-VRAG with three types of baselines for evaluation. The first type includes fundamental methods that are general for visual understanding or video understanding without RAG. The second type includes offline video RAG methods that extract information of frames and queries separately by retrieval models for RAG. The third type includes online video RAG methods that extract information jointly by retrieval models for RAG. Through the above comparisons, the efficiency and accuracy 

of our EVIRAG can be comprehensively demonstrated. 

**Implementation Details** We conducted all experiments on 8 NVIDIA A800 80G GPUs without training. Following the experimental protocol in (Liu et al. 2025a; Huang et al. 2025), we uniformly sample 256 candidate frames and retrieve 64 frames with dynamic resolution of 1 to generate answers. For accuracy reporting, we use consistent settings across four datasets: a group number of 26, a scoring type of two words, a multi-view number of 2. The impact and select reason of these main hyperparameters are discussed in the ablation study and supplementary materials. The prefiltering stage acts as a middle step between uniform sampling and retrieval. Thus, the number of filtered frames is set to 128, accounting for 50% of candidate frames and twice the retrieval stage frames. Its group number is also set to twice that of the retrieval stage to maintain proportionality. The lightweight LLM used for query decomposition is Qwen3-1.7B (Team 2025). 

## **Main Results** 

**Accuracy Comparison with SOTA Methods** Table 1 presents a comparison between our method and three types of baseline methods. It can be find that the performances of most general fundamental methods are not as good as video RAG methods, especially in long video benchmarks like LVB or MLVU. Among these fundamental methods, the LLaVA-Video has the highest accuracy across four benchmarks, thus we use it as the answer model and perform comparisons with other SOTA video RAG methods equally. The 

|FP.<br>FR.<br>QA.<br>TFs<br>**VE**<br>**MU**<br>**LB**<br>**NA**<br>–<br>–<br>–<br>177<br>64.3<br>69.5<br>61.2<br>83.8<br>✓<br>–<br>–<br>178<br>64.5<br>69.2<br>60.4<br>83.5<br>–<br>✓<br>–<br>381<br>65.0<br>**70.4**<br>63.0<br>83.7<br>✓<br>✓<br>–<br>280<br>65.0<br>70.0<br>**63.4**<br>83.6<br>✓<br>✓<br>✓<br>475<br>**65.4**<br>70.2<br>63.1<br>**84.0**<br>Tabl|FPGR.<br>FRGR.<br>**VE**<br>**MU**<br>**LB**<br>**NA**|
|---|---|
||–<br>–<br>62.5<br>69.4<br>60.1<br>82.6<br>✓<br>–<br>64.3<br>**70.3**<br>60.5<br>83.0<br>✓<br>✓<br>**65.0**<br>70.0<br>**63.4**<br>**83.6**|
||e4:Theablationresultsofretrievalwithinter-f|



Table 4: The ablation results of retrieval with inter-frame grouping. FPGR.: group retrieval in frame pre-filtering, FRGR.: group retrieval in frame retrieval. 

Table 2: The ablation results of three stages in EfficientVRAG. FP.: frame pre-filting, FR.: frame retrieval, QA.: multi-view QA, TFs: TFLOPs, VE: Video-MME, MU: MLVU, LB: LVB, NA: NextQA. The same abbreviation is used in the following tables to represent the same meaning. 

|Views|TFLOPs|**VE**|**MU**|**LB**|**NA**|
|---|---|---|---|---|---|
|1|280|65.0|70.0|**63.4**|83.6|
|2|475|65.4|70.2|63.1|84.0|
|3|688|65.4|**70.3**|63.0|**84.1**|
|4|919|**65.5**|**70.3**|63.0|**84.1**|
|5|1168|**65.5**|**70.3**|63.2|**84.1**|



|Query Decom.|**VE**|**MU**|**LB**|**NA**|
|---|---|---|---|---|
|–|64.9|**70.3**|61.6|83.5|
|✓|**65.0**|70.0|**63.4**|**83.6**|



Table 5: The accuracy of different view number in multiview QA. 

Table 3: The ablation results of query decomposition. Query Decom.: query decomposition. 

accuracy of the offline methods is rarely higher than that of the online methods across the four benchmarks, which highlights the advantage of the online methods in accuracy. Our E-VRAG achieves the highest accuracy on three benchmarks comparing with other baselines, with an accuracy improvement of 0.8% in Video-MME, 0.9% in LVB, and 0.2% in NextQA, comparing with the highest accuracy of all baselines. 

To comprehensively validate the generalization of E- VRAG, we further use LLaVA-OV and Qwen2.5VL as answer models and apply E-VRAG to them. The results are shown in the supplementary materials, where accuracy increases after apply E-VRAG. These results demonstrate that our E-VRAG is applicable to multiple fundamental models and provides consistent improvements. 

**Efficiency Comparison with SOTA Methods** Table 1 shows the computation costs each method used. All answer models have a similar parameter size of 7B or 8B, so efficiency differences mainly due to the retrieval models. Fundamental methods uniformly sample frames without retrieval, resulting in minimal computational cost but the lowest accuracy. Offline and online RAG methods uses model to compute retrieval clues, which improves accuracy but increases computational load. Moreover, although the accuracy of online methods is higher than offline methods, the cost of increased computation outweighs its benefits. Our E-VRAG method incorporates online retrieval similar to FRAG and BOLT, but pre-filters irrelevant frames and uses a lightweight VLM, enhancing efficiency at both the data and model levels. Thus, it achieving about 50% reduction in computational cost with even higher accuracy compared with baseline online methods. Moreover, according to the results in Table 5, our E-VRAG can be more efficiency by replace the multi-view QA with a single-view QA, achieving about 70% reduction in computational cost and still maintains good performances. These results demonstrate that 

combining a compact retrieval model with an appropriate frame retrieval method is an effective way to achieve a balance between efficiency and accuracy. 

## **Quantitative Comparison** 

**Global Ablation** We conducted ablation studies on three stages of E-VRAG, as shown in Table 2. Omitting all stages equals uniform sampling, which is computationally efficient but has low accuracy. Using only frame pre-filtering corresponding to the offline RAG, where decomposed queries are matched with static frame features and frames are retrieved based on coarse-grained understanding. It offers limited accuracy gains, likely due to its tendency to retrieve incorrect or redundant frames. Thus, it is more suitable for coarse filtering rather than direct retrieval. Using only frame retrieval corresponding to the online RAG, which models the relationship between queries and frames in detail and achieves better accuracy than previous methods. However, despite employing a lightweight VLM, the computational cost remains high, as inference is required for each frame-query pair. Combining above two stages leverages the strengths of each, resulting in an approximately 30% reduction in computational cost while maintaining comparable performance, thus achieving a balance between accuracy and efficiency. Incorporating multi-view QA typically improves accuracy, but at the cost of higher computation. A more detailed analysis of three stages is presented in the following section. 

**Analysis of Query Decomposition** We evaluated the effectiveness of query decomposition using the ablation study shown in Table 3, where only the query decomposition in frame pre-filtering is replaced by directly inputting the original query into CLIP to compute text features for retrieval. After removing it, the accuracy on Video-MME, LVB and NextQA decreased, while that on MLVU increased. Among these benchmarks, the improvement brought by query decomposition is most significant on LVB, which may because 

**==> picture [483 x 136] intentionally omitted <==**

**----- Start of picture text -----**<br>
 Q: Which vegetable is not visible in this<br> video when introducing the first tip?<br>Pepper. Potato. Carrot. Beans.<br>BOLT: Beans  ( ✘ )<br>E-VRAG: Potato  ( ✔ )<br>a5<br>ae ee eee<br>SUEUR HLI LIT eT<br>0 Frame Index 256<br>BOLT<br>E-VRAG<br>**----- End of picture text -----**<br>


Figure 3: The visualization comparison with E-VRAG and online video RAG baseline. 

queries in LVB are relatively long and contain more information, making them less suitable for retrieval using a single short caption as recommended by CLIP. 

**Analysis of Inter-frame Group Retrieval** We evaluate the effectiveness of inter-frame group retrieval through the ablation study shown in Table 4, where the inter-frame group retrieval in frame pre-filtering (FPGR.) and frame retrieval (FRGR.) stages is progressively removed, and replaced with Top-K retrieval. Removing it in both stages has the lowest accuracy, even lower than the basic LLaVA-Video with uniform sampling shown in Table 1, which indicates that TopK selection is likely to be redundant and incomplete. After incorporating inter-frame group retrieval into both the frame pre-filtering and retrieval stages, the accuracy on almost all benchmarks increased compared to the previous results, demonstrating its effectiveness at each stage. The decrease on MLVU at frame retrieval stage may partially attributed to the use of a unified group number hyperparameter. However, the overall performance still surpasses that of the methods without group retrieval. We also comprehensively analyze how group number and score type affect interframe group retrieval in the supplementary materials. 

**Analysis of Multi-view QA** The accuracy results for different numbers of views are shown in Table 5. For VideoMME, MLVU, and NextQA, accuracy consistently improves as the number of views increases, demonstrating the effectiveness of the multi-view QA. However, as the number of views continues to grow, accuracy improvements gradually plateau, while the computational cost keeps increasing, resulting in diminishing returns. We report the accuracy when the number of views is set to 2, as it achieves the most improvement in accuracy with relatively low computational overhead, striking a balance between efficiency and accuracy. For LVB, accuracy initially decreases and then improves. This may be attributed to each view generating divergent answers at first, which causes confusion and consequently requires more views to refine the final answer. 

## **Visualization Comparison** 

Figure 3 visualizes the distribution of retrieved frames for the online baseline BOLT and our E-VRAG, based on scores generated by the lightweight VLM. The baseline mainly retrieves frames with scores near the maximum (inside the yellow translucent boxes), while ignoring those with low scores (inside the blue translucent boxes). This results in incomplete frame sampling, with redundancy in high-score regions and missing information in low-score regions. Although higher scores ideally indicate greater relevance, this is not always achieved due to limitations of the scoring model. This highlights the need for a dedicated frame retrieval method rather than simply relying on Top-K selection. In contrast, E- VRAG ensures high-scoring frames are included, while redistributing redundant high-score selections to some lowerscore frames. This ensures key information is preserved and provides a more comprehensive video representation. In the case illustrated in Figure 3, E-VRAG compensates for the scoring limitations of the lightweight VLM, accurately retrieving frames with low scores and thereby enhancing robustness without sacrificing efficiency. 

**Limitation** Our E-VRAG significantly improves the efficiency of video RAG. However, it still leaves room for further optimization towards real-time video understanding. Therefore, continuous improvements and research efforts are necessary to further reduce latency, so as to better meet the stringent requirements of real-time video analysis. 

## **Conclusions** 

In this paper, we propose E-VRAG, an efficient and accurate RAG-based method for video understanding. The E-VRAG hierarchically decomposes queries to quickly identify potentially relevant frames, and utilizes a lightweight VLM to efficiently score the relationships between queries and frames. As a result, E-VRAG reduces computational costs throughout the entire video RAG process at both data and model levels. To mitigate the potential impact on performance resulting from reduced computational workload, E-VRAG retrieves frames by grouping them according to inter-frame 

score distribution to emphasize global statistics, rather than relying solely on individual frames. Moreover, E-VRAG further enhances the long video understanding by answering query from multiple complementary views. Extensive experiments across four benchmarks demonstrate that our E- VRAG outperforms baseline methods in both efficiency and accuracy. Furthermore, E-VRAG adopts a plug-and-play design that requires no additional training, and can be seamlessly integrated with future advancements in video VLMs to provide highly efficient solutions. 

## **References** 

Ataallah, K.; Shen, X.; Abdelrahman, E.; Sleiman, E.; Zhuge, M.; Ding, J.; Zhu, D.; Schmidhuber, J.; and Elhoseiny, M. 2024. Goldfish: Vision-Language Understanding of Arbitrarily Long Videos. arXiv:2407.12679. 

Bai, S.; Chen, K.; Liu, X.; Wang, J.; Ge, W.; Song, S.; Dang, K.; Wang, P.; Wang, S.; Tang, J.; Zhong, H.; Zhu, Y.; Yang, M.; Li, Z.; Wan, J.; Wang, P.; Ding, W.; Fu, Z.; Xu, Y.; Ye, J.; Zhang, X.; Xie, T.; Cheng, Z.; Zhang, H.; Yang, Z.; Xu, H.; and Lin, J. 2025. Qwen2.5-VL Technical Report. arXiv:2502.13923. 

Chen, L.; Wei, X.; Li, J.; Dong, X.; Zhang, P.; Zang, Y.; Chen, Z.; Duan, H.; Lin, B.; Tang, Z.; Yuan, L.; Qiao, Y.; Lin, D.; Zhao, F.; and Wang, J. 2024a. ShareGPT4Video: Improving Video Understanding and Generation with Better Captions. arXiv:2406.04325. 

Chen, Y.; Xue, F.; Li, D.; Hu, Q.; Zhu, L.; Li, X.; Fang, Y.; Tang, H.; Yang, S.; Liu, Z.; He, E.; Yin, H.; Molchanov, P.; Kautz, J.; Fan, L.; Zhu, Y.; Lu, Y.; and Han, S. 2024b. LongVILA: Scaling Long-Context Visual Language Models for Long Videos. arXiv:2408.10188. 

Chen, Z.; Wang, W.; Cao, Y.; Liu, Y.; Gao, Z.; Cui, E.; Zhu, J.; Ye, S.; Tian, H.; Liu, Z.; Gu, L.; Wang, X.; Li, Q.; Ren, Y.; Chen, Z.; Luo, J.; Wang, J.; Jiang, T.; Wang, B.; He, C.; Shi, B.; Zhang, X.; Lv, H.; Wang, Y.; Shao, W.; Chu, P.; Tu, Z.; He, T.; Wu, Z.; Deng, H.; Ge, J.; Chen, K.; Zhang, K.; Wang, L.; Dou, M.; Lu, L.; Zhu, X.; Lu, T.; Lin, D.; Qiao, Y.; Dai, J.; and Wang, W. 2025. Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling. arXiv:2412.05271. 

Chen, Z.; Wang, W.; Tian, H.; Ye, S.; Gao, Z.; Cui, E.; Tong, W.; Hu, K.; Luo, J.; Ma, Z.; et al. 2024c. How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites. _arXiv preprint arXiv:2404.16821_ . 

Cheng, Z.; Leng, S.; Zhang, H.; Xin, Y.; Li, X.; Chen, G.; Zhu, Y.; Zhang, W.; Luo, Z.; Zhao, D.; and Bing, L. 2024. VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs. arXiv:2406.07476. 

Fu, C.; Dai, Y.; Luo, Y.; Li, L.; Ren, S.; Zhang, R.; Wang, Z.; Zhou, C.; Shen, Y.; Zhang, M.; et al. 2025. Videomme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , 24108–24118. 

Huang, D.-A.; Radhakrishnan, S.; Yu, Z.; and Kautz, J. 2025. FRAG: Frame Selection Augmented Generation for Long Video and Long Document Understanding. arXiv:2504.17447. 

Jeong, S.; Kim, K.; Baek, J.; and Hwang, S. J. 2025. VideoRAG: Retrieval-Augmented Generation over Video Corpus. arXiv:2501.05874. 

Kaplan, J.; McCandlish, S.; Henighan, T.; Brown, T. B.; Chess, B.; Child, R.; Gray, S.; Radford, A.; Wu, J.; and Amodei, D. 2020. Scaling Laws for Neural Language Models. arXiv:2001.08361. 

Li, B.; Zhang, Y.; Guo, D.; Zhang, R.; Li, F.; Zhang, H.; Zhang, K.; Zhang, P.; Li, Y.; Liu, Z.; and Li, C. 2024a. LLaVA-OneVision: Easy Visual Task Transfer. arXiv:2408.03326. 

Li, K.; He, Y.; Wang, Y.; Li, Y.; Wang, W.; Luo, P.; Wang, Y.; Wang, L.; and Qiao, Y. 2024b. VideoChat: Chat-Centric Video Understanding. arXiv:2305.06355. 

Li, K.; Wang, Y.; He, Y.; Li, Y.; Wang, Y.; Liu, Y.; Wang, Z.; Xu, J.; Chen, G.; Luo, P.; et al. 2024c. Mvbench: A comprehensive multi-modal video understanding benchmark. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , 22195–22206. 

Li, X.; Wang, Y.; Yu, J.; Zeng, X.; Zhu, Y.; Huang, H.; Gao, J.; Li, K.; He, Y.; Wang, C.; Qiao, Y.; Wang, Y.; and Wang, L. 2025. VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling. arXiv:2501.00574. 

Lin, B.; Ye, Y.; Zhu, B.; Cui, J.; Ning, M.; Jin, P.; and Yuan, L. 2024. Video-LLaVA: Learning United Visual Representation by Alignment Before Projection. arXiv:2311.10122. 

Liu, S.; Zhao, C.; Xu, T.; and Ghanem, B. 2025a. BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding. arXiv:2503.21483. 

Liu, Z.; Zhu, L.; Shi, B.; Zhang, Z.; Lou, Y.; Yang, S.; Xi, H.; Cao, S.; Gu, Y.; Li, D.; Li, X.; Fang, Y.; Chen, Y.; Hsieh, C.-Y.; Huang, D.-A.; Cheng, A.-C.; Nath, V.; Hu, J.; Liu, S.; Krishna, R.; Xu, D.; Wang, X.; Molchanov, P.; Kautz, J.; Yin, H.; Han, S.; and Lu, Y. 2025b. NVILA: Efficient Frontier Visual Language Models. arXiv:2412.04468. 

Luo, Y.; Zheng, X.; Yang, X.; Li, G.; Lin, H.; Huang, J.; Ji, J.; Chao, F.; Luo, J.; and Ji, R. 2024. Video-RAG: Visuallyaligned Retrieval-Augmented Long Video Comprehension. arXiv:2411.13093. 

Maaz, M.; Rasheed, H.; Khan, S.; and Khan, F. S. 2024. Video-ChatGPT: Towards Detailed Video Understanding via Large Vision and Language Models. arXiv:2306.05424. 

Min, J.; Buch, S.; Nagrani, A.; Cho, M.; and Schmid, C. 2025. MoReVQA: Exploring Modular Reasoning Models for Video Question Answering. arXiv:2404.06511. 

Qin, M.; Liu, X.; Liang, Z.; Shu, Y.; Yuan, H.; Zhou, J.; Xiao, S.; Zhao, B.; and Liu, Z. 2025. Video-XL-2: Towards Very Long-Video Understanding Through Task-Aware KV Sparsification. arXiv:2506.19225. 

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; 

Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models From Natural Language Supervision. arXiv:2103.00020. 

Ren, S.; Yao, L.; Li, S.; Sun, X.; and Hou, L. 2024. TimeChat: A Time-sensitive Multimodal Large Language Model for Long Video Understanding. arXiv:2312.02051. 

Shen, X.; Xiong, Y.; Zhao, C.; Wu, L.; Chen, J.; Zhu, C.; Liu, Z.; Xiao, F.; Varadarajan, B.; Bordes, F.; Liu, Z.; Xu, H.; Kim, H. J.; Soran, B.; Krishnamoorthi, R.; Elhoseiny, M.; and Chandra, V. 2024. LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding. arXiv:2410.17434. 

Song, E.; Chai, W.; Wang, G.; Zhang, Y.; Zhou, H.; Wu, F.; Chi, H.; Guo, X.; Ye, T.; Zhang, Y.; Lu, Y.; Hwang, J.-N.; and Wang, G. 2024. MovieChat: From Dense Token to Sparse Memory for Long Video Understanding. arXiv:2307.16449. 

Tan, X.; Ye, Y.; Luo, Y.; Wan, Q.; Liu, F.; and Cai, Z. 2025. RAG-Adapter: A Plug-and-Play RAG-enhanced Framework for Long Video Understanding. arXiv:2503.08576. 

Tang, X.; Qiu, J.; Xie, L.; Tian, Y.; Jiao, J.; and Ye, Q. 2025. Adaptive Keyframe Sampling for Long Video Understanding. arXiv:2502.21271. 

Yuan, H.; Liu, Z.; Qin, M.; Qian, H.; Shu, Y.; Dou, Z.; Wen, J.-R.; and Sebe, N. 2025. Memory-enhanced Retrieval Augmentation for Long Video Understanding. arXiv:2503.09149. 

Zhang, B.; Li, K.; Cheng, Z.; Hu, Z.; Yuan, Y.; Chen, G.; Leng, S.; Jiang, Y.; Zhang, H.; Li, X.; Jin, P.; Zhang, W.; Wang, F.; Bing, L.; and Zhao, D. 2025a. VideoLLaMA 3: Frontier Multimodal Foundation Models for Image and Video Understanding. arXiv:2501.13106. 

Zhang, P.; Zhang, K.; Li, B.; Zeng, G.; Yang, J.; Zhang, Y.; Wang, Z.; Tan, H.; Li, C.; and Liu, Z. 2024a. Long Context Transfer from Language to Vision. arXiv:2406.16852. 

Zhang, S.; Yang, J.; Yin, J.; Luo, Z.; and Luan, J. 2025b. Q- Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs. arXiv:2506.22139. 

Zhang, Y.; Wu, J.; Li, W.; Li, B.; Ma, Z.; Liu, Z.; and Li, C. 2024b. Video Instruction Tuning With Synthetic Data. arXiv:2410.02713. 

Zhou, J.; Shu, Y.; Zhao, B.; Wu, B.; Xiao, S.; Yang, X.; Xiong, Y.; Zhang, B.; Huang, T.; and Liu, Z. 2024. Mlvu: A comprehensive benchmark for multi-task long video understanding. _arXiv e-prints_ , arXiv–2406. 

Team, Q. 2025. Qwen3 Technical Report. arXiv:2505.09388. 

Wang, X.; Song, D.; Chen, S.; Zhang, C.; and Wang, B. 2024a. LongLLaVA: Scaling Multi-modal LLMs to 1000 Images Efficiently via a Hybrid Architecture. arXiv:2409.02889. 

Wang, X.; Zhang, Y.; Zohar, O.; and Yeung-Levy, S. 2024b. VideoAgent: Long-form Video Understanding with Large Language Model as Agent. arXiv:2403.10517. 

Wang, Y.; Li, K.; Li, X.; Yu, J.; He, Y.; Wang, C.; Chen, G.; Pei, B.; Yan, Z.; Zheng, R.; Xu, J.; Wang, Z.; Shi, Y.; Jiang, T.; Li, S.; Zhang, H.; Huang, Y.; Qiao, Y.; Wang, Y.; and Wang, L. 2024c. InternVideo2: Scaling Foundation Models for Multimodal Video Understanding. arXiv:2403.15377. 

Wu, H.; Li, D.; Chen, B.; and Li, J. 2024. Longvideobench: A benchmark for long-context interleaved video-language understanding. _Advances in Neural Information Processing Systems_ , 37: 28828–28857. 

Xiao, J.; Shang, X.; Yao, A.; and Chua, T.-S. 2021. Nextqa: Next phase of question-answering to explaining temporal actions. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , 9777–9786. 

Xu, Z.; Dai, Q.; Xie, T.; Yang, Y.; Qiu, K.; Chen, D.; Wu, Z.; and Luo, C. 2025. ViaRL: Adaptive Temporal Grounding via Visual Iterated Amplification Reinforcement Learning. arXiv:2505.15447. 

Yao, L.; Wu, H.; Ouyang, K.; Zhang, Y.; Xiong, C.; Chen, B.; Sun, X.; and Li, J. 2025. Generative Frame Sampler for Long Video Understanding. arXiv:2503.09146. 

Yu, S.; Jin, C.; Wang, H.; Chen, Z.; Jin, S.; Zuo, Z.; Xu, X.; Sun, Z.; Zhang, B.; Wu, J.; Zhang, H.; and Sun, Q. 2025. Frame-Voyager: Learning to Query Frames for Video Large Language Models. arXiv:2410.03226. 

