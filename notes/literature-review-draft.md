# Historical Approaches and Early Sequential Recommenders

Early recommendation methods use information about a user's general preferences and recent interactions. [@P001_FPMC] combines matrix factorization with personalized Markov chains, taking into account both general interests and transitions between items.

Recurrent models process interactions in order and update a hidden state [@P002_GRU4Rec]. Further work improves their ranking objectives [@P003_GRU4RecTopkGains] and introduces ways to carry information between sessions [@P004_HRNN]. Convolutional models use filters to capture patterns in recent interactions [@P006_Caser] and dilated layers to capture longer dependencies [@P007_NextItNet].

Other approaches use gating to select relevant features and interactions [@P008_HGN] or hierarchical attention to combine long-term and short-term interests [@P009_SHAN]. Graph-based models represent sessions through connections between items [@P010_SRGNN]. These approaches provide different ways to summarize a user's history, with different costs of processing and updating it.

# Self-Attention and Learning User Representations

Self-attention allows a model to capture dependencies between different positions in a sequence [@P081_AttentionIsAllYouNeed]. In sequential recommendation, causal models predict the next item using preceding interactions [@P011_SASRec], while bidirectional models learn to reconstruct masked items using the surrounding history [@P012_BERT4Rec]. This difference also affects whether previously computed states can be reused when new interactions arrive.

Further approaches add time intervals between interactions [@P013_TiSASRec], frequency patterns in user behaviour [@P017_FEARec], Gaussian item representations [@P018_STOSA], or user-specific embeddings [@P019_SSEPT]. Other work addresses recommendation with sparse interaction data [@P020_CoarsetoFineSparseSR].

The training objective also affects representation quality. Self-supervised approaches use additional signals from attributes and sequences [@P014_S3Rec]. Contrastive approaches learn from augmented histories [@P015_CL4SRec] or address representation degeneration [@P016_DuoRec]. Other work reduces the training cost of cross-entropy over large item catalogs [@P057_ScalableCrossEntropy]. When comparing refresh policies, training choices should remain fixed so that changes in quality can be attributed to representation maintenance.

# Long-History Modeling and History Selection

A long interaction history contains information about a user's general preferences, recent interests, and activity across different sessions. Models use candidate-dependent attention to select relevant interactions [@P021_DIN], model how interests evolve [@P022_DIEN], or combine interests from different sessions [@P023_DSIN].

Processing the complete history can be expensive. Some approaches first retrieve relevant events and then process them in more detail [@P025_SIM]. Others aggregate interactions with matching hash signatures [@P027_SDIM]. Further work improves consistency between the selection stage and the following attention stage [@P028_TWIN]. This reduces the amount of detailed processing, but may also remove useful information.

Further approaches compress the interaction history through offline clustering [@P029_TWINV2], use separate embeddings for history selection and representation learning [@P030_DARE], combine selected older events with recent activity [@P033_TransActV2], or summarize the history into a smaller number of tokens [@P039_VISTA]. These methods address which information to keep. Maintaining the resulting representation also requires deciding when to update it.

# Efficient Alternatives to Full Self-Attention

Efficient sequence models aim to reduce the cost of processing long histories. Low-rank attention compresses history into a smaller set of representations [@P042_LightSANs], while linear attention changes the attention computation [@P044_LinRec]. Further work separates temporal and positional components in an efficient attention model [@P051_FuXiLinear].

Other alternatives use linear recurrent states [@P045_LRURec], behaviour-dependent recurrent updates [@P046_RecBLR], or selective state-space models [@P047_Mamba4Rec]. Further approaches use structured state-space duality [@P048_SSD4Rec] and incorporate temporal information into it [@P049_TiM4Rec]. Filtering and MLP models [@P043_FMLPRec] and autoencoder models [@P052_AutoSeqRec] provide additional alternatives.

These models differ in processing time, memory use, and the way new events are incorporated. A model that processes a full sequence cheaply does not necessarily support cheap incremental updates. Whether old states can be reused also depends on the model's attention rules, normalization, and positional inputs.

# Inference and User Representation Maintenance

## Cached Representations and Incremental Updates

Computing user representations offline reduces the amount of work required at prediction time. [@P031_PinnerFormer] trains representations for longer-term engagement and computes them in batches, while [@P034_ARGUS] uses offline embeddings from a large history encoder. A cached representation can be combined with recent activity [@P032_TransAct], and a split architecture allows the effects of representation lag and model age to be studied separately [@P035_LongHistoryUserTransformers].

Other approaches update memory when events arrive [@P024_MIMNUIC], process user behaviour separately from prediction requests [@P036_TransX], or maintain changing user and item states [@P054_JODIE]. These methods show different ways to incorporate new activity without recomputing the complete history at every prediction.

## Reuse of Computation

Computation can be shared between candidates and requests through reuse of history states [@P037_HSTUMFALCON]. Cache admission and placement can reduce repeated prefix processing [@P038_RelayGR], while block-based memory management can reduce KV-cache fragmentation [@P065_PagedAttentionvLLM]. Other methods reduce memory transfers during attention [@P064_FlashAttention] or retain initial tokens and a recent window to bound the attention cache [@P066_StreamingLLM].

These methods reduce different parts of the total cost. Reusing states requires checking whether changes to the history, positions, or model weights make them invalid. Methods developed for language models also require separate evaluation before their benefits can be assumed for recommenders.

## Refresh Policies and Adaptive Computation

Refresh policies determine when user representations are updated. Previous work studies representation staleness and combines event-triggered and periodic updates [@P035_LongHistoryUserTransformers]. Related approaches adapt the input sequence length [@P053_SequenceAdaptationviaRL] or learn how many internal computation steps to use [@P068_AdaptiveComputationTime].

Policy-gradient methods provide one way to learn decisions from rewards [@P077_REINFORCE_Williams] and have also been applied to recommendation policies trained from historical interactions [@P078_TopK_OffPolicy_REINFORCE]. For a refresh controller, the decision depends on whether the expected improvement in prediction quality is worth the update cost. This study first compares simple schedules, then examines whether a lightweight controller can improve upon them.

# Datasets, Evaluation, and Reproducibility

[@P058_Yambda5B] provides timestamped music interactions with implicit and explicit feedback. This allows recommendation quality to be studied as new events arrive. At each prediction, the model should only use information available before that point. Violating this order can introduce data leakage and produce misleading results [@P059_DataLeakageStudy].

The candidate set also affects measured quality. Metrics calculated using sampled candidates may differ from those calculated over the full catalog, so comparisons require a consistent candidate selection procedure [@P060_OnSampledMetrics]. Other studies compare simple and neural session-based methods [@P061_SessionEvaluationComparison], examine differences between implementations and training settings [@P062_BERT4RecReplicability], or show how weakly tuned baselines can distort conclusions [@P063_AreWeMakingProgress].

This study compares refresh policies through chronological replay using a fixed recommender and the same candidates at each prediction. Quality is measured with NDCG@10 and Recall@100. Processing time and memory measurements include representation updates, recent-history encoding, scoring, and cache handling. These experiments evaluate offline prediction quality; changes in user satisfaction require separate evaluation.

# Synthesis and Research Gap

Previous work provides several ways to compute and maintain user representations: periodic offline updates [@P031_PinnerFormer], combined long-history and recent-history encoders [@P035_LongHistoryUserTransformers], incremental states [@P024_MIMNUIC], and reuse of intermediate computation [@P037_HSTUMFALCON]. Efficient recurrent models [@P045_LRURec], adaptive sequence length [@P053_SequenceAdaptationviaRL], and learned computation steps [@P068_AdaptiveComputationTime] provide further ways to reduce processing costs.

This study examines these choices in music recommendation. It investigates how prediction quality changes as the cache ages, which simple refresh policies give a useful quality-cost trade-off, and whether a lightweight controller can improve upon them. It also examines when KV-state reuse produces equivalent outputs to full recomputation while reducing update cost.

The experiments use public music data, a fixed recommender, and chronological replay. The potential contribution is to measure when updates are useful and use this information to distribute computation. An adaptive controller is evaluated if the initial experiments show sufficient benefit from refreshing representations.
