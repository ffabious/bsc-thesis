# Thesis discussion record: efficient inference for large recommendation models

This document records the substantive discussion so far. It separates the user's circumstances, previously reported research findings, our proposed research direction, and unresolved questions. It is a discussion record, not a new literature review or a verbatim transcript. No final thesis topic has been chosen, and no experiments or implementation have been performed.

## How the discussion started

The user asked whether skills can be used in voice mode; the assistant answered yes. The broader aim was to choose a thesis collaboratively, taking account of the user's interests, day job, and research papers rather than selecting a topic in isolation.

The user works in machine learning, particularly ranking and recommender systems. Their interest is faster and cheaper inference for large, potentially transformer-based models in real-time environments. They described a practical tension: large models are mostly usable offline, while online models tend to be relatively small because of computational cost.

An early statement suggested working mostly with the published ARGUS architecture. The user then explicitly clarified that **they personally do not work with ARGUS yet**. That correction governs this record. We have no basis to assume hands-on ARGUS experience, access to its production infrastructure, a suitable checkpoint, or an existing cache implementation.

The user requested a public-source investigation of which approaches people use and how, with a report. They also recalled an August paper comparing sliding-window and linear attention, particularly inference changes that do not require retraining.

## Research discussed: reported findings and their limits

The following summarizes the earlier investigation. Results are the authors' reports, not independently reproduced findings. Speedups and resource savings across papers are not directly comparable because workloads, hardware, baselines, and measurement methods differ.

### Sliding-window attention without retraining

[Sliding-window beats linear attention](https://arxiv.org/html/2608.28444v1), by Alexia Jolicoeur-Martineau, Rhea Sanjay Sukthanker, Pashmina Cameron, and Emy Gervais (August 28, 2026), compares training-free sliding-window attention with four initial attention sinks against post-training conversions of pretrained full-attention LLMs to linear attention. It competes with or beats most such conversions; this does not establish superiority over linear attention trained from scratch.

The reported average baseline recovery is approximately 99% across six short-context benchmarks, with the best nonteacher average in 9 of 11 model cases. However, BABILong at 4K gives SWA 15%, LoLCATs 3%, and full attention 60%: outperforming converted models can still involve substantial quality loss.

Speed measurements use a four-layer transformer, batch size one, and one RTX PRO 6000 Blackwell Max-Q GPU, measuring decoding throughput and cache/state memory rather than production ranking latency. SWA bounds attention context/state without weight updates. These results do not establish equivalent quality retention for ARGUS.

### ARGUS and its published serving arrangement

[Scaling Recommender Transformers to One Billion Parameters](https://arxiv.org/html/2507.15994v1), by Kirill Khrylchenko, Artem Matveev, Sergei Makeev, and Vladimir Baikalov (July 21, 2025), describes ARGUS: a causal transformer over user context, items, and feedback histories. Pretraining combines next-item and feedback prediction, with a simplified representation using one embedding per interaction.

The fine-tuned two-tower setup computes user and item embeddings offline daily. Serving uses dot products and features in a downstream ranker. The reported deployed encoder has 126 million parameters and an 8,192-event history, compared with an earlier real-time encoder of 3.2 million parameters and 1,024 events. Scaling experiments reach approximately one billion encoder parameters; these counts should not be confused with total embedding parameters. Longer histories improve ranking.

The published setup uses learned absolute positional embeddings. Valid cache reuse and the policy for positions as a window advances need investigation; neither is assumed solved.

### Pinterest: select history and optimize serving

[TransAct V2](https://arxiv.org/html/2506.02267v1) selects candidate-relevant historical actions through nearest-neighbor retrieval and combines them with recent actions. This reduces roughly 10,000 history events to several hundred inputs for a small transformer. Serving optimizations include feature deduplication, int8 feature quantization, fused kernels, batching, and reduced data transfer. The authors report production deployment. The approach illustrates that history selection and serving engineering can both matter to practical inference cost.

### Meta: share history computation across candidates

[Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations](https://arxiv.org/html/2402.17152v3) describes HSTU and M-FALCON. Candidate microbatching and reuse of shared history and encoder KV caches across passes or requests help make models with large FLOP requirements practical. Reusing intermediate history calculations is distinct from caching a final user embedding, as in the published ARGUS serving arrangement.

### LinkedIn: incremental nearline encoding and online queries

[TransX](https://arxiv.org/html/2607.28940v1) incrementally encodes behavior nearline and caches representations per viewer. Online candidate queries use recent representations and a pooled long-term summary through sparse cross-attention, sharing KV within a request. The authors report production A/B testing and approximately 80% less online compute. Nearline work still incurs cost. This is relevant prior art: incremental processing and a combination of local and global history are already established ideas, constraining what would count as a novel thesis contribution.

### RelayGR: schedule computation and place caches

[RelayGR](https://arxiv.org/abs/2601.01712) pre-infers history during earlier pipeline stages, retains KV in accelerator memory, and routes ranking to the same instance. It also uses selective admission and host-memory reuse. Ascend NPU measurements report up to 1.5 times longer sequences and 3.6 times the SLO-compliant throughput. This highlights scheduling and cache placement as performance mechanisms alongside the attention computation itself.

### An existing ARGUS adaptation

Aleksandr Semikras's 2026 HSE thesis, [Recommendation Pipeline Modification Using Argus](https://www.hse.ru/en/edu/vkr/1163036181), concerns late ranking in Yandex.Images. Its public abstract describes inference under real-time constraints and offline experiments. It does not provide sufficient evidence to establish a particular cache design or a production rollout. It is relevant prior art against claiming that simply applying ARGUS to another domain is a new contribution.

### A practical implementation reference

The [NVIDIA HSTU Triton serving workflow](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tutorials/Popular_Models_Guide/HSTU/README.html) provides implementation context, including AOTInductor, KV caching, and serving examples. It is an HSTU reference, not an ARGUS implementation.

## Our synthesis of the research

Our interpretation is that many systems make large recommenders practical by reusing expensive history computation, moving it out of the request path, and limiting work per candidate. This suggests separating three design questions:

| Design question | Choices to investigate |
| --- | --- |
| What can be reused? | Final user embeddings, intermediate history states, or computation shared across candidates. |
| What can be discarded? | Old or redundant events, or attention connections that contribute little to ranking quality. |
| When should computation happen? | Periodically, when an event arrives, or when a scoring request arrives. |

Sliding-window attention primarily addresses the discard question. Choosing an attention rule alone does not resolve the reuse and scheduling questions.

## How the proposed thesis direction evolved

The discussion moved through three related formulations:

1. Allocate an attention budget while retaining ranking quality.
2. Retain recent actions and selectively keep useful older events.
3. Investigate efficient incremental ARGUS inference with bounded attention memory, subject to validating model access and technical feasibility.

The current provisional title is:

**Quality-latency trade-offs in incremental inference for large sequential recommenders under a bounded memory budget.**

In simpler language: **make large recommender models fast enough to use fresh user activity.**

This remains a candidate direction rather than an agreed final topic or a commitment to use ARGUS.

### Working hypothesis

Keeping recent user events, a small selection of important older events, and reusable intermediate calculations may improve the trade-off between recommendation quality, freshness, and computational cost.

This hypothesis has not been tested. Its novelty has not been established. Caching and combining recent with longer-term information are not, by themselves, a sufficient contribution.

### Possible research contribution

A defensible contribution could be a specific rule for choosing which history to retain or when to refresh cached representations, together with evidence of the conditions under which that rule helps. The thesis would need to identify an improvement over appropriate existing approaches and explain its limits, rather than present reuse itself as new.

## Proposed evaluation plan

The following is a proposal only. No baseline has been implemented or measured.

### Baselines to compare

- Full-history recomputation.
- Incremental inference with a full KV cache, where reuse is valid for the model and input construction.
- A recent-event window, with and without retained initial tokens.
- Recent events plus selected older events.
- Periodically refreshed offline user representations.

### Measurements

- Recommendation or ranking quality.
- p95 and p99 request latency.
- Throughput.
- Memory per active user.
- Refresh and nearline computation cost.

The workload should include cache misses and variation in user activity, rather than only timing warm-cache requests. The quality/freshness benefit should be assessed alongside the total work required to maintain representations.

## Limits and unresolved assumptions

- It is not established that attention is the dominant bottleneck for the eventual model and serving workload.
- Whether initial attention sinks transfer usefully to recommendation histories is unverified.
- Incremental reuse may depend on positional embeddings, input construction, and whether previously computed states remain valid. These require explicit checks.
- The appropriate division between history encoding and candidate-dependent computation remains to be specified.
- We do not know whether the user's eventual system would recompute histories or reuse caches. That question was raised but remains unanswered because the user has not yet worked with ARGUS.
- We have not established a dataset, checkpoint, implementation, compute allocation, or realistic serving target.
- Published results discussed above do not establish the outcome of the proposed recommender experiments.

## Questions for the next discussion

1. What degree is the thesis for, what is the deadline, and how much work can reasonably fit within its scope?
2. Is there a chronological interaction dataset suitable for measuring ranking quality and the effect of fresh activity?
3. Which model and checkpoint can actually be accessed and modified? Is ARGUS feasible, or should the question first be studied on another sequential recommender?
4. What GPU resources and experimental budget are available?
5. What would the model consume at serving time: user history alone, candidate inputs, or both? Which computation would be shared across candidates?
6. What latency, throughput, memory, and freshness requirements would make the experiment realistic?
7. Under that model's input and position policy, which states can be reused correctly, and when must they be refreshed?
8. Which concrete retention or refresh policy offers a plausible improvement over the proposed baselines?

## Documents produced and current status

The user requested a short, simple Markdown summary. It was delivered in chat and saved as [thesis-ideas.md](/Users/k-greshnov/Documents/thesis/thesis-ideas.md).

This separate document, [thesis-discussion.md](/Users/k-greshnov/Documents/thesis/thesis-discussion.md), preserves the fuller discussion, corrections, research context, proposals, and open questions. The short summary remains unchanged. No new research was conducted to prepare this record.
