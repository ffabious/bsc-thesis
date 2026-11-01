# Plan for the Introduction and Literature Review

This plan follows the thesis proposal dated 9 October 2026. Its scope is the maintenance of user representations for next-item music recommendation: a SASRec-style model with a cached long-history representation and a small recent-history encoder, evaluated through chronological offline replay on Yambda-50M.

Earlier notes on sliding-window attention and ARGUS are useful background, but they do not replace the current proposal's scope. The literature review should establish the gap from the cited prior work rather than assume that the proposed method is novel.

## Introduction — target: 3–4 pages

1. **Background and motivation (½–¾ page).** Explain how sequential recommenders use interaction histories to predict future items. Longer histories can provide useful context but cost more to encode.
2. **The freshness problem (½–¾ page).** Introduce the trade-off between recomputing user representations after new activity and serving from a stale cached representation. State the central quality-versus-computation problem.
3. **Existing approaches and the open question (about ½ page).** Briefly introduce long-/short-term history architectures, incremental updates, and reuse of intermediate computation. Motivate the question of when a refresh is useful for music next-item recommendation. Treat this as a provisional research gap to substantiate in the Literature Review.
4. **Aim, research questions, and hypothesis (about ½ page).** State the aim: reduce representation-maintenance computation while preserving recommendation quality. Questions:
   - How does recommendation quality change as the cached representation becomes older?
   - Which simple refresh schedules offer a useful quality–cost trade-off?
   - Can a lightweight controller identify when a refresh is useful?
   - Under what conditions can transformer KV states be reused correctly and efficiently?

   Working hypothesis: the value of refreshing a representation varies across users and prediction opportunities.
5. **Method and scope (about ½ page).** Preview Yambda-50M, the SASRec-style two-branch model, chronological replay using only past events, and comparisons of refresh policies with the recommender held fixed. Clarify that the study is an offline predictive evaluation.
6. **Expected contribution and thesis outline (¼–½ page).** Describe the planned comparison of quality and total maintenance cost, analysis of staleness, and a possible adaptive controller if initial experiments show that freshness matters. Present the controller as a candidate contribution, not an established result. End with a short chapter roadmap.

## Literature Review — target: 10–12 pages

| Section | Target | Content and role in the argument |
| --- | ---: | --- |
| 1. Scope and review framework | ½ page | Define the review's focus and organize prior work by representation, update mechanism, reuse, and evaluation of quality and cost. |
| 2. Sequential recommendation and the base encoder | 1–1½ pages | Briefly trace sequential recommendation models and motivate SASRec as the experimental starting point. Cover temporal signals or alternative architectures only where relevant to the study. |
| 3. Long histories and multi-timescale representations | 1½–2 pages | Compare approaches that combine long-term context with recent events, including PinnerFormer, TransAct, and Long-History User Transformers. Discuss architectural choices and limitations. |
| 4. Computing and updating user state | 2–2½ pages | Review periodic recomputation, event-driven/incremental updates, and nearline processing, including MIMN/UIC, TransX, ARGUS, and MuSeR. Distinguish updating a user representation from sharing computations across candidates or requests. |
| 5. Reuse of computation and KV states | 1–1½ pages | Discuss HSTU/M-FALCON and related systems. Explain the conditions for valid reuse and invalidation when history, positions, or model parameters change. |
| 6. Adaptive allocation of computation | 1–1½ pages | Compare activity-based and load-based schedules, refresh controllers, and adaptive computation ideas. Separate recommender-system precedents from methods transferred from other domains. Frame benefit-aware refresh as a potential contribution whose novelty requires comparison with prior work. |
| 7. Data and evaluation protocol | 1–1½ pages | Introduce Yambda and relevant properties of music interactions. Discuss chronological splits and replay without leakage, NDCG@10 and Recall@100, and total cost including refresh and cache handling. Explain why quality and cost must be assessed together. |
| 8. Synthesis, gap, and method choice | 1 page | Summarize prior work in a comparison table (update policy, freshness, reuse, cost, limitations), state the defensible research gap, and connect it to the planned baselines, optional controller, and KV-reuse experiment. |

The total is approximately 10½–12 pages. Sections 3–6 should carry most of the discussion; avoid turning the chapter into a catalogue of all papers in the registry. Work on event retrieval, alternative encoders, and controllers from adjacent domains can be treated briefly as alternatives or scope boundaries.

## Writing guardrails

- Ground the introduction and gap in the 9 October proposal and the reviewed sources; earlier notes are search leads only.
- Distinguish published findings from hypotheses and proposed experiments. Do not imply that the thesis has already demonstrated an improvement.
- Do not claim novelty for caching, combining long and recent histories, or using a controller by itself. State the precise proposed distinction and check it against close prior work.
- Keep industrial online deployment claims separate from this thesis's planned offline evaluation.
- Structure the Literature Review as orientation, thematic synthesis/evaluation, and a conclusion that justifies the selected method.
