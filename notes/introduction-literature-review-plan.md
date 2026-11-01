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

## Literature Review — revised outline and coverage of the 84-paper registry

Рабочий план пересмотрен по просьбе использовать большинство собранных статей. Он сохраняет линию «pipeline → исторические и ранние последовательные методы → self-attention → inference», но выделяет длинную историю, эффективные энкодеры и адаптивное распределение вычислений в самостоятельные разделы.

**Объём:** ориентир — около 15 страниц основного текста (примерно 14–16 после вёрстки), вместо прежних 10–12. При таком охвате статьи следует обсуждать семействами, а не делать 84 отдельных пересказа.

**Покрытие:** 78 работ назначены в основной текст; 6 смежных — в опциональные абзацы или приложение. Все 84 имеют конкретное место в карте ниже. Опциональное назначение не означает, что статья уже процитирована в дипломе. Подробное рассмотрение запланировано для 21 опорной работы; остальные 57 основных работ обсуждаются кратко в сравнительных группах.

**Основа:** распределение опирается на summaries и ограничения текущего реестра, а не на новое полное чтение всех статей. Перед написанием точных утверждений нужно проверить соответствующие разделы первоисточников. Для TransX и REPAIR в реестре отдельно отмечено ограничение глубины чтения.

### 2.1. Introduction and the Recommendation Pipeline
*Ориентир: 1½ pages.*

Определить retrieval, ranking и re-ranking, различия в вычислительных бюджетах и роль user/item representations. Кратко проследить переход к нейросетевому retrieval и ANN-поиску. Различать retrieval объектов из каталога и поиск событий внутри пользовательской истории.

#### 2.1.1. Multi-stage production systems

Промышленный каскад и границы оценки одной стадии.

**Работы:** P082 YouTube DNN; P083 COLD; P084 RecFlow.

**Ключи:** [@P082_YouTubeDNN; @P083_COLD; @P084_RecFlow].

#### 2.1.2. Embedding-based retrieval and candidate search

Поиск по готовым векторам и обучение retrieval-модели; HNSW не является энкодером истории.

**Работы:** P079 HNSW; P080 Sampling-Bias-Corrected Neural Retrieval.

**Ключи:** [@P079_HNSW; @P080_SamplingBiasCorrected].

#### 2.1.3. Ranking and re-ranking

Взаимодействия признаков и отдельный reranker; Jev-Rec — краткий пример, без расширения экспериментов диплома.

**Работы:** P076 DCN-V2; P069 Jev-Rec.

**Ключи:** [@P076_DCNV2; @P069_JevRec].

### 2.2. Historical Approaches and Early Sequential Recommenders
*Ориентир: 1½ pages.*

Сохранить историческую линию пользователя: от несеквенциальных предпочтений и переходов между объектами к явному кодированию порядка действий. Не утверждать, что все ранние методы использовали только агрегированные статистики. Предпосылки collaborative filtering и matrix factorization изложить кратко; для отдельного подробного исторического обзора потребуются дополнительные первичные источники, которых нет в текущем списке.

#### 2.2.1. From static preferences to sequential transitions

FPMC как переход от предпочтений к переходам между объектами; учитывать отличие next-basket от next-item.

**Работы:** P001 FPMC.

**Ключи:** [@P001_FPMC].

#### 2.2.2. Recurrent and hierarchical models

Сессионное состояние, обучение ранжированию и перенос пользовательской памяти между сессиями.

**Работы:** P002 GRU4Rec; P003 GRU4Rec+ / Top-k Gains; P004 HRNN.

**Ключи:** [@P002_GRU4Rec; @P003_GRU4RecTopkGains; @P004_HRNN].

#### 2.2.3. Convolutional sequence models

Локальные зависимости, свёртки и длина учитываемой истории.

**Работы:** P006 Caser; P007 NextItNet.

**Ключи:** [@P006_Caser; @P007_NextItNet].

#### 2.2.4. Attention, gating, and graph-based models

Другие способы агрегирования последовательности; attention вне Transformer и различие сессионной и пользовательской постановок.

**Работы:** P005 NARM; P008 HGN; P009 SHAN; P010 SR-GNN.

**Ключи:** [@P005_NARM; @P008_HGN; @P009_SHAN; @P010_SRGNN].

### 2.3. Self-Attention and Learning User Representations
*Ориентир: 2 pages.*

Объяснить архитектурную основу рекомендателя и отделить изменения цели обучения от изменений политики обслуживания. Организовать сравнение по causal/bidirectional processing, временным признакам, персонализации и обучающим целям, а не пересказывать каждую статью отдельно.

#### 2.3.1. Transformer and sequential attention encoders

Scaled dot-product attention, causal mask, позиционные входы; SASRec и BERT4Rec.

**Работы:** P081 Transformer / Attention Is All You Need; P011 SASRec; P012 BERT4Rec.

**Ключи:** [@P081_AttentionIsAllYouNeed; @P011_SASRec; @P012_BERT4Rec].

#### 2.3.2. Temporal, personalized, and structured representations

Временные интервалы, частотная структура, распределения интересов, персонализация и разреженные данные. SparseSR не считать методом sparse attention.

**Работы:** P013 TiSASRec; P017 FEARec; P018 STOSA; P019 SSE-PT; P020 Coarse-to-Fine Sparse SR.

**Ключи:** [@P013_TiSASRec; @P017_FEARec; @P018_STOSA; @P019_SSEPT; @P020_CoarsetoFineSparseSR].

#### 2.3.3. Self-supervised and contrastive objectives

Как цели обучения влияют на качество представлений; не смешивать эффект обучения и freshness.

**Работы:** P014 S3-Rec; P015 CL4SRec; P016 DuoRec.

**Ключи:** [@P014_S3Rec; @P015_CL4SRec; @P016_DuoRec].

#### 2.3.4. Strong baselines and scalable training

Обоснование настройки SASRec и стоимости обучения на большом каталоге; training cost отличать от стоимости обновления состояния.

**Работы:** P056 Revisiting SASRec; P057 Scalable Cross-Entropy.

**Ключи:** [@P056_RevisitingSASRec; @P057_ScalableCrossEntropy].

### 2.4. Long-History Modeling and History Selection
*Ориентир: 2 pages.*

Показать, какую информацию модель извлекает из длинной истории и что сохраняет при сжатии. Это архитектурная подготовка к вопросу о поддержании кэша. Различать candidate-dependent history processing и универсальный пользовательский вектор.

#### 2.4.1. Candidate-aware interests and session structure

Релевантность кандидату, эволюция интересов и интересы разных сессий.

**Работы:** P021 DIN; P022 DIEN; P023 DSIN.

**Ключи:** [@P021_DIN; @P022_DIEN; @P023_DSIN].

#### 2.4.2. Retrieving relevant historical events

Поиск, хэширование и согласование coarse/exact обработки истории; стоимость выбора событий учитывать отдельно.

**Работы:** P025 SIM; P026 ETA; P027 SDIM; P028 TWIN.

**Ключи:** [@P025_SIM; @P026_ETA; @P027_SDIM; @P028_TWIN].

#### 2.4.3. Compression and ultra-long histories

Кластеры, разделение embedding spaces, сочетание retrieval/recent событий и компактные токены истории.

**Работы:** P029 TWIN V2; P030 DARE; P033 TransAct V2; P039 VISTA.

**Ключи:** [@P029_TWINV2; @P030_DARE; @P033_TransActV2; @P039_VISTA].

### 2.5. Efficient Alternatives to Full Self-Attention
*Ориентир: 1½ pages.*

Разделить уменьшение стоимости одного кодирования и уменьшение частоты кодирования. Сравнить альтернативы по зависимости от длины истории, размеру сохраняемого состояния и возможности инкрементального обновления. Архитектурная эффективность сама по себе не доказывает корректность streaming/KV reuse.

#### 2.5.1. Low-rank and linear attention

Сжатие attention и линейные операторы; проверять ограничения конкретного режима обслуживания.

**Работы:** P042 LightSANs; P044 LinRec; P050 RecGRELA; P051 FuXi-Linear.

**Ключи:** [@P042_LightSANs; @P044_LinRec; @P050_RecGRELA; @P051_FuXiLinear].

#### 2.5.2. Recurrent and state-space models

Компактное обновляемое состояние; отдельно учитывать причинность, двунаправленность и аппаратную реализацию.

**Работы:** P045 LRURec; P046 RecBLR; P047 Mamba4Rec; P048 SSD4Rec; P049 TiM4Rec.

**Ключи:** [@P045_LRURec; @P046_RecBLR; @P047_Mamba4Rec; @P048_SSD4Rec; @P049_TiM4Rec].

#### 2.5.3. MLP and autoencoder alternatives

Фильтрация, MLP и другие способы кодирования; учитывать цену хранения и обновления входного состояния.

**Работы:** P043 FMLP-Rec; P052 AutoSeqRec.

**Ключи:** [@P043_FMLPRec; @P052_AutoSeqRec].

### 2.6. Inference and User Representation Maintenance
*Ориентир: 3 pages.*

Центральный раздел обзора. Сопоставить конечный user vector, компактную память, summary tokens и KV как разные сохраняемые состояния. Раскрыть, когда выполняются вычисления, что именно устаревает и какие затраты остаются вне request path.

#### 2.6.1. Offline representations and real-time recent activity

Периодические представления и две ветви long/recent; зависимость полезности свежести от архитектуры и training objective.

**Работы:** P031 PinnerFormer; P032 TransAct; P034 ARGUS; P035 Long-History User Transformers.

**Ключи:** [@P031_PinnerFormer; @P032_TransAct; @P034_ARGUS; @P035_LongHistoryUserTransformers].

#### 2.6.2. Incremental state, staleness, and refresh schedules

Событийное обновление, активность и нагрузка, восстановление состояния. Разделять обновление user-state, item-state и весов; подробности REPAIR и TransX требуют дочитывания полного текста.

**Работы:** P024 MIMN / UIC; P036 TransX; P040 MuSeR; P041 REPAIR; P054 JODIE; P055 SA-CAISR.

**Ключи:** [@P024_MIMNUIC; @P036_TransX; @P040_MuSeR; @P041_REPAIR; @P054_JODIE; @P055_SACAISR].

#### 2.6.3. Shared computation and KV-state reuse

Reuse между кандидатами и запросами, размещение и память кэша. Перенос LLM memory management на RecSys рассматривать как гипотезу.

**Работы:** P037 HSTU / M-FALCON; P038 RelayGR; P065 PagedAttention / vLLM.

**Ключи:** [@P037_HSTUMFALCON; @P038_RelayGR; @P065_PagedAttentionvLLM].

#### 2.6.4. Kernel efficiency and bounded attention context

Отделить IO/kernel acceleration от ограничения контекста и refresh scheduling; результаты attention sinks/SWA на LLM не переносить на SASRec без проверки.

**Работы:** P064 FlashAttention; P066 StreamingLLM; P067 Sliding-window beats linear attention.

**Ключи:** [@P064_FlashAttention; @P066_StreamingLLM; @P067_Slidingwindowbeatslinearattention].

### 2.7. Adaptive Allocation of Computation
*Ориентир: 1 page.*

Сформулировать решение refresh/keep через ожидаемую пользу и стоимость обновления. Начать с простых расписаний из 2.6 и перейти к адаптивным решениям. RL описывать как возможную альтернативу supervised controller, а не как обязательный метод диплома.

#### 2.7.1. Adaptive sequence length and computation

Адаптация длины истории и числа вычислительных шагов; отличия от частоты refresh.

**Работы:** P053 Sequence Adaptation via RL; P068 Adaptive Computation Time.

**Ключи:** [@P053_SequenceAdaptationviaRL; @P068_AdaptiveComputationTime].

#### 2.7.2. Learning decision policies

REINFORCE и off-policy обучение рекомендательной политики; отделить выбор показываемого объекта от внутреннего управления кэшем и не обещать причинные выводы по offline replay.

**Работы:** P077 REINFORCE (Williams); P078 Top-K Off-Policy REINFORCE.

**Ключи:** [@P077_REINFORCE_Williams; @P078_TopK_OffPolicy_REINFORCE].

### 2.8. Datasets, Evaluation, and Reproducibility
*Ориентир: 1½ pages.*

Обосновать музыкальные данные и оценку качества/стоимости. Установить требования к глобальному времени, доступности событий, candidate pool, повторным прослушиваниям, настройке baselines и общему вычислительному пути. Конкретные гиперпараметры оставить Methodology.

#### 2.8.1. Music interactions and the Yambda benchmark

Типы feedback, временные метки, глобальное разбиение и ограничения offline predictive evaluation.

**Работы:** P058 Yambda-5B.

**Ключи:** [@P058_Yambda5B].

#### 2.8.2. Chronological evaluation and candidate metrics

Утечки, sampled/full-catalog metrics и сравнимость кандидатов.

**Работы:** P059 Data Leakage Study; P060 On Sampled Metrics.

**Ключи:** [@P059_DataLeakageStudy; @P060_OnSampledMetrics].

#### 2.8.3. Baselines and reproducible comparisons

Сильные простые baselines, различия реализаций и полнота настройки; затраты refresh, recent encoder, scoring и cache handling учитывать совместно.

**Работы:** P061 Session Evaluation Comparison; P062 BERT4Rec Replicability; P063 Are We Making Progress?.

**Ключи:** [@P061_SessionEvaluationComparison; @P062_BERT4RecReplicability; @P063_AreWeMakingProgress].

### 2.9. Synthesis and Research Gap
*Ориентир: ½ page.*

Свести ближайшие подходы в таблицу: архитектура, сохраняемое состояние, trigger обновления, recent branch, reuse, оценка свежести и учёт стоимости. Повторно сослаться на опорные работы из предыдущих разделов; новых уникальных статей сюда не назначать. Вывести четыре вопроса диплома и условность benefit controller из проверенных сравнений, не из отсутствия точного аналога названия темы.

### Опциональные смежные работы: ещё 6 источников

Их можно включить все, если соответствующие понятия действительно используются в аргументации. Не добавлять самостоятельный раздел о Jev только ради числа цитирований. Если связь остаётся слабой, сохранить обсуждение в приложении или reading notes. Основной обзор уже покрывает 78 из 84 работ.

- **A.1. Decision calibration and readout:** P070 Jev-Benchmark; P072 AnyJev; P074 OpenJev-RLCD. Возможный короткий смежный абзац для 2.7/2.8: calibration и decision readout. Включать только при использовании соответствующих понятий; это не evidence о refresh user embeddings. [@P070_JevBenchmark; @P072_AnyJev; @P074_OpenJevRLCD].

- **A.2. Shared-prefix inference in another modality:** P071 Visual Jev. Опциональная иллюстрация факторного разделения reuse и batching для 2.6.3; визуальные задачи, не рекомендации. [@P071_VisualJev].

- **A.3. Controllers for agentic memory:** P073 Jev-Mem. Смежный prior art для 2.7: benefit/budget decisions при работе с агентной памятью. Не приравнивать память агента к состоянию рекомендателя. [@P073_JevMem].

- **A.4. Training-data selection:** P075 SAGE. Отдельная смежная заметка/приложение: SAGE выбирает обучающую подвыборку, а не refresh schedule. Сохранить ограничения локальной диагностики; в основной текст включать только при реальном обсуждении training-data efficiency. [@P075_SAGE].

### Карта покрытия: основное место каждой статьи

Каждой статье назначено ровно одно основное место, чтобы не завышать покрытие повторными ссылками. Повторные ссылки на ближайшие работы в 2.9 и других тематических разделах допустимы и полезны.

| ID | Ключ BibTeX | Работа | Основное место | Глубина |
| --- | --- | --- | --- | --- |
| P001 | `P001_FPMC` | FPMC | 2.2.1 — From static preferences to sequential transitions | Групповой обзор |
| P002 | `P002_GRU4Rec` | GRU4Rec | 2.2.2 — Recurrent and hierarchical models | Групповой обзор |
| P003 | `P003_GRU4RecTopkGains` | GRU4Rec+ / Top-k Gains | 2.2.2 — Recurrent and hierarchical models | Групповой обзор |
| P004 | `P004_HRNN` | HRNN | 2.2.2 — Recurrent and hierarchical models | Групповой обзор |
| P005 | `P005_NARM` | NARM | 2.2.4 — Attention, gating, and graph-based models | Групповой обзор |
| P006 | `P006_Caser` | Caser | 2.2.3 — Convolutional sequence models | Групповой обзор |
| P007 | `P007_NextItNet` | NextItNet | 2.2.3 — Convolutional sequence models | Групповой обзор |
| P008 | `P008_HGN` | HGN | 2.2.4 — Attention, gating, and graph-based models | Групповой обзор |
| P009 | `P009_SHAN` | SHAN | 2.2.4 — Attention, gating, and graph-based models | Групповой обзор |
| P010 | `P010_SRGNN` | SR-GNN | 2.2.4 — Attention, gating, and graph-based models | Групповой обзор |
| P011 | `P011_SASRec` | SASRec | 2.3.1 — Transformer and sequential attention encoders | Подробно |
| P012 | `P012_BERT4Rec` | BERT4Rec | 2.3.1 — Transformer and sequential attention encoders | Групповой обзор |
| P013 | `P013_TiSASRec` | TiSASRec | 2.3.2 — Temporal, personalized, and structured representations | Групповой обзор |
| P014 | `P014_S3Rec` | S3-Rec | 2.3.3 — Self-supervised and contrastive objectives | Групповой обзор |
| P015 | `P015_CL4SRec` | CL4SRec | 2.3.3 — Self-supervised and contrastive objectives | Групповой обзор |
| P016 | `P016_DuoRec` | DuoRec | 2.3.3 — Self-supervised and contrastive objectives | Групповой обзор |
| P017 | `P017_FEARec` | FEARec | 2.3.2 — Temporal, personalized, and structured representations | Групповой обзор |
| P018 | `P018_STOSA` | STOSA | 2.3.2 — Temporal, personalized, and structured representations | Групповой обзор |
| P019 | `P019_SSEPT` | SSE-PT | 2.3.2 — Temporal, personalized, and structured representations | Групповой обзор |
| P020 | `P020_CoarsetoFineSparseSR` | Coarse-to-Fine Sparse SR | 2.3.2 — Temporal, personalized, and structured representations | Групповой обзор |
| P021 | `P021_DIN` | DIN | 2.4.1 — Candidate-aware interests and session structure | Групповой обзор |
| P022 | `P022_DIEN` | DIEN | 2.4.1 — Candidate-aware interests and session structure | Групповой обзор |
| P023 | `P023_DSIN` | DSIN | 2.4.1 — Candidate-aware interests and session structure | Групповой обзор |
| P024 | `P024_MIMNUIC` | MIMN / UIC | 2.6.2 — Incremental state, staleness, and refresh schedules | Подробно |
| P025 | `P025_SIM` | SIM | 2.4.2 — Retrieving relevant historical events | Подробно |
| P026 | `P026_ETA` | ETA | 2.4.2 — Retrieving relevant historical events | Групповой обзор |
| P027 | `P027_SDIM` | SDIM | 2.4.2 — Retrieving relevant historical events | Групповой обзор |
| P028 | `P028_TWIN` | TWIN | 2.4.2 — Retrieving relevant historical events | Групповой обзор |
| P029 | `P029_TWINV2` | TWIN V2 | 2.4.3 — Compression and ultra-long histories | Подробно |
| P030 | `P030_DARE` | DARE | 2.4.3 — Compression and ultra-long histories | Групповой обзор |
| P031 | `P031_PinnerFormer` | PinnerFormer | 2.6.1 — Offline representations and real-time recent activity | Подробно |
| P032 | `P032_TransAct` | TransAct | 2.6.1 — Offline representations and real-time recent activity | Подробно |
| P033 | `P033_TransActV2` | TransAct V2 | 2.4.3 — Compression and ultra-long histories | Подробно |
| P034 | `P034_ARGUS` | ARGUS | 2.6.1 — Offline representations and real-time recent activity | Подробно |
| P035 | `P035_LongHistoryUserTransformers` | Long-History User Transformers | 2.6.1 — Offline representations and real-time recent activity | Подробно |
| P036 | `P036_TransX` | TransX | 2.6.2 — Incremental state, staleness, and refresh schedules | Подробно |
| P037 | `P037_HSTUMFALCON` | HSTU / M-FALCON | 2.6.3 — Shared computation and KV-state reuse | Подробно |
| P038 | `P038_RelayGR` | RelayGR | 2.6.3 — Shared computation and KV-state reuse | Подробно |
| P039 | `P039_VISTA` | VISTA | 2.4.3 — Compression and ultra-long histories | Групповой обзор |
| P040 | `P040_MuSeR` | MuSeR | 2.6.2 — Incremental state, staleness, and refresh schedules | Подробно |
| P041 | `P041_REPAIR` | REPAIR | 2.6.2 — Incremental state, staleness, and refresh schedules | Групповой обзор |
| P042 | `P042_LightSANs` | LightSANs | 2.5.1 — Low-rank and linear attention | Групповой обзор |
| P043 | `P043_FMLPRec` | FMLP-Rec | 2.5.3 — MLP and autoencoder alternatives | Групповой обзор |
| P044 | `P044_LinRec` | LinRec | 2.5.1 — Low-rank and linear attention | Групповой обзор |
| P045 | `P045_LRURec` | LRURec | 2.5.2 — Recurrent and state-space models | Подробно |
| P046 | `P046_RecBLR` | RecBLR | 2.5.2 — Recurrent and state-space models | Групповой обзор |
| P047 | `P047_Mamba4Rec` | Mamba4Rec | 2.5.2 — Recurrent and state-space models | Групповой обзор |
| P048 | `P048_SSD4Rec` | SSD4Rec | 2.5.2 — Recurrent and state-space models | Групповой обзор |
| P049 | `P049_TiM4Rec` | TiM4Rec | 2.5.2 — Recurrent and state-space models | Групповой обзор |
| P050 | `P050_RecGRELA` | RecGRELA | 2.5.1 — Low-rank and linear attention | Групповой обзор |
| P051 | `P051_FuXiLinear` | FuXi-Linear | 2.5.1 — Low-rank and linear attention | Групповой обзор |
| P052 | `P052_AutoSeqRec` | AutoSeqRec | 2.5.3 — MLP and autoencoder alternatives | Групповой обзор |
| P053 | `P053_SequenceAdaptationviaRL` | Sequence Adaptation via RL | 2.7.1 — Adaptive sequence length and computation | Групповой обзор |
| P054 | `P054_JODIE` | JODIE | 2.6.2 — Incremental state, staleness, and refresh schedules | Групповой обзор |
| P055 | `P055_SACAISR` | SA-CAISR | 2.6.2 — Incremental state, staleness, and refresh schedules | Групповой обзор |
| P056 | `P056_RevisitingSASRec` | Revisiting SASRec | 2.3.4 — Strong baselines and scalable training | Подробно |
| P057 | `P057_ScalableCrossEntropy` | Scalable Cross-Entropy | 2.3.4 — Strong baselines and scalable training | Групповой обзор |
| P058 | `P058_Yambda5B` | Yambda-5B | 2.8.1 — Music interactions and the Yambda benchmark | Подробно |
| P059 | `P059_DataLeakageStudy` | Data Leakage Study | 2.8.2 — Chronological evaluation and candidate metrics | Подробно |
| P060 | `P060_OnSampledMetrics` | On Sampled Metrics | 2.8.2 — Chronological evaluation and candidate metrics | Подробно |
| P061 | `P061_SessionEvaluationComparison` | Session Evaluation Comparison | 2.8.3 — Baselines and reproducible comparisons | Групповой обзор |
| P062 | `P062_BERT4RecReplicability` | BERT4Rec Replicability | 2.8.3 — Baselines and reproducible comparisons | Групповой обзор |
| P063 | `P063_AreWeMakingProgress` | Are We Making Progress? | 2.8.3 — Baselines and reproducible comparisons | Групповой обзор |
| P064 | `P064_FlashAttention` | FlashAttention | 2.6.4 — Kernel efficiency and bounded attention context | Групповой обзор |
| P065 | `P065_PagedAttentionvLLM` | PagedAttention / vLLM | 2.6.3 — Shared computation and KV-state reuse | Групповой обзор |
| P066 | `P066_StreamingLLM` | StreamingLLM | 2.6.4 — Kernel efficiency and bounded attention context | Групповой обзор |
| P067 | `P067_Slidingwindowbeatslinearattention` | Sliding-window beats linear attention | 2.6.4 — Kernel efficiency and bounded attention context | Групповой обзор |
| P068 | `P068_AdaptiveComputationTime` | Adaptive Computation Time | 2.7.1 — Adaptive sequence length and computation | Групповой обзор |
| P069 | `P069_JevRec` | Jev-Rec | 2.1.3 — Ranking and re-ranking | Групповой обзор |
| P070 | `P070_JevBenchmark` | Jev-Benchmark | A.1 — Decision calibration and readout | Опционально |
| P071 | `P071_VisualJev` | Visual Jev | A.2 — Shared-prefix inference in another modality | Опционально |
| P072 | `P072_AnyJev` | AnyJev | A.1 — Decision calibration and readout | Опционально |
| P073 | `P073_JevMem` | Jev-Mem | A.3 — Controllers for agentic memory | Опционально |
| P074 | `P074_OpenJevRLCD` | OpenJev-RLCD | A.1 — Decision calibration and readout | Опционально |
| P075 | `P075_SAGE` | SAGE | A.4 — Training-data selection | Опционально |
| P076 | `P076_DCNV2` | DCN-V2 | 2.1.3 — Ranking and re-ranking | Групповой обзор |
| P077 | `P077_REINFORCE_Williams` | REINFORCE (Williams) | 2.7.2 — Learning decision policies | Групповой обзор |
| P078 | `P078_TopK_OffPolicy_REINFORCE` | Top-K Off-Policy REINFORCE | 2.7.2 — Learning decision policies | Групповой обзор |
| P079 | `P079_HNSW` | HNSW | 2.1.2 — Embedding-based retrieval and candidate search | Групповой обзор |
| P080 | `P080_SamplingBiasCorrected` | Sampling-Bias-Corrected Neural Retrieval | 2.1.2 — Embedding-based retrieval and candidate search | Групповой обзор |
| P081 | `P081_AttentionIsAllYouNeed` | Transformer / Attention Is All You Need | 2.3.1 — Transformer and sequential attention encoders | Групповой обзор |
| P082 | `P082_YouTubeDNN` | YouTube DNN | 2.1.1 — Multi-stage production systems | Подробно |
| P083 | `P083_COLD` | COLD | 2.1.1 — Multi-stage production systems | Подробно |
| P084 | `P084_RecFlow` | RecFlow | 2.1.1 — Multi-stage production systems | Подробно |

### Как использовать распределение при написании

- Для каждой группы сформулировать общий вопрос, затем сравнить решения, доказательства и ограничения; ссылкой поддерживать конкретное утверждение.
- У опорных работ описать архитектуру, сохраняемое состояние, update policy и границы результатов. Для группового обзора достаточно короткого сравнения с точными ссылками.
- Обсуждение training objectives не превращать в обещание менять checkpoint между refresh policies.
- Разделять retrieval объектов из каталога, retrieval событий истории и хранение итоговых представлений.
- Не считать causal attention, linear attention или название incremental достаточным доказательством корректного reuse.
- Не выводить качество рекомендаций из одного cosine drift и не сравнивать опубликованные speedups как результаты на одинаковом workload.
- Не обещать все рассмотренные модели в экспериментальной части. Литературное покрытие не расширяет утверждённый scope экспериментов.

## Writing guardrails

- Ground the introduction and gap in the 9 October proposal and the reviewed sources; earlier notes are search leads only.
- Distinguish published findings from hypotheses and proposed experiments. Do not imply that the thesis has already demonstrated an improvement.
- Do not claim novelty for caching, combining long and recent histories, or using a controller by itself. State the precise proposed distinction and check it against close prior work.
- Keep industrial online deployment claims separate from this thesis's planned offline evaluation.
- Structure the Literature Review as orientation, thematic synthesis/evaluation, and a conclusion that justifies the selected method.
