# Источники о многостадийных промышленных рекомендательных системах

Проверено 1 ноября 2026. Задача — обосновать предложение во введении о том, что production-рекомендатели используют несколько последовательных стадий отбора и ранжирования. YouTube 2016, COLD и RecFlow добавлены в общий библиографический реестр как `P082_YouTubeDNN`, `P083_COLD` и `P084_RecFlow`; [отдельный BibTeX для Marginalia](../references/imports/multi-stage-recommenders.bib) содержит эти три записи.

## Вывод для введения

Многостадийная архитектура хорошо документирована в промышленных системах разных компаний. Для общего утверждения достаточно сочетания YouTube (Covington et al., 2016), COLD (Wang et al., 2020) и RecFlow (Liu et al., препринт 2024): первая работа даёт классический пример, вторая прямо говорит о распространённости каскада, третья описывает современный полный recommendation funnel. Это обоснование распространённой практики, а не статистическая оценка доли всех production-систем.

При желании уточнить формулировку без изменения смысла:

> Large-scale production recommender systems commonly use multi-stage pipelines, where candidate generation is followed by one or more ranking stages that progressively narrow the candidate set.

Фраза «highest-scoring items from a previous stage» описывает обычный каскад, но не все переходы между стадиями буквально сводятся к top-K одного предыдущего ранкера. Возможны объединение нескольких источников кандидатов, фильтры, ограничения разнообразия и бизнес-правила. В [официальном описании Instagram Explore](https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/) отдельно описаны объединение retrieval sources и final reranking. Этот материал — инженерная статья компании, не научная публикация.

## 1. YouTube: классический двухстадийный рекомендатель

**Paul Covington, Jay Adams, Emre Sargin. Deep Neural Networks for YouTube Recommendations. RecSys 2016.**

Источник: [официальная страница Google Research](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/). Авторский PDF: [Google Research](https://research.google.com/pubs/archive/45530.pdf).

В abstract прямо названы отдельные candidate generation и ranking models:

> “first, we detail a deep candidate generation model and then describe a separate deep ranking model.”

Работа подтверждает разделение retrieval и ranking в реальной системе YouTube. Это прямой первичный источник для архитектурного утверждения во введении; описание относится к опубликованной системе 2016 года, а не к текущему устройству YouTube. В данном проходе подтверждён abstract на официальной странице; PDF по Google URL не загрузился через веб-инструмент.

## 2. Alibaba COLD: прямое утверждение о распространённости каскада

**Zhe Wang, Liqin Zhao, Biye Jiang, Guorui Zhou, Xiaoqiang Zhu, Kun Gai. COLD: Towards the Next Generation of Pre-Ranking System. DLP-KDD 2020 workshop.**

Источники: [arXiv, метаданные и abstract](https://arxiv.org/abs/2007.16122), [полный текст v1](https://arxiv.org/html/2007.16122v1). В метаданных также доступна версия v2 от 17 августа 2020.

Начало abstract:

> “Multi-stage cascade architecture exists widely in many industrial systems such as recommender systems and online advertising”

Introduction и Figure 1 перечисляют matching → pre-ranking → ranking → reranking. Авторы описывают уменьшение множества от десятков тысяч кандидатов на pre-ranking до сотен на ranking и внедрение COLD в рекламной системе Alibaba. Это особенно подходящий источник для слова «typically/common», однако конкретный deployment относится к display advertising. Указывать публикацию как workshop DLP-KDD 2020, не как основную конференцию KDD.

## 3. RecFlow: полный шестистадийный recommendation funnel Kuaishou

**Qi Liu et al. RecFlow: An Industrial Full Flow Recommendation Dataset. ICLR 2025; препринт 2024, arXiv:2410.20868.**

Источники: [полный текст arXiv v1](https://arxiv.org/html/2410.20868v1), [официальный репозиторий авторов](https://github.com/RecFlow-ICLR/RecFlow), [страница финальной публикации OpenReview](https://openreview.net/forum?id=vVHc8bGRns). Прямое открытие OpenReview потребовало browser verification; статус ICLR 2025 подтверждён индексированным текстом [официального PDF](https://openreview.net/pdf?id=vVHc8bGRns). Изложение методов проверено по arXiv, название платформы — также по репозиторию авторов.

Введение:

> “To meet stringent online latency requirements, industrial RS predominantly employs a multi-stage funnel-like pipeline”

Section 2.1 описывает шесть стадий: retrieval → pre-ranking → coarse ranking → ranking → re-ranking → edge ranking, с выходными размерами 8000 → 3000 → 500 → 120 → 10 → 6. Это прямой пример постепенного сужения кандидатов в промышленном feed-рекомендателе; авторский репозиторий связывает данные с Kuaishou. Статья полезна и для будущего обсуждения разницы между оценкой отдельной стадии и всего pipeline.

## 4. JD.com ECM: ещё один независимый промышленный каскад

**Jinbo Song et al. Rethinking Large-scale Pre-ranking System: Entire-chain Cross-domain Models. CIKM 2022, pp. 4495–4499.**

Источники: [arXiv с journal reference и DOI конференции](https://arxiv.org/abs/2310.08039), [полный текст](https://arxiv.org/html/2310.08039v1). DOI: [10.1145/3511808.3557683](https://doi.org/10.1145/3511808.3557683). Год конференции — 2022; размещение arXiv — 2023.

Abstract:

> “Industrial systems such as recommender systems and online advertising, have been widely equipped with multi-stage architectures”

Introduction и Figure 1 описывают matching, pre-ranking, ranking и re-ranking, а также влияние pre-ranking на вход последующего ранкера. Работа выполнена авторами JD.com и подтверждает ту же схему на другом промышленном материале. Конкретная постановка тесно связана с e-commerce advertising, поэтому для введения про музыку её лучше сочетать с YouTube и RecFlow.

## Дополнительный кандидат

**Zhe Zhao et al. Recommending What Video to Watch Next: A Multitask Ranking System. RecSys 2019.** [Официальная страница Google Research](https://research.google/pubs/recommending-what-video-to-watch-next-a-multitask-ranking-system/).

Официальный abstract подтверждает промышленную многоцелевую ranking-систему. Дополнительно проверен [опубликованный текст статьи в PDF-копии](https://daiwk.github.io/assets/youtube-multitask.pdf), соответствующий метаданным Google Research (DOI: 10.1145/3298689.3346997). В Section 1 прямо названа типичная двухстадийная схема candidate generation → ranking; Section 3.1 описывает объединение результатов нескольких алгоритмов генерации кандидатов в один набор перед ранжированием. Это дополнительное прямое подтверждение и иллюстрация того, почему вход ранкера не обязательно является top-K одного предыдущего ранкера.
