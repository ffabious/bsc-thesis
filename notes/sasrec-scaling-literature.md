# Литература для разбора качества SASRec при переходе с 256 на 1 000 пользователей

7 ноября 2026. Прочитаны первичные статьи и авторский код. Ниже разделены опубликованные результаты, вывод из нашей функции loss и гипотезы о конкретном пилоте. Библиография и обученные модели не изменялись.

## Что непосредственно полезно

| Источник | Проверенный результат | Связь с нашим случаем и предел вывода |
| --- | --- | --- |
| **gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling**, RecSys 2023, DOI 10.1145/3604915.3608783 | В §5 выводится смещение вероятностей при sampled BCE; эмпирически исследуются число negatives и gBCE. В стандартной формуле отрицательные слагаемые **суммируются**, затем вся сумма делится на `K+1`. Авторский PyTorch-код делает `BCE(..., reduction='none').mean(-1)` по общему набору из одного positive и K negatives. | Это наиболее близкая работа о нашем loss, но рекомендации нельзя переносить до проверки reduction. У нас веса positive/negative-групп другие. Из статьи не следует, что любой рост каталога обязательно ухудшает ranking. [Статья, §5](https://arxiv.org/html/2308.07192v1#S5), [авторский train-код, строки 46–66](https://github.com/asash/gSASRec-pytorch/blob/main/train_gsasrec.py#L46-L66). |
| **Repetition and Exploration in Sequential Recommendation**, SIGIR 2023, DOI 10.1145/3539618.3591914 | Авторы разделяют repeat/explore, показывают, что общая accuracy скрывает различия сложности и качества для этих режимов. В §7 проверяют связь shared input/output embeddings с повторными рекомендациями; часть экспериментов специально проводится на pure-exploration данных. | Поддерживает нашу сегментацию: 56% потери NDCG связаны с последним токеном. Но опубликованный эффект обычно связан с **избыточными** повторами; он не доказывает причину нашей потери copy-last. Удалять повторы из разрешающей relistening музыкальной задачи по этой работе нельзя. Их repeat — присутствие в истории, а наш last-token сегмент уже этого определения. [Авторская версия, §§5, 7–8](https://irlab.science.uva.nl/wp-content/papercite-data/pdf/li-2023-repetition.pdf). |
| **RSS: Effective and Efficient Training for Sequential Recommendation using Recency Sampling**, ACM TORS, DOI 10.1145/3604436 | Работа меняет sampling обучающих последовательностей через функцию важности давности и проверяет подход, в том числе на SASRec. Следовательно, выбор обучающих targets рассматривается как часть learning objective, а не только способ экономить время. | Наши event-uniform prefixes по всей train-истории отличаются от fixed recent suffix/user sampling исходного SASRec. Это существенная переменная эксперимента, однако RSS не устанавливает причину конкретного падения на нашем 1k. Статья впервые опубликована онлайн в 2023, журнальный выпуск — 2025; это одна работа. [Авторская университетская запись и abstract](https://eprints.gla.ac.uk/299238/). |
| **Scalable Cross-Entropy Loss for Sequential Recommendations with Large Item Catalogs**, RecSys 2024; уже есть `P057_ScalableCrossEntropy` | В §2.2 явно приведены multi-negative BCE с суммой negatives и sampled CE; SCE приближает CE, выбирая информативные элементы каталога. Эксперименты показывают важность цели обучения и вычислительного бюджета для большого каталога. | Поддерживает приоритет проверки objective/sampling перед заменой encoder. Их BCE+ не совпадает с нашим average-negative loss; их результаты не предсказывают численного улучшения на Yambda. Полный CE/SCE — возможный отдельный эксперимент, а не требуемая сейчас инфраструктура. [Статья, §§2.2, 3–4](https://arxiv.org/html/2409.18721v2#S2.SS2). |
| **Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders**, RecSys 2025, DOI 10.1145/3705328.3748164 | В §3 различаются способы выбора targets при global temporal split; в §4.5 проверяется согласованность validation/test. Для successive evaluation авторы обсуждают предварительное усреднение внутри пользователя, чтобы активные пользователи не определяли весь результат. | Поддерживает сохранение хронологии и раздельную проверку original/added cohorts. Общий event-weighted score 1k и score исходных 256 измеряют разные смеси пользователей. Это не свидетельство ошибки наших границ split и не основание менять их после просмотра test. [Статья, §§3, 4.5](https://arxiv.org/html/2507.16289v1#S3). |
| **Исходный SASRec**, ICDM 2018; уже есть `P011_SASRec` | Авторский sampler выбирает пользователя и recent suffix, создаёт per-position targets и исключает из negatives все train-items пользователя. | Наш baseline — SASRec-style: event-uniform targets, один последний target на prefix, другая политика negatives. Сравнивать epochs или абсолютные метрики с оригиналом напрямую нельзя. Слепо исключать из выдачи все прошлые треки тоже нельзя: в нашей задаче повторы допустимы. [Авторский sampler](https://github.com/kang205/SASRec/blob/master/sampler.py), [статья](https://arxiv.org/abs/1808.09781). |

`P056_RevisitingSASRec` полезна как постановка вопросов, но сама заявляет отсутствие новой модели и является BlueSky position paper. Она не является экспериментальным доказательством, что увеличение пользователей должно вредить качеству. [Текст v2](https://arxiv.org/html/2504.09596v2#S1).

## Главная оговорка: наш K-negative loss не является стандартной K-negative BCE

Локальный код [scripts/sasrec.py](../scripts/sasrec.py) использует

\[
L_{ours}=\ell_+ + \frac{1}{K}\sum_{j=1}^{K}\ell_j^-,
\qquad \ell_+=-\log\sigma(s_+),\quad \ell_j^-=-\log(1-\sigma(s_j^-)).
\]

В gSASRec и BCE+ из SCE используется

\[
L_{BCE}=\frac{1}{K+1}\left(\ell_+ + \sum_{j=1}^{K}\ell_j^-\right).
\]

При `K>1` это различие **относительных весов**, а не общий множитель. Поэтому фраза «у нас 16 negatives» ещё не означает тот режим обучения, для которого в литературе измерено преимущество многих negatives. [gSASRec, Eq. 7](https://arxiv.org/html/2308.07192v1#S3.SS3), [официальная реализация](https://github.com/asash/gSASRec-pytorch/blob/main/train_gsasrec.py#L65-L66), [SCE, Eq. 3](https://arxiv.org/html/2409.18721v2#S2.SS2).

**Следующая алгебра — наш вывод из локального кода, не новый эксперимент и не утверждение статьи о нашем запуске.** Для фиксированного контекста пусть `p_i` — вероятность positive item i, `C` — размер каталога; negatives равномерны с возвращением и исключают только текущий positive. Тогда

\[
\mathbb E[L_{ours,i}]
=-p_i\log q_i-\frac{1-p_i}{C-1}\log(1-q_i),
\qquad
q_i^*=\frac{p_i}{p_i+(1-p_i)/(C-1)}.
\]

K сократился. Увеличение K при сохранении нашего отдельного negative mean улучшает оценку градиента/покрытие, но **не меняет ожидаемую цель**. В идеализированном анализе это соответствует эффективной negative rate `1/(C−1)`, как при одном negative, а не `K/(C−1)`. Это объясняет, почему нельзя просто взять число negatives или параметр gBCE из статьи и считать проблему исправленной.

Важные ограничения этого вывода:

- `q_i*` монотонен по `p_i`: сама формула **не доказывает ухудшение ranking** в идеальном оптимуме. Для реальной модели существенны общие параметры, ограниченная capacity и конечная оптимизация.
- Наши logits вычисляются через стабильный softplus. Overconfidence как гипотеза не означает найденный overflow; сохранённые веса конечны.
- `p_i` здесь — условное распределение следующего positive listen, не вероятность CTR и не causal relevance всех ненаблюдавшихся треков.
- Большее число sampled negatives и другой вес negative-группы — разные изменения. Их нельзя одновременно менять и приписывать эффект только количеству K.

## Что это позволяет сказать о нашем падении

Наблюдения о потере copy-last, изменении смеси validation и сохранении падения на естественных историях взяты из [локального аудита](../reports/experiments/sasrec-1k-audit/history-ablation.json), [validation composition](../reports/experiments/sasrec-1k-audit/validation-composition.json) и [scoring ablations](../reports/experiments/sasrec-1k-audit/scoring-ablation.json). Литература делает **loss weighting, target sampling и repeat/explore balance** обоснованными направлениями проверки; она не устанавливает, какой из них причинно дал наши −30.6% NDCG.

В частности, уменьшение числа negatives из старого каталога **на один исходный positive** не означает, что negatives стало меньше во всём обучении: добавились другие пользователи и contexts. Утверждение о простом «недостатке общего количества negatives» не подтверждается [подсчётами аудита](../reports/experiments/sasrec-1k-audit/objective-changes.json).

## Проверка reduction выполнена

На фиксированных detached logits **одного и того же validation-batch** и том же наборе 16 negatives без `optimizer.step()` проверены gradients текущего loss и BCE по общим `K+1` элементам. Переход к общему mean умножает positive gradient на `1/17`, каждый negative gradient — на `16/17`: относительный вес negative-группы меняется в 16 раз. Это подтверждает различие objectives, не улучшение рекомендаций. [Локальная проверка](../reports/experiments/sasrec-1k-audit/loss-gradient-probe.json).

Если дальше понадобится отдельный обучающий контроль, менять только reduction/relative weighting при фиксированных данных, sampler, K и правилах выбора checkpoint; это уже новый эксперимент, и его улучшение заранее не гарантировано. Ни RSS, ни gBCE, ни full CE, ни history filtering в этой заметке не объявляются готовым исправлением текущей модели.

## Последующий обучающий контроль выполнен

Изменён только reduction на общий mean при прежних данных, K=16 и early stopping. На полном 1k-test NDCG@10 выросла с 0.017327 до 0.020908, Recall@100 — с 0.161678 до 0.194329. Обучение остановилось после 13 эпох, выбран checkpoint эпохи 11. На фиксированной старой когорте разрыв NDCG с 256-контролем уменьшился, но сохранился. Это один seed; данные результата и ограничения сравнения приведены в [отчёте](../reports/experiments/sasrec-1k-joint-bce.md).

8 ноября выполнен и [256-контроль с общим mean](../reports/experiments/sasrec-256-joint-bce.md): NDCG@10 0.026805, Recall@100 0.215659. Он улучшил обе метрики старого 256; при одинаковом новом loss 1k уступает ему на общей когорте по NDCG на 23.4%, Recall — на 5.7%. Переход к общему mean полезен в обоих пилотах, но отставание 1k сохраняется; около 58.3% NDCG-разрыва приходится на повторы последнего трека. Это не выделяет единственную причину: при увеличении пилота меняются также пользователи, словарь, размер item-таблицы и validation.
