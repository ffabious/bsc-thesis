# Синтез для экспериментальной главы

14 ноября 2026. Основа — [индекс завершённой серии](../reports/experiments/autonomous-series.md).
Ниже разделены выводы, которые уже поддержаны результатами, и формулировки,
которые данные не позволяют использовать. Новое обучение для этого синтеза
не проводится; готовые encoder weights и frozen protocol сохранены.

## Постановка и методы

Исследуется next-positive-listen prediction. История содержит все доступные
closed-audio listens, включая короткие; target требует played_ratio_pct>50,
строго прошлую историю и известный item. Текущая timestamp-группа полностью
исключается. После наблюдения события оно может войти в будущие histories,
но encoder и controller weights на evaluation не меняются.

Базовый SASRec-style encoder: last50, item/hidden64,2 блока/2 головы,
FFN64, full CE; обучаемая ID table, без входных аудиопризнаков. Frozen
student-варианты используют прежний single-teacher T1 рецепт. Это сильный
рабочий baseline в нашем протоколе, не установленный music SOTA/quality ceiling.
CE1k обучен на2799860 train targets, catalogue244194,15682304 параметра,
из них53760 вне item table. Само представление — summary последних50
событий, а не всего жизненного профиля пользователя.

Encoder checkpoint выбирается на model_valid. Для controllers CT делится
на8/2/4 дня: fit/early stopping/threshold tuning; PV — исследовательская
внешняя проверка. Простая grid106 включает on-request TTL/event/OR правила,
always и never. Mandatory initial refresh учитывается. Политики развивают
собственный cache state, а не получают искусственный fixed lag на запросе.

Отдельные [fixed-lag replay](../reports/experiments/paired-staleness-replay.md)
сохраняют одни events и candidates для fresh/stale; там уже есть repeat/activity
срезы. Не смешивать fixed24h lag и TTL24h: второе правило даёт разные
фактические ages и обновляется только при запросе.

## Ответ на RQ1: что делает staleness

Cache freshness имеет существенное значение. На development PV суточный
fixed lag теряет примерно половину NDCG для всех четырёх CE/student encoders.
Student fresh gains не превращаются в систематически меньшую потерю
от staleness. Вывод поддержан парными сравнениями и activity slices, а не
raw accuracy разных пользователей/age-групп.

На независимой user cohort TTL24h даёт0.012727 против0.022778 fresh NDCG,
то есть44.1% point loss,95% CI разницы[−0.015847;−0.006196]. TTL30min
даёт0.018263, loss19.8%, CI[−0.006604;−0.002733]. Это эффекты двух
конкретных политик на тех же events; не изолированная причинная роль
возраста или активности во всех возможных settings.

## Ответ на RQ2: какие простые правила конкурентны

Есть воспроизводимый trade-off, но он зависит от serving contract.
При неизменном каталоге, item table и контексте final top100 также
неизменен до refresh. Vector-only cache с полным scoring на каждом hit
не должен быть единственным baseline в таком сценарии. Top100-cache
проверен отдельно и воспроизводит используемые метрики; при сменных
candidates/downstream features применяется другой контракт.

В frozen user-holdout заранее выбранный OR2events/5min имеет point NDCG
0.022642 против0.022778 always и cost1.462ms против2.448ms в list режиме.
Разница NDCG95% CI включает0, но equivalence/non-inferiority не доказана.
Recall0.122226 против0.124637 ниже с CI ниже0. Это конкурентная точка
в зафиксированной grid, не доказательство глобально оптимальной политики.

Top100 — сохранённый результат scoring полного каталога, а не candidate set
из100 items. Recall@100 остаётся нетривиальной метрикой по всем244194 candidates.

Resource scope — serial local MPS B=1 complete pipeline: решение CPU,
encoder/cache/scoring/top100 output, без сети/дискового history loading.
Direct mixed workloads нужны: homogeneous branch sums на H4 ошиблись
до21.3% при редких refresh. Стоимость measured means и actual refresh
fractions публикуется, не скрывается за одним условным compute coefficient.
B=1 latency не является datacenter throughput.

## Ответ на RQ3: помогает ли learned controller

В проверенной постановке надёжного общего преимущества нет. Ridge,
MLP32×32, два MLP seeds на student43, перенос на student42/CE1k и
5-request supervised label не дали устойчивой пользы относительно
сильных simple comparators/CT-selected expected mixtures. Небольшой MSE
gain сам по себе не означает улучшения recommendation quality–cost.

Primary final comparison q25−OR2events/5min: NDCG−0.000726,
97.5% CI[−0.001389;−0.000112]. q75−events10:−0.000622,
CI[−0.001771;+0.000504]. Secondary Recall CI ниже0 в обеих точках.
Costs близки, но не тождественны; в list режиме q75 дешевле events10
примерно на3.9%, что не отменяет явного trade-off по качеству.

H8 добавил четыре дешёвых item cues, но улучшил early-validation MSE лишь
на0.121% при заранее заданном gate1%. Он остановлен до replay. Нельзя
использовать этот probe как доказательство ухудшения NDCG или ненужности
семантических признаков во всех learners. Текущая negative conclusion
ограничена проверенными моделями, labels, features и finite grid.

## KV и интерпретация computational savings

Основной refresh полностью пересчитывает текущий window50. Causal mask
сама по себе не делает projected KV reusable при right-aligned learned
positions. Сдвиг позиций ломает старые K/V, а при eviction retained
higher-layer states сохраняют удалённый контекст даже при fixed positions.
70 CPU feasibility cases поддерживают это различие и корректность rebuild.

Fixed-position append без eviction — positive control с другим reference,
а не измеренный speedup текущего encoder. Не заявлять exact rolling KV
equivalence или resource savings от невалидного update. KV payload50KiB/user,
user vector256B; vector+top100 payload1056B/user без metadata/history.

## Честная итоговая проверка

Ранний encoder search уже открывал старый test. Даже при model_valid
checkpoint selection это не untouched end-to-end test. Новый протокол
опубликован в b590e78 до outcomes;256 uid выбраны по train SHA-sort вне
1000 development users. После открытия модели/пороги не менялись.

Итог:249 active users,236 с known positives,58383 requests,40638 targets
из44779 positives-with-history,90.75% coverage. Применяются frozen CE1k
vocabulary/embeddings; unknown histories→UNK, unknown targets исключены.
Это unseen-user generalization с warm histories и последующим периодом,
а не чистый эффект времени на той же population. User bootstrap также
не устраняет зависимость через общие items и временные факторы. Replay
сохраняет logged histories: он не моделирует изменение будущего поведения
пользователя из-за других рекомендаций и не измеряет online satisfaction.

Два primary NDCG comparisons имеют97.5% intervals каждый; secondary analyses
pointwise95%. Full final frontier descriptive: победители по нему не
возвращаются в selection. H8 и новые variants не оцениваются на этом
holdout. Последняя проверка кода —25 релевантных CPU/MPS tests.

## Формулировка contribution для текста

The study evaluates how cached user representations affect next-item music
recommendation under explicit serving assumptions. It compares fixed refresh
rules and lightweight learned controllers on the same prediction events and
candidate catalog. The evaluation accounts for refreshes, cache initialization,
CPU decisions, and the complete serving path. It also separates vector caching
from caching a final recommendation list when the catalog remains unchanged.

The results show a clear loss from stale representations and competitive
trade-offs for simple refresh rules. The tested learned controllers do not
provide a consistent advantage over these rules. A separate user holdout,
frozen before its outcomes are evaluated, supports this conclusion for the
tested setup. The study also identifies why naive KV reuse does not match full
recomputation in the current positional and sliding-window implementation.

Не заявлять: первое исследование staleness; globally optimal refresh policy;
новый SOTA encoder; equivalence, доказанную только CI с0; production-wide
speedup; superiority всех простых rules над всеми learned controllers.
Empirical contribution — воспроизводимая music evaluation с явным serving
contract, budget accounting, stronger controls и границами negative results.

## Состояние работы

Текущая очередь8 проверок закрыта. Данных достаточно для experimental
setup/results/discussion с перечисленными ограничениями. Новые objectives,
другие архитектуры или RL могут быть отдельными research directions;
они требуют новой постановки и независимого holdout, а не дальнейшего
подбора по зафиксированному H7. Прямое редактирование thesis-local/thesis.tex
в этой серии не выполнялось; этот файл — материал для экспериментальной главы.
