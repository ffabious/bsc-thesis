# Автономная серия: результаты и решения

14 ноября 2026. Восемь обоснованных проверок по экспериментальному плану
завершены локально. Encoder weights переиспользованы; обучались небольшие
контроллеры. Все изменения сохраняются отдельными коммитами в origin/main.
[Очередь и критерии](../../notes/autonomous-experiments.md),
[синтез для диплома](../../notes/experimental-synthesis.md).

## Что установлено

1. Learned refresh в проверенном виде не даёт устойчивого преимущества
   перед простыми правилами. Результат повторился на student42/43, CE1k
   и отдельной frozen user cohort. Это вывод о проверенных learners/features,
   не об отсутствии полезных adaptive policies вообще.
2. При fixed full catalog нужно учитывать кэш готового top100. На новом
   user-holdout events3 занимает0.890мс вместо1.942мс при тех же метриках.
   Смена candidate set, item table или recent/context ветви меняет контракт.
3. Staleness снижает качество. В новом holdout TTL24ч теряет44.1% fresh
   NDCG по point estimate; paired95% CI ниже0. Сравнение выполнено на одних
   events, с одинаковыми кандидатами, а не между разными группами users.
4. Naive KV reuse не эквивалентен текущим right-aligned learned positions
   и sliding window. Все refresh в основной серии полностью пересчитывают
   encoder. Positive fixed-position controls и rebuild проходят проверку.
5. Старый test открывался при encoder search. Для новой проверки протокол
   опубликован до outcomes, users выбраны только по train и не пересекаются
   с1000 development uid. Этот holdout больше не участвует в выборе моделей.

## Проверки по гипотезам

| Этап | Результат | Артефакт | Коммит этапа |
| --- | --- | --- | --- |
| H1: плотнее простая grid |6 правил закрыли один из двух ridge frontier points; оставшийся gain неубедителен |[Отчёт](student43-dense-policies.md)|1703b68|
| H2: nonlinear predictor |MLP MSE ниже ridge на1.3–1.5%, но NDCG CI к CT cost-matched mixtures включают0 |[Отчёт](controller-models.md)|639c4cb|
| H3: future-aware labels |Nonzero8.9%→22.2%;5-request label не дал подтверждённого quality–cost выигрыша |[Отчёт](controller-horizon.md)|3d7c40b|
| H4: static top100 cache |218 metric rows совпали;436 direct workload cases заменили невалидную additive model |[Отчёт](static-list-cache.md)|eb347be|
| H5: transfer |Второй student seed/CE1k не подтвердили общий controller gain; static-list экономия повторилась |[Отчёт](policy-transfer.md)|c01d9c4|
| H6: KV feasibility |70 ожидаемых исходов,55 совпадений/15 несовпадений;50KiB KV payload против256B vector |[Отчёт](kv-reuse-feasibility.md)|cd6c6bb|
| H7: frozen user-holdout |40638 targets; high-mode primary NDCG97.5% CI ниже0, Recall ниже в обеих точках |[Отчёт](final-user-holdout.md)|b590e78 freeze; ddb4f39 results|
| H8: item cues |Early MSE−0.121% при gate−1%; остановлен до PV replay/serving |[Отчёт](content-controller-probe.md)|4390ad6 protocol/code; результаты в текущей версии|

## Итоговая frozen проверка

256 заранее выбранных users вне development;249 активных,236 с known
positive targets.58383 requests,40638 targets, known-positive coverage90.75%.
Frozen CE1k catalogue244194, history50, current timestamp group исключена.

| Primary comparison | Delta NDCG@10 |97.5% user-bootstrap CI|
| --- | ---: | --- |
|q75−events10|−0.000622|[−0.001771;+0.000504]|
|q25−hybrid2events/5min|−0.000726|[−0.001389;−0.000112]|

Secondary Recall95% CI ниже0 в обоих сравнениях. Primary family2 и уровни
intervals записаны до outcomes. Costs сравнимы, но не объявляются точными
равными budgets: timing шум и сдвиг частоты запросов публикуются отдельно.

В static-list режиме hybrid2events/5min даёт0.022642 NDCG против0.022778
fresh при1.462мс против2.448мс. Это99.4% NDCG point retention за59.7% cost;
non-inferiority не доказана и заранее не проверялась. Recall ниже fresh.

## Воспроизводимость и границы

Последняя проверка:25 релевантных CPU/MPS regression tests прошли.
H7 содержит228 независимо сверенных bootstrap intervals,216 paired serving
cases и CPU full-rank oracle256 пар с одинаковыми метриками/top100.
Повторный запуск final evaluation блокируется до чтения новых event rows
и сохраняет готовые outputs. Data/checkpoint/source hashes и protocol commit
записаны в JSON. H8 проверен только на development CT и не получает H7 files.

Raw events, checkpoints, snapshot lists и ranks остаются в ignored data/.
В Git находятся код, configs, компактные JSON/CSV, PNG/SVG и отчёты.
Исторические отчёты пинят версии кода момента запуска; воспроизводить их
нужно на соответствующем коммите таблицы, сохраняя исходные hash checks.
В H4 сохранены исполнявшиеся drivers как provenance; рабочий driver исправлен.

Текущая обоснованная очередь закрыта, а не исчерпан весь класс мыслимых
ML-подходов. Новые training objectives, richer encoders или RL controllers
требуют отдельной постановки и нового независимого holdout. Эта серия
даёт материал для экспериментальной главы и честный отрицательный
результат о tested controllers, без дальнейшего подбора по её final outcomes.
