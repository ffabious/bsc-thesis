# Финальная проверка: user-disjoint holdout

14 ноября 2026. Этот протокол фиксируется отдельным коммитом **до** чтения
event rows нового holdout. Пользователь разрешил автономное продвижение
по основному плану, включая локальную итоговую оценку. Никаких внешних
вычислений, нового обучения или подбора по этому holdout нет.

## Почему требуется отдельная когорта

Ранний [baseline search](../reports/experiments/sasrec-baseline-search.md)
публикует test метрики256 и использует их для итогового сравнения рецептов.
[CE1k](../reports/experiments/sasrec-1k-full-ce.md) уже оценён на175831 test
targets951 development users. Checkpoints выбирались по model_valid,
но этот test нельзя называть полностью неоткрывавшимся holdout всей
исследовательской цепочки. H1–H6 test events не читали; это более узкое
утверждение, которое не отменяет ранние оценки.

Аудит всех завершённых training.json обнаружил только три dataset roots:
ID-only256, audio-closed256 и audio-closed1k. Их объединение —1000 uid.
Из общего подготовленного closed-audio набора выбираются только другие
пользователи, до чтения их будущих событий/исходов. Полный
[machine-readable audit](../reports/experiments/final-user-holdout/protocol.json).

## Зафиксированный выбор

- Source pool: yambda50m-audio, условие train_events≥20.
- Исключаются все uid трёх development datasets, а не только encoder users.
- Среди7443 оставшихся — первые256 по sha256(`20261114:uid`), ascending digest.
  Нет фильтра по будущей активности, test targets, качеству или известности
  будущих треков. Выбор независимо воспроизведён DuckDB SHA256-sort.
- Encoder — готовый CE1k, checkpoint SHA
  `dcb3b56d459928e44b4e18264f76a4f707c2203595ff91e7d130133fe9c994af`.
- Catalog и mapping ровно CE1k:244194 known items. Новые history items→UNK;
  unknown targets исключаются с публикацией coverage. Треки не добавляются
  из новой train/history части и embeddings не дообучаются.
- Controller — готовый CE1k immediate-MLP42. Пороги q75=0.018529653549194336
  и q25=0.00671705394418387, выбранные на прежнем CT. Прямые comparators
  events10 и OR2events+300sec. CT expected-mixture weights также заморожены.
- Полная простая grid106 фиксируется в protocol.json; её test frontier
  описательный. Новые budget choices по test не выбираются.
- Оценивается test24790405…26000005,14 дней. Все прежние closed-audio
  listens нового пользователя доступны только как наблюдённая история;
  текущая timestamp-группа исключается целиком. Cache reset на старте test;
  mandatory initial refresh, все запросы и неположительные события входят
  в replay/стоимость. Model/controller weights никогда не обновляются.

## Анализ до открытия исходов

Primary: две paired NDCG@10 разницы controller−зафиксированный simple.
Whole-user bootstrap2000, seed42;97.5% intervals для каждой из двух
primary comparisons, Bonferroni family2. Secondary Recall@100/Hit@1,
paired fresh losses, CT expected mixtures и descriptive frontier;95%
pointwise интервалы там, где указаны. Общая denominator — положительные
known targets, scoring all244194 items, repeats разрешены.

Стоимость проверяется отдельно для vector+full-catalog scoring и static
vector+top100-cache. Полные direct paired B=1 workloads на identical samples,
128×3 requests; фактические refresh fractions и drift показываются явно.
CT budget не гарантирует равного observed cost в другой когорте.
Основная оценка качества использует exact top100/sentinel вне100, не MRR.

Это проверка переносимости на unseen users с доступной историей и frozen
vocabulary, а не обучение на всех8454 и не строго та же population, что
первые1к. User groups могут иметь общие items и временные факторы;
user bootstrap не устраняет эти зависимости. После открытия исходов
этой серии параметры/пороги/когорта не меняются. Вывод может быть
отрицательным; новый поиск потребовал бы нового независимого holdout.

[Конфиг](../configs/experiments/final-user-holdout.json),
[зафиксированный протокол](../reports/experiments/final-user-holdout/protocol.json),
[проверка выбора](../reports/experiments/final-user-holdout/cohort-verification.json).
