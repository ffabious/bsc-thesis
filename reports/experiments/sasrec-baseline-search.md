# Обычный SASRec encoder: поиск сильного baseline

9 ноября 2026. Все обучение и scoring локальны: Apple M3 Pro, MPS FP32,
CPU fallback отключён. [План и решения](../../notes/sasrec-baseline-search.md).

**Выбран полный cross-entropy при прежней архитектуре dim 64 / heads 2.**
На test среднее по двум seed: NDCG@10 **0.045531 вместо 0.026218 (+73.7%)**,
Recall@100 **0.230732 вместо 0.217202 (+6.2%)**. Улучшение обеих метрик
повторилось на seed 42 и 43; парные user-bootstrap интервалы выше нуля.
Это один encoder с полным пересчётом истории при каждом предсказании.

## Что зафиксировано

Те же 256 пользователей, данные `yambda50m-pilot-audio`, OOT и каталог
**114 125** треков: **739 571 train-target**, **29 313 validation-target**,
**45 711 test-target у 244 пользователей**. История содержит все предшествующие
прослушивания, target требует `played_ratio_pct > 50`. Текущая timestamp-группа
целиком исключается из prefix. Один пример на каждый target, loss по последнему
выходу окна. Holdout-события могут входить в историю последующих предсказаний;
веса по ним не обновляются. Аудиопризнаки не подаются.

Исходная архитектура: история 50, item/hidden dim 64, два блока, две головы,
FFN 64 → 64 → 64, dropout 0.2, tied обучаемая ID-таблица. Всего **7 357 888**
параметров, вне item-таблицы **53 760**. Batch 256, Adam 0.001 / betas (0.9, 0.98),
clipping 5. Все запуски с нуля, без лимита эпох, до двух эпох без строгого
улучшения полной validation NDCG@10. Начальная случайная модель в выборе
checkpoint не участвует.

## Последовательные проверки гипотез

Отбор рецепта: validation NDCG@10 выше исходной, Recall@100 не ниже 98% исходного.
Guard зафиксирован до следующих результатов. Промежуточные dropout/CE16 модели
не оценивались на test; для dim 50 выполнено ранее прямо запрошенное сравнение.

| Рецепт, seed 42 | Validation NDCG@10 | Recall@100 | Best / эпохи | Train, мин |
| --- | ---: | ---: | ---: | ---: |
| Исходный dim 64 / heads 2, BCE16 | 0.043825 | 0.290213 | 6 / 8 | 8.25 |
| Dim 50 / head 1, остальное как у исходного | 0.038983 | 0.271825 | 5 / 7 | 6.36 |
| Исходный encoder, dropout 0 | 0.048805 | 0.283901 | 5 / 7 | 7.15 |
| Исходный encoder, sampled CE16 | 0.040369 | 0.275373 | 5 / 7 | 8.07 |
| Исходный encoder, полный CE | **0.072085** | **0.305769** | 4 / 6 | 12.63 |

- Уменьшение dim вместе с числом голов ухудшило метрики. Отдельные эффекты dim
  и heads этим контролем не разделяются. [Полный результат dim 50](sasrec-256-d50-h1.md).
- Dropout 0 поднял NDCG на 11.4%, но снизил Recall на 2.17% и не прошёл guard.
  Около 90% общего NDCG-прироста приходится на last-item повторы; это компромисс.
- Замена BCE на sampled CE при прежних 16 negatives ухудшила обе метрики.
- Расширение CE до всего каталога дало наибольший выигрыш и прошло guard.
  Сравнение sampled/full CE проверяет охват каталога в рамках CE; сравнение
  full CE с BCE16 оценивает новый рецепт целиком.

Точные метрики, user-macro, сегменты и парные интервалы:
[validation summary](sasrec-baseline-search/validation-summary.json).
Обоснование objective-контроля: [SCE, §2.2, формулы 3–4](https://arxiv.org/html/2409.18721v2#S2.SS2).

## BCE и полный CE в этом эксперименте

BCE16 независимо классифицирует один positive и 16 uniform negatives через
sigmoid; loss усредняется по 17 logits. Полный CE нормирует scores через softmax
по **всем известным трекам** и повышает вероятность target относительно всего
каталога. Любой конкурирующий item с высоким score участвует в loss. PAD/UNK
исключены. Это меняет обучение; inference по-прежнему ранжирует обычные dot products,
без дополнительного softmax или второй модели.

В конфиге full CE сохранён `negatives: 16`: массив по-прежнему генерируется для
одинакового потока RNG и порядка batch, но loss его не использует. Эффективный
набор — все 114 125 классов. Benchmark-поля отличаются только длиной короткого
замера и не влияют на отдельный train-процесс. [Реализация](../../scripts/sasrec.py).

## Повтор с seed 43 и выбор до test

Повторены исходный BCE и полный CE с изменением только seed. Начальные
validation-метрики внутри каждого seed совпали. Выигрыш NDCG и Recall guard
выполнены на обоих seed; нового подбора гиперпараметров между повторами нет.

| Рецепт | Seed | Validation NDCG@10 | Recall@100 | Best / эпохи | Train, мин |
| --- | ---: | ---: | ---: | ---: | ---: |
| BCE16 | 43 | 0.042654 | 0.297786 | 9 / 11 | 11.53 |
| Полный CE | 43 | **0.071945** | **0.305223** | 4 / 6 | 12.86 |

[Парный validation-повтор](sasrec-baseline-search/replication-validation-summary.json),
[условия повтора](sasrec-baseline-search/replication-plan.json).
Основной seed 42, рецепт и hashes обоих checkpoint
[зафиксированы до test](sasrec-baseline-search/selected-baseline.json).
Seed по test не выбирался; обе модели оцениваются и публикуются.

## Итоговый test

Одинаковые события, users, histories и полный каталог для всех сравнений.
Сохранённый test исходного seed 42 переиспользован; остальные три checkpoint
оценены один раз после фиксации рецепта.

| Рецепт | Seed | NDCG@10 | Recall@100 | Hit@1 |
| --- | ---: | ---: | ---: | ---: |
| BCE16 | 42 | 0.026805 | 0.215659 | 0.009385 |
| Полный CE | 42 | **0.045111** | **0.229420** | **0.018704** |
| BCE16 | 43 | 0.025632 | 0.218744 | 0.008663 |
| Полный CE | 43 | **0.045950** | **0.232045** | **0.019098** |
| BCE16, среднее двух seed | — | 0.026218 | 0.217202 | 0.009024 |
| Полный CE, среднее двух seed | — | **0.045531** | **0.230732** | **0.018901** |

Среднее — арифметическое среднее метрик двух отдельных моделей, без ансамбля.
Парные интервалы: 2 000 перевыборок целых пользователей, event-weighted, seed 42.

| Full CE − BCE того же seed | Разница NDCG@10, 95% CI | Разница Recall@100, 95% CI |
| --- | --- | --- |
| Seed 42 | +0.018306 [0.011918; 0.026973] | +0.013760 [0.007431; 0.021502] |
| Seed 43 | +0.020318 [0.013479; 0.029476] | +0.013301 [0.005860; 0.021898] |

[Test seed 42](sasrec-baseline-search/test-summary-seed42.json),
[test seed 43](sasrec-baseline-search/test-summary-seed43.json).
User-macro метрики тоже выросли. На основном seed улучшение есть во всех сегментах:

| Target относительно истории | События | NDCG BCE → CE | Recall BCE → CE |
| --- | ---: | --- | --- |
| Последний трек | 723 | 0.411060 → 0.521094 | 0.786999 → 0.796680 |
| Другой трек из окна | 7 268 | 0.052130 → 0.090388 | 0.311915 → 0.365300 |
| Отсутствует в окне | 37 720 | 0.014560 → 0.027264 | 0.186161 → 0.192365 |

Для контекста переиспользованы ranks прежней смеси 50/2 с зафиксированным alpha:
NDCG 0.038987 / Recall 0.230579. Один full-CE encoder seed 42 даёт более высокий
NDCG при близком Recall. Парный CI разницы NDCG [0.001245; 0.012070], Recall
[−0.007814; 0.006006]. Смесь не перенастраивалась. [Сводка](sasrec-baseline-search/summary.json).

## Стоимость, проверки и воспроизведение

Шесть новых train-запусков суммарно заняли **58.60 мин** с загрузкой примеров
и validation. Старый seed-42 контроль повторно не обучался. Среднее время двух
полных CE — **12.74 мин**, двух BCE — **9.89 мин**; бюджеты не уравнивались,
сохранено одинаковое правило остановки. Архитектура и serving-операции одинаковы.

Перед полным CE выполнен короткий отдельный benchmark: 2 warm-up + 3 × 5 шагов,
его веса отброшены. Около 42 мс/шаг, оценка train-эпохи 121 с; фактическое обучение
подтвердило доступность на MPS. Peak RSS первого полного CE 1 033.72 MiB,
MPS tensors после benchmark-forward 298.66 MiB, driver около 1 051 MiB.
Показатели unified memory не складываются как независимая физическая память.
[Benchmark](sasrec-baseline-search/full-ce-benchmark.json).

Прошли **41 тест**, включая формулы/градиенты BCE и CE, исключение PAD/UNK,
CPU/MPS, resume и validation-only evaluation. На 256 одинаковых test-событиях
каждого полного CE CPU/MPS дали одинаковые NDCG/Recall: 253 и 255 из 256 ranks совпали
точно, остальные различаются максимум на одну позицию. Проверены данные,
checkpoint hashes, validation maxima, stop reasons, неизменность исходных
артефактов и выбор до test. [Проверки](sasrec-baseline-search/verification.json),
[тесты](sasrec-baseline-search/tests-full-ce.txt).

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-256-full-ce.json --output data/runs/sasrec-256-full-ce/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --split model_valid --config configs/experiments/sasrec-256-full-ce.json --checkpoint data/runs/sasrec-256-full-ce/train-mps/best.pt --output data/runs/sasrec-256-full-ce/validation.json
.venv/bin/python scripts/summarize_sasrec_baseline.py --reference data/runs/sasrec-baseline-search/reference-validation.json --evaluations data/runs/sasrec-256-full-ce/validation.json --output data/runs/sasrec-baseline-search/full-ce-validation-summary.json
```

Для повторного запуска выбирается новая output-директория; для продолжения
прерванного train используется `--resume`. Test-команда аналогична evaluate
без `--split model_valid`. Конфиги, training/evaluation JSON и hashes опубликованы;
веса, логи и per-event ranks остаются в локальном `data/runs/`.

Результат относится к этому 256-пилоту и закрытому train-каталогу. Два seed дают
проверку повторяемости, но не надёжную оценку распределения результатов обучения;
user-bootstrap интервалы относятся к фиксированным моделям. Test уже использовался
в предыдущих исследованиях, поэтому результат исследовательский. После итогового
test настройки не менялись. Выбранный checkpoint служит fresh-encoder reference
для дальнейшего исследования поддержания user representation.
