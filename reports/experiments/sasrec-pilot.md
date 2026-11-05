# SASRec: локальный пилот на 256 пользователях

Запуск 5 ноября 2026. Обучение, CPU/MPS benchmarks, оценка и хранение выполнены на локальной машине. [План](../../notes/experiment-plan.md), [конфигурация](../../configs/experiments/sasrec-pilot.json), [исходная подготовка данных](../data/preparation.md).

## Результат

- Обучены 5 эпох на MPS; лучший checkpoint — **эпоха 4**, выбранная по NDCG@10 на полном `model_valid` до обращения к test.
- Полный запуск обучения, включая подготовку примеров, validation и сохранение: **355.16 с**. CPU использован для training-step benchmark и оценки того же checkpoint; отдельное полное CPU-обучение не проводилось.
- Test: **NDCG@10 0.019312; Recall@100 0.172522**. Метрики совпали на CPU и MPS.
- MPS быстрее CPU в training-step benchmark в **2.74×**, в одном полном проходе оценки — в **1.39×** при указанных ниже настройках.

## Модель и данные

- Apple M3 Pro, 12 CPU, 36 GiB объединённой памяти; CPU benchmark ограничен 4 потоками. macOS 26.6.2, Python 3.14.8, PyTorch 2.15.0, NumPy 2.5.4. [Полное окружение](sasrec-pilot/environment.txt).
- История 50 событий, learned absolute positional embeddings, embedding 64, 2 causal attention-блока, 2 головы, FFN 64→64→64 с ReLU, dropout 0.2, tied input/output item embeddings. Всего 7 711 168 параметров. FP32, без autocast и compile.
- Adam, learning rate 0.001, betas (0.9, 0.98), gradient clipping 5.0; batch 256, seed 42; максимум 5 эпох, patience 2 по validation NDCG@10.
- 761 770 train-примеров и 29 975 validation-примеров. Каждый пример предсказывает положительный listen по последним 50 строго прошлым прослушиваниям. Все timestamp ties имеют общий prefix; короткие прослушивания остаются в истории. PAD — 0, UNK — 1, padding слева.
- Loss: `mean(softplus(-positive_score)) + mean(softplus(negative_score))`. На цель выбираются 16 равномерных отрицательных кандидатов из train-каталога, исключается только текущий positive. Сэмплирование не использует будущие взаимодействия. Это адаптация SASRec под наш next-positive-listen протокол, не точная репликация оригинального benchmark/sequence-wise sampler.
- Test содержит 46 794 допустимых положительных целей у 244 пользователей. Для каждой цели считается точный ранг среди всех **119 645** train-items. PAD/UNK исключены из кандидатов, ранее прослушанные треки разрешены, ties scores разрешаются меньшим item index.
- Из 58 259 положительных test-событий 11 465 (19.68%) относятся к неизвестным train-каталогу трекам. Первичные метрики условны на известный каталог; неизвестные события сохраняются как UNK для последующей истории.
- Веса фиксированы после выбора checkpoint. Контексты test содержат наблюдённые ранее события всех предшествующих периодов, но не текущее событие и не его playback outcome. Обновление модели по holdout не выполнялось.

## Обучение и выбор checkpoint

| Эпоха | Train loss | Train, с | Train примеров/с | Validation NDCG@10 | Validation Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| До обучения | — | — | — | 0.020482 | 0.021284 |
| 1 | 0.722724 | 68.13 | 11181 | 0.011930 | 0.135847 |
| 2 | 0.410908 | 66.87 | 11391 | 0.020743 | 0.185021 |
| 3 | 0.300349 | 66.93 | 11381 | 0.026978 | 0.218315 |
| **4** | 0.240143 | 67.49 | 11287 | 0.031253 | 0.229691 |
| 5 | 0.203416 | 67.35 | 11310 | 0.030928 | 0.237965 |

Пятая эпоха увеличивает Recall@100, но её NDCG@10 ниже четвёртой: критерий выбора не менялся. Начальная модель с tied embeddings не является random-ranking baseline. MostPop на validation: NDCG@10 0.003159, Recall@100 0.046539.

## Качество на test

| Модель | NDCG@10 | 95% CI | Recall@100 | 95% CI |
| --- | ---: | --- | ---: | --- |
| SASRec, checkpoint 4 | 0.019312 | [0.015050; 0.024957] | 0.172522 | [0.148746; 0.199945] |
| MostPop | 0.002375 | [0.001716; 0.003175] | 0.037633 | [0.030087; 0.046858] |

MostPop использует только число положительных train-прослушиваний и тот же каталог. Интервалы рассчитаны 2 000 bootstrap-перевыборками пользователей с сохранением их событий; point estimates усреднены по событиям. Один seed обучения не оценивает вариативность между обучениями.

## Скорость CPU и MPS

Training benchmark запущен в отдельных процессах последовательно: 10 warm-up обновлений, затем 3 повтора по 50 обновлений на одном resident batch. Включены forward, loss, backward, clipping и Adam step; исключены подготовка данных, генерация negatives и перенос batch. Время MPS измерено с `torch.mps.synchronize()` на границах каждого повтора. CPU fallback запрещён (`PYTORCH_ENABLE_MPS_FALLBACK=0`).

| Устройство | Training step: медиана, мс | 3 измерения, мс | Training примеров/с | Test оценка, с | Test примеров/с |
| --- | ---: | --- | ---: | ---: | ---: |
| CPU, 4 потока | 55.965 | 56.169; 55.965; 54.889 | 4574 | 6.656 | 7030 |
| MPS | 20.405 | 20.405; 20.498; 20.083 | 12546 | 4.776 | 9797 |

Реальное обучение MPS включает batching, генерацию negatives и перенос batch: примерно 67.36 с на эпоху. Validation и initialization учитываются отдельно; полная длительность приведена выше.

Оценка — один проход после warm-up с batch 256: перенос prefixes, fresh encoding, полный catalog scoring, точный ранг и возврат рангов на CPU. Чтение Parquet, построение prefixes, загрузка checkpoint и bootstrap исключены из этого таймера. Это throughput батчевой офлайн-оценки, не latency одиночного production request и не стоимость обслуживания refresh-кэша.

## Память

| Измерение | CPU, MiB | MPS, MiB |
| --- | ---: | ---: |
| Peak process RSS в training benchmark | 1123.70 | 1026.62 |
| Peak process RSS полного train + validation | — | 1044.50 |
| Peak process RSS test evaluation | 870.36 | 662.16 |
| MPS tensor allocation после forward в отдельной memory-пробе | — | 192.23 |
| Metal driver allocation после training benchmark | — | 1044.78 |
| Metal driver allocation после test evaluation | — | 1042.70 |

RSS — process high-water mark `resource.getrusage`, включая Python/PyTorch, подготовку примеров и данные. MPS tensor/driver значения — snapshots, а не точный пик всех GPU-временных выделений. Driver allocation включает allocator cache и Metal/MPS ресурсы. На объединённой памяти эти числа не складываются с RSS. Memory-проба выполнена отдельно от timed training trials.

## Проверки и воспроизводимость

- 10 тестов пройдены, включая forward/backward CPU–MPS, causal mask, конечность padding gradients, точные ранги и ties, negative sampling и отсутствие событий с текущим timestamp в prefix. [Вывод тестов](sasrec-pilot/tests.txt).
- Проверка исходного пилота повторно сверила SHA-256 Parquet и строгий `history_size`; нарушений не найдено.
- 147 из 46794 рангов различаются между CPU и MPS максимум на 1. Полностью совпадают 99.6859% рангов; NDCG@10 и Recall@100 одинаковы. [Проверка рангов](sasrec-pilot/parity.json). Численная близость не выдаётся за побитовую идентичность.
- SHA-256 checkpoint одинаков в обеих оценках: `ead576c9209c0ab783eafc5e5edf2afbf6743694a5d491bc719a9126e28130e5`. Оценщик проверяет checksum и соответствие data manifest завершённому training run.
- Checkpoint: `data/runs/sasrec-pilot/train-mps/best.pt` (локальный, около 30.85 MB); покомпонентные test ranks — `data/runs/sasrec-pilot/evaluation-{cpu,mps}.npz`. Эти файлы исключены из Git.
- [Training trace](sasrec-pilot/training.json), [CPU benchmark](sasrec-pilot/benchmark-cpu.json), [MPS benchmark](sasrec-pilot/benchmark-mps.json), [CPU evaluation](sasrec-pilot/evaluation-cpu.json), [MPS evaluation](sasrec-pilot/evaluation-mps.json), [краткий JSON](sasrec-pilot/summary.json).
- [Хэши публикуемых исходников](sasrec-pilot/code-sha256.json). [Команды запуска](../../README.md#sasrec-на-пилоте).

## Ограничения и следующий этап

Это одна небольшая SASRec-конфигурация и один seed на train-only отобранном пилоте. Оценка относится к позитивным событиям известных треков с общей историей для timestamp ties. Числа не являются репликацией benchmark статьи и не доказывают преимущества refresh-политики. CPU/MPS ranking сравнивает одни веса, а не два независимо обученных рекомендателя. Производительность зависит от batch, длины, каталога и числа CPU-потоков.

Следующий этап: обучить long-history + recent-history архитектуру, проверить пользу длинной ветви и её обновления; для сравнения refresh-политик фиксировать checkpoint. Настройки дальнейших экспериментов выбирать по train/validation, не по опубликованным pilot test-метрикам.

## Первичные источники реализации и измерений

- [SASRec: статья](https://arxiv.org/abs/1808.09781), [авторский код](https://github.com/kang205/SASRec).
- [PyTorch MPS backend](https://docs.pytorch.org/docs/2.15/notes/mps.html).
- [MPS tensor allocation](https://docs.pytorch.org/docs/2.15/generated/torch.mps.current_allocated_memory.html), [driver allocation](https://docs.pytorch.org/docs/2.15/generated/torch.mps.driver_allocated_memory.html).
