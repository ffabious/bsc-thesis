# Полностью покрытые аудиоэмбеддингами части Yambda

5 ноября 2026. Все пять частей основного набора и пилота пересобраны для эксперимента с фиксированными готовыми аудиовекторами. [Эксперимент](../experiments/sasrec-frozen-audio.md), [исходная подготовка](preparation.md).

## Правило отбора

1. Сохраняем исходную train-only выборку пользователей и seed 42: 8 454 пользователя основного набора, те же 256 пользователей пилота.
2. Train-каталог содержит только треки, наблюдавшиеся в train выбранных пользователей и имеющие готовые аудиовекторы в закреплённом источнике.
3. Во всех пяти частях оставляем только события этого train-каталога. Удаляются как треки без аудиовекторов, так и новые для train треки holdout. Никаких средних/случайных замен не делаем.
4. Перенумеровываем item indices и заново считаем `history_size` по оставшемуся потоку. Входы/цели сохраняют строгую хронологию; timestamp ties обрабатываются одной группой. Каждое реальное событие имеет `item_idx >= 2`.
5. Для всего полученного каталога локально материализуем готовую таблицу FP32×128. Издательские `normalized_embed` приводятся к FP32 и единичной L2-норме до обучения. PAD/неиспользуемый UNK — технические нулевые строки, не подстановка для реального события.

Критерий позитивного target прежний: `played_ratio_pct > 50`, хотя бы одно строго прошлое событие. История — последние 50 оставшихся прослушиваний, включая короткие; никакого текущего playback outcome на входе нет. Поскольку поток изменился, старые метрики первого пилота напрямую не сравниваются с новыми. Контрольная ID-модель переобучена на этой же очищенной выборке.

В `users.parquet` поле `selection_train_events` содержит исходное число train-событий, использованное для выбора пользователей, а `train_events` — число после фильтрации. Пользователи не отбираются по holdout-активности.

## Размеры

| Набор | Пользователи | Старый каталог | Новый каталог | Удалено событий | Осталось событий |
| --- | ---: | ---: | ---: | ---: | ---: |
| yambda50m-audio | 8 454 | 782 513 | 730 920 | 2 003 590 | 43 966 640 |
| yambda50m-pilot-audio | 256 | 119 645 | 114 125 | 84 814 | 1 375 435 |

## yambda50m-audio

| Split | Начало, сек. | Конец, сек. | События | Допустимые цели | UNK-цели |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0 | 22 371 205 | 36 557 639 | 23 092 466 | 0 |
| model_valid | 22 371 205 | 22 976 005 | 1 294 252 | 823 653 | 0 |
| controller_train | 22 976 005 | 24 185 605 | 2 488 399 | 1 586 204 | 0 |
| policy_valid | 24 185 605 | 24 790 405 | 1 225 196 | 788 513 | 0 |
| test | 24 790 405 | 26 000 005 | 2 401 154 | 1 566 769 | 0 |

Локальная таблица векторов: `data/processed/yambda50m-audio/audio/vectors.npy`, 730 920 реальных треков, 128 измерений, пропусков **0**. [Manifest данных](../experiments/sasrec-frozen-audio/yambda50m-audio.json), [manifest признаков](../experiments/sasrec-frozen-audio/yambda50m-audio-features.json), [проверки](../experiments/sasrec-frozen-audio/yambda50m-audio-checks.json).

## yambda50m-pilot-audio

| Split | Начало, сек. | Конец, сек. | События | Допустимые цели | UNK-цели |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 0 | 22 371 205 | 1 152 986 | 739 571 | 0 |
| model_valid | 22 371 205 | 22 976 005 | 43 467 | 29 313 | 0 |
| controller_train | 22 976 005 | 24 185 605 | 76 610 | 52 191 | 0 |
| policy_valid | 24 185 605 | 24 790 405 | 35 137 | 23 394 | 0 |
| test | 24 790 405 | 26 000 005 | 67 235 | 45 711 | 0 |

Локальная таблица векторов: `data/processed/yambda50m-pilot-audio/audio/vectors.npy`, 114 125 реальных треков, 128 измерений, пропусков **0**. [Manifest данных](../experiments/sasrec-frozen-audio/yambda50m-pilot-audio.json), [manifest признаков](../experiments/sasrec-frozen-audio/yambda50m-pilot-audio-features.json), [проверки](../experiments/sasrec-frozen-audio/yambda50m-pilot-audio-checks.json).

Временные границы прежние: train примерно 259 дней, затем model_valid 7 дней, controller_train 14, policy_valid 7, test 14. История переносится через границы. Разделение фиксировано до сравнения моделей.

## Источник и проверки

- Источник: `yandex/yambda`, revision `dd6f3a19eef5866e346c3270e098baa641a44948`, `embeddings.parquet`, 13 814 230 943 байт. Полный файл скачан официальным HF/Xet-клиентом и проверен по SHA-256 `c8959a584257473ab9f5dab7e88b05ea3d03b4daf7c16e9a617e8e216d811c83`.
- Подготовка и обучение полностью локальны. Скачанный raw, индексы, матрицы и checkpoints исключены из Git.
- Проверены хэши выходных Parquet и матриц, связь словаря с матрицей, размерность, конечность и единичная норма каждого реального вектора.
- Нарушений порядка времени, пересечений event_id, неверного строгого `history_size`, отсутствия аудио или событий с UNK нет на обоих наборах.
- 16 тестов пройдены, включая удаление отсутствующих/новых треков из всех частей, пересчёт истории, отсутствие fallback, проверку источника кэша и неизменность frozen-таблицы при optimizer updates.

## Команды

```sh
.venv/bin/python -m pip install -r requirements-audio.txt
HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python scripts/prepare_audio_embeddings.py --download --index-only
.venv/bin/python scripts/prepare_yambda.py --audio-items data/raw/yambda-audio/items.parquet --output data/processed/yambda50m-audio
.venv/bin/python scripts/prepare_yambda.py --users 256 --audio-items data/raw/yambda-audio/items.parquet --output data/processed/yambda50m-pilot-audio
.venv/bin/python scripts/prepare_audio_embeddings.py --data data/processed/yambda50m-audio
.venv/bin/python scripts/prepare_audio_embeddings.py --data data/processed/yambda50m-pilot-audio
.venv/bin/python scripts/check_yambda.py data/processed/yambda50m-audio
.venv/bin/python scripts/check_yambda.py data/processed/yambda50m-pilot-audio
```

Существующие outputs не перезаписываются; для повторения задайте новые директории и соответствующий data path в экспериментальном конфиге. Исходные ID-only данные сохранены как исторический baseline. Результаты нового эксперимента относятся к отфильтрованному потоку; удалённые взаимодействия не учитываются в его метриках или возрасте представления по числу событий.

[Официальная карточка источника](https://huggingface.co/datasets/yandex/yambda/blob/dd6f3a19eef5866e346c3270e098baa641a44948/README.md).
