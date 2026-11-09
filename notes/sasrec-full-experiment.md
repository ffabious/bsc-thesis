# Большой SASRec на полном наборе

Согласовано 9 ноября 2026. Все вычисления локальны: Apple M3 Pro, 36 GiB, MPS FP32 без CPU fallback. Цель — получить сильный ориентир качества для последующих экспериментов с устареванием представлений. Это не доказательство абсолютного потолка качества.

**Полное обучение отложено по указанию пользователя.** После замеров был начат полный запуск короткой модели, затем он остановлен во время начальной validation: завершённых эпох и checkpoints нет. Следующий согласованный объём — увеличенные модели на 256 и 1 000 пользователях. Ниже сохранён протокол полного набора для воспроизводимости уже выполненных замеров.

Позднее пользователь также отменил 1k. Итоговое сравнение выполнено только для 256; [результаты](../reports/experiments/sasrec-large.md). Ни 1k, ни полный набор автоматически не продолжаются.

## Зафиксированный протокол

- Все 8 454 пользователя с минимум 20 train-прослушиваниями. Убираем ограничение наличия аудиовекторов, поскольку обучаются ID-эмбеддинги. Закрытый каталог только по train: 782 513 треков; неизвестные holdout-треки исключаются из истории и целей. Train: 23 805 630 примеров.
- Прежние границы OOT и next-positive-listen target (`played_ratio_pct > 50`), история строго раньше timestamp целевого события, включая переходы между периодами. Likes/dislikes не добавляются. Одновременные события исключаются всей группой.
- Две независимые модели: окна 50 и 2. Item embedding 64, hidden 256, 4 блока, 8 голов, FFN 1024, dropout 0.2. Короткая модель использует фактический тензор длины 2, в отличие от пилотной маски в окне 50.
- Общий BCE, 16 равномерных негативов, Adam 0.001 / betas (0.9, 0.98), clipping 5, seed 42. По замерам выбран batch 1024 для обеих моделей: длинная ускорилась примерно на 7%, короткая — втрое против 256, память помещается в локальные ресурсы. Нет лимита эпох; остановка после двух эпох без строгого улучшения полной model_valid NDCG@10. Начальная случайная модель не участвует в выборе checkpoint.
- Перед обучением замеряются обе модели на полном каталоге и train-данных. Память процесса отделяется от памяти MPS; короткий benchmark оценивает скорость, но не заменяет фактические времена эпох.
- Атомарные best/last checkpoints. Last содержит optimizer, порядок примеров, позицию внутри эпохи и CPU/MPS/NumPy RNG. Сохранение каждые 5 000 шагов и после эпохи; продолжение через `--resume` с неизменной конфигурацией и manifest.
- После обучения — per-query z-score по одинаковым кандидатам, линейная смесь `(1-alpha)*z50 + alpha*z2`. Сетка alpha 0, 0.1, ..., 1; выбор максимума model_valid NDCG@10, при равенстве меньший alpha. Нормировка согласует шкалы и не калибрует вероятности. Selection фиксируется до любых новых test-оценок.
- Test: отдельные модели и смесь на полном каталоге; дополнительно исходные события, истории и каталоги пилотов 256 и 1k с mapping по item_id. Для сравнения — парные пользовательские интервалы и разрез повторов последнего трека. Native-метрики разных каталогов напрямую не сравниваются. Эффекты объёма данных и ёмкости этим экспериментом не разделяются.

## Воспроизведение

```sh
.venv/bin/python scripts/prepare_yambda.py --closed-catalog --output data/processed/yambda50m-closed
.venv/bin/python scripts/check_yambda.py data/processed/yambda50m-closed
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --config configs/experiments/sasrec-full-long.json --output data/runs/sasrec-full-long/benchmark-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --config configs/experiments/sasrec-full-recent.json --output data/runs/sasrec-full-recent/benchmark-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-full-long.json --output data/runs/sasrec-full-long/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-full-recent.json --output data/runs/sasrec-full-recent/train-mps
```

Для продолжения прерванного обучения к той же train-команде добавляется `--resume`. Raw, обработанные события, веса и per-event ranks остаются в игнорируемом `data/`; в Git сохраняются код, конфигурации, агрегированные результаты и hashes.

Полный запуск автоматически не продолжается. Для активных пилотов используется отдельная команда `caffeinate -i .venv/bin/python scripts/run_sasrec_pairs.py`. `caffeinate -i` предотвращает автоматический сон во время работы, не мешая отключению экрана.
