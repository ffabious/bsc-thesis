# Контроль авторских гиперпараметров SASRec

9 ноября 2026. Один локальный запуск на 256 пользователях, история 50; сравнение с успешным `sasrec-256-joint-bce`, история 50. Короткая ветвь, смешивание и 1k в этот эксперимент не входят.

## Зафиксированный setup

- Повторно используется неизменный `yambda50m-pilot-audio`: 739 571 train-target, 29 313 validation-target, 45 711 test-target, 114 125 кандидатов. Прежние OOT и target `played_ratio_pct > 50`; все предшествующие прослушивания могут входить в историю. Текущая timestamp-группа целиком исключена из prefix.
- Авторские гиперпараметры: trainable item embedding = hidden = 50, 2 блока, 1 голова, FFN 50 → 50 → 50, ReLU, dropout 0.5, batch 128, Adam LR 0.001 / betas (0.9, 0.98), один равномерный negative. Общая входная/выходная item-таблица; размерности совпадают, дополнительные input/output adapters — Identity. L2/weight decay отсутствуют.
- Loss: `mean(softplus(-positive_score) + softplus(negative_score))`. При одном negative существующий режим `separate_mean` точно даёт эту формулу. В прежнем контроле общий BCE усреднялся по одному positive и 16 negatives; абсолютные train loss разных рецептов напрямую не сравниваются.
- Сохраняется наша PyTorch-реализация и формат обучения: один строго прошлый prefix на каждый положительный target, supervision только последней позиции, negative sampler исключает только текущий target. Авторские равномерное сэмплирование пользователей/последних окон и обучение сразу на всех позициях здесь не переносятся. Стандартный PyTorch MultiheadAttention также содержит выходную проекцию attention; число параметров слегка отличается от исходного TensorFlow-кода.
- Оставлены seed 42, gradient clipping 5 и прежняя остановка по полной validation NDCG@10: две эпохи без строгого улучшения, без максимального числа эпох. Начальный случайный checkpoint не участвует в выборе. Авторские 201 эпоха из кода не копируются из-за другого определения эпохи и sampler.
- M3 Pro, MPS, FP32, CPU fallback отключён. Перед обучением — короткий benchmark на train в отдельном процессе, его веса отбрасываются. В основной train сохраняются best и resumable last checkpoints.
- Test оценивается после validation-остановки, с фиксированным best checkpoint и полным прежним каталогом. Сравнение с сохранёнными ranks исходной длинной 256 на одинаковых событиях и пользователях: NDCG@10, Recall@100, Hit@1, парные 95% интервалы по 2 000 перевыборкам пользователей; отдельно повторы последнего трека и остальные цели.
- Один запуск проверяет весь набор гиперпараметров. Причину возможного изменения качества по отдельному параметру он не выделяет. Test уже использовался в предыдущих исследованиях; новых настроек после просмотра test не выполняется.

Источники гиперпараметров: [авторский main.py](https://github.com/kang205/SASRec/blob/master/main.py), [model.py и loss](https://github.com/kang205/SASRec/blob/master/model.py), [sampler](https://github.com/kang205/SASRec/blob/master/sampler.py).

Результат: обучение остановилось после 7 эпох, best 5; test NDCG@10 0.007815 и Recall@100 0.087616, ниже исходного контроля. [Отчёт и проверки](../reports/experiments/sasrec-256-author-params.md).

## Воспроизведение

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --config configs/experiments/sasrec-256-author-params.json --output data/runs/sasrec-256-author-params/benchmark-mps.json
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-256-author-params.json --output data/runs/sasrec-256-author-params/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-256-author-params.json --checkpoint data/runs/sasrec-256-author-params/train-mps/best.pt --output data/runs/sasrec-256-author-params/evaluation-mps.json
```

При прерывании та же train-команда с `--resume` восстанавливает optimizer, RNG, порядок и позицию примеров. Веса и per-event ranks остаются локально в игнорируемом `data/`; в Git сохраняются код/config, агрегированные результаты и hashes.
