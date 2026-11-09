# Полный CE на 1 000 пользователей

9 ноября 2026. Один локальный запуск seed 42, перенос выбранного 256-рецепта на прежний 1k-пилот.

- Конфиг отличается от `sasrec-256-full-ce.json` только путём данных. История 50, item/hidden 64, два блока/головы, FFN 64, dropout 0.2, batch 256, Adam 0.001 / betas (0.9, 0.98), clipping 5. Все ID-векторы обучаются с нуля; аудиопризнаки не подаются. Полный CE включает все 244 194 известных трека; PAD/UNK исключены. `negatives: 16` сохраняет RNG stream, но sampled negatives не участвуют в full CE.
- Повторно используются неизменные `yambda50m-1k-audio`, OOT, target и строгие prefixes: 2 799 860 train-target, 100 809 validation-target. Одна последняя позиция на каждый target. Сначала короткий отдельный benchmark, его веса отбрасываются; затем MPS FP32 без CPU fallback до двух эпох без строгого улучшения полной validation NDCG@10, без лимита эпох. Начальное случайное состояние не участвует в выборе.
- Best epoch и checkpoint hash фиксируются до test. Native A/B с прежним 1k joint BCE: одни 175 831 test-target и 244 194 кандидата, естественные 1k-истории. Отдельное сравнение трёх моделей на ровно прежних 45 711 событиях и histories 256, каталог 114 125. Для 1k выполняется проверенное сопоставление item ID; старые ranks 1k BCE и основного 256 CE seed 42 переиспользуются. Эти две оценки не смешиваются.
- Метрики NDCG@10, Recall@100, Hit@1; парные 95% интервалы по 2 000 перевыборкам пользователей, отдельно last-item / other-in-window / outside-window. Проверяются hashes данных, исходных артефактов, inputs/targets/users/candidates и максимум validation. Код модели/trainer не меняется.
- Один seed 1k не оценивает training variance; больший набор меняет пользователей, каталог и размер item-таблицы. Сравнение с 256 использует её точные истории, поэтому не отражает дополнительный естественный контекст 1k. Результаты не настраиваются по уже исследовавшемуся test. Все вычисления и checkpoints локальны.

Результат: 4 эпохи, best 2, 52.66 мин. Native test: NDCG@10 0.034495 (+65.0% к прежнему 1k BCE), Recall@100 0.187930 (−3.3%). На общей когорте 0.033055 / 0.198880, ниже 256 CE 0.045111 / 0.229420. [Отчёт и проверки](../reports/experiments/sasrec-1k-full-ce.md).

## Команды

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --config configs/experiments/sasrec-1k-full-ce.json --output data/runs/sasrec-1k-full-ce/benchmark-mps.json
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-1k-full-ce.json --output data/runs/sasrec-1k-full-ce/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_sasrec.py --device mps --checkpoint data/runs/sasrec-1k-full-ce/train-mps/best.pt --reference-evaluation data/runs/sasrec-1k-joint-bce/evaluation-mps.json --output data/runs/sasrec-1k-full-ce/comparison-1k.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_sasrec.py --device mps --checkpoint data/runs/sasrec-1k-full-ce/train-mps/best.pt --reference-evaluation data/runs/sasrec-256-full-ce/evaluation-mps.json --output data/runs/sasrec-1k-full-ce/comparison-256.json
```

Train сохраняет best и resumable last; при прерывании та же команда с `--resume`. В Git сохраняются конфиг, протокол и агрегированные результаты, веса/логи/ranks остаются в `data/`.
