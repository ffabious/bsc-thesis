# SASRec dim 50 / одна голова с прежним обучением

9 ноября 2026. Один локальный запуск на прежнем пилоте 256, история 50.

- От `sasrec-256-joint-bce.json` меняются только `dim: 64 → 50` и `heads: 2 → 1`. Item embedding и FFN по прежним defaults имеют размер dim; два блока, dropout 0.2, batch 256, 16 uniform negatives, общий BCE по 17 logits, Adam 0.001 / betas (0.9, 0.98), clipping 5 и seed 42 сохранены. Все ID-векторы обучаются с нуля.
- Повторно используются те же данные, OOT, каталог и примеры: 739 571 train-target, 29 313 validation-target. Один строго прошлый prefix до 50 событий на каждый target, loss по последнему выходу. Из negatives исключается только текущий positive.
- Локальное MPS FP32, CPU fallback отключён. Существующий trainer не меняется. Время и память берутся из фактического обучения; дополнительный benchmark не нужен для этого сравнения.
- Лимита эпох нет; остановка после двух эпох без строгого улучшения полной validation NDCG@10. Начальное случайное состояние в выборе не участвует. Перед test фиксируются best epoch и checkpoint SHA-256.
- Единственная test-оценка через существующий `compare_sasrec.py`: новый checkpoint на тех же inputs, 45 711 targets и 114 125 кандидатах, с сохранёнными ranks исходной 256 joint BCE. Парные 95% интервалы — 2 000 bootstrap-перевыборок пользователей. Сравнение с прошлым рецептом dim 50 дополнительно использует его сохранённые ranks.
- Один seed, уже использованный исследовательский test и разный compute до одинаковой validation-остановки ограничивают вывод. Этот контроль проверяет сочетание размерностей и числа голов, а не выделяет их эффекты по отдельности. Test не используется для новых настроек.

Результат: 7 эпох, best 5, 6.36 мин; test NDCG@10 0.023453 / Recall@100 0.206690. [Отчёт](../reports/experiments/sasrec-256-d50-h1.md).

## Команды

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-256-d50-h1.json --output data/runs/sasrec-256-d50-h1/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_sasrec.py --device mps --checkpoint data/runs/sasrec-256-d50-h1/train-mps/best.pt --reference-evaluation data/runs/sasrec-256-joint-bce/evaluation-mps.json --allow-different-architecture --output data/runs/sasrec-256-d50-h1/comparison.json
```

При прерывании train-команда с `--resume` восстанавливает optimizer/RNG из `last.pt`. Checkpoints, лог и per-event ranks остаются в локальном `data/`; в Git сохраняются конфиг, протокол и агрегированные результаты.
