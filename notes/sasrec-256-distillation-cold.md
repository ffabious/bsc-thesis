# Student с нуля: CE50+CE2 distillation, 256

11 ноября 2026. Один новый локальный MPS FP32 запуск, 4 CPU-потока,
fallback отключён. [Протокол](../configs/experiments/sasrec-256-distillation-cold.json).

1. Восстановить случайную CE50-инициализацию по source commit591575a,
   config и seed42. Сравнить все тензоры с текущим student constructor,
   CPU RNG с сохранённым CE last.pt и initial validation. Готовый CE50
   не переобучать. Initial checkpoint старого запуска отдельно не сохранялся;
   это reconstruction с дополнительной проверкой по сохранённым артефактам.
2. Переиспользовать зафиксированного teacher CE50+CE2, alpha0.4 из model_valid.
   Учить student со случайных весов: T2, lambda1, loss CE+T²KL batchmean;
   остальные настройки исходного CE50, включая lr0.001, batch256,
   dropout0.2, history50, dim64, 2 heads/blocks и patience2. Лимита эпох нет.
   Warm-start отсутствует; случайный epoch0 не участвует в выборе, как у
   исходного CE50. Все ID-векторы student trainable, teacher frozen/eval.
3. В реальном training loop до первого update проверить state/RNG SHA
   восстановленного CE initializer. Выбрать checkpoint только по model_valid
   NDCG, затем зафиксировать его до CT/PV. Alpha/T/lambda не подбирать заново.
4. Сравнить с готовым CE50 и учителем на прежних CT/PV событиях/кандидатах,
   paired user-bootstrap CI; исходные контрольные ranks переиспользовать.
   Измерить full serving CPU/MPS B1/256 вперемешку в одном процессе;
   training time/memory сравнить с сохранённым CE training log отдельно.
   Test не читать. Сохранить код/JSON/CSV/графики/отчёт, commit и push main.

## Воспроизведение

Из корня repo, с активным локальным MPS и `PYTORCH_ENABLE_MPS_FALLBACK=0`:

```sh
.venv/bin/python scripts/check_sasrec_initialization.py --config configs/experiments/sasrec-256-distillation-cold.json --output data/runs/sasrec-256-distillation-cold --device mps
.venv/bin/python scripts/distill_sasrec.py train --weight 1 --config configs/experiments/sasrec-256-distillation-cold.json --output data/runs/sasrec-256-distillation-cold --device mps
.venv/bin/python scripts/distill_sasrec.py evaluate --config configs/experiments/sasrec-256-distillation-cold.json --output data/runs/sasrec-256-distillation-cold --device mps
.venv/bin/python scripts/distill_sasrec.py profile --config configs/experiments/sasrec-256-distillation-cold.json --output data/runs/sasrec-256-distillation-cold --device cpu
.venv/bin/python scripts/distill_sasrec.py profile --config configs/experiments/sasrec-256-distillation-cold.json --output data/runs/sasrec-256-distillation-cold --device mps
.venv/bin/python scripts/summarize_sasrec_distillation_cold.py --config configs/experiments/sasrec-256-distillation-cold.json --input data/runs/sasrec-256-distillation-cold --output reports/experiments/sasrec-256-distillation-cold
```

`initialization.json`, training/evaluation и serving files не перезаписываются:
для нового опыта нужен новый output. Для training resume после прерывания
есть `--resume`; текущий запуск выполняется непрерывно. Логи и tests.txt
копируются в output перед экспортом. Данные, ranks и checkpoints остаются
в игнорируемом `data/`; небольшие JSON/CSV/графики и отчёт публикуются в Git.

## Результат

Все 29 начальных тензоров и 29 313 initial validation ranks совпали с
восстановленной CE50. Guard в фактическом training loop подтвердил state и
CPU/MPS RNG. 8 эпох, best 6, validation patience 2; trainer 46.94 мин против
исторических 12.63 мин CE50. Model_valid: 0.073060 / 0.315935 против
0.072085 / 0.305769. На CT выросли обе метрики; на PV — 0.053961 / 0.264512
против 0.055093 / 0.252201: NDCG −2.05% (CI включает 0), Recall +4.88%
(CI выше 0). Inference student и CE50 практически одинаков: CPU B1
1.3345 / 1.3436 мс, MPS 1.8668 / 1.8696 мс. Teacher дороже и сохраняет
лучший PV NDCG. Выигрыш NDCG не перенесён, исходный CE50 остаётся
референсом для прежнего replay/frontier. 79 тестов, artifact checks,
test не открыт. [Полный отчёт](../reports/experiments/sasrec-256-distillation-cold.md).
