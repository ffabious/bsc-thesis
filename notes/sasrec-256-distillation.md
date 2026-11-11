# CE50 control и дистилляция CE50+CE2 на 256

11 ноября 2026. Всё локально, CPU/MPS FP32, 4 CPU-потока, fallback отключён.
[Протокол](../configs/experiments/sasrec-256-distillation.json).

1. Заморозить существующие CE50 epoch4 / CE2 epoch7. Выбрать alpha учителя
   на model_valid из 0/0.1/…/1 по NDCG@10; tie — меньший alpha. CT/PV/test
   при выборе и обучении не читаются.
2. Продолжить обучение из одинакового CE50 checkpoint для control weight=0
   и двух students weight=0.25/1.0. Все ID-векторы trainable; архитектура,
   train prefixes/targets, batch256, dropout0.2, Adam0.001, seed42, dummy
   negatives16 и patience2 сохранены. Adam новый у каждого запуска. Лимита
   эпох нет. Checkpoint epoch0 участвует в выборе; остановка по model_valid NDCG.
3. Loss = CE(student raw logits, true item) + weight × T² × KL(teacher || student),
   T=2 фиксирована. Учитель — прежняя per-query z-score смесь, умноженная
   на population std raw long logits: ранги смеси сохраняются, softmax logits
   выражены в шкале исходного CE50. Mean shift не влияет на softmax. Это
   определение soft targets, не калибровка вероятностей прослушивания.
   Учитель eval/frozen, получает только прошлую историю; recent — native2
   с learned positions48/49. Dense probabilities не сохраняются на диск.
   KL суммируется по классам и усредняется по batch (`batchmean`), без
   дополнительного деления на 114 125 кандидатов.
4. Выбрать nonzero weight по лучшему model_valid NDCG, tie — меньший weight;
   зафиксировать control/student checkpoints и alpha до чтения CT/PV. Условие
   принятия нового baseline: NDCG выше control, Recall не ниже. Негативный
   результат сохраняется; новые веса по CT/PV не подбираются.
5. Сравнить original CE50, continued control, selected student и учителя на
   одинаковых CT/PV событиях/каталоге, paired user-bootstrap CI. Учитель для
   этого сравнения использует model_valid alpha, а не прежний CT alpha.
6. Измерить training time/memory отдельно по запуску и полный fresh serving
   CPU/MPS B=1/256 в одном процессе с перемешанным порядком вариантов.
   Student/control имеют обычный single-encoder inference без учителя.
   Test не открывается. Проверки, отчёт, commit и push в origin/main.

Команды исходных запусков; для нового повторения нужен новый output root:

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py teacher --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 0 --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 0.25 --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 1 --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py evaluate --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py profile --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device cpu
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py profile --config configs/experiments/sasrec-256-distillation.json --output data/runs/sasrec-256-distillation --device mps
.venv/bin/python scripts/summarize_sasrec_distillation.py --input data/runs/sasrec-256-distillation --output reports/experiments/sasrec-256-distillation
```

76 CPU/MPS тестов прошли перед обучением. На model_valid выбран alpha=0.4:
teacher 0.081946 / 0.312762 NDCG@10 / Recall@100 против исходного CE50
0.072085 / 0.305769. Native2 и scaling teacher logits проверены на всех
29 313 validation targets: primary metrics совпадают, max tail rank delta1.
Checkpoint weights и dense training tensors остаются локально в `data/runs/`;
в Git сохраняются конфиги, код, JSON/CSV, графики и отчёт.

## Результат

Control: 2 эпохи / best0 / 3.72мин; KD0.25: 2 / best0 / 11.09мин;
KD1: 2 / best0 / 11.47мин (полные run times). Веса best checkpoints
побитово равны initializer; initial validation и CPU/MPS RNG совпадают.
Формальный выбор weight0.25 разрешает ничью epoch0 candidates и не означает
успешную дистилляцию. Последняя KD1 model_valid: 0.071607 NDCG / 0.313990
Recall, компромисс −0.7% / +2.7% к initializer.

Locked CT/PV сравнение сохранило метрики исходного CE50; teacher PV
0.063086 / 0.259853. Стоимость single inference прежняя, KD training около3×
CE control. После обучения исправлен relative output export; training math
не менялась, исходник с фактическим SHA сохранён в source-history артефактах.
77 финальных CPU/MPS тестов и независимая проверка результатов пройдены.
[Отчёт](../reports/experiments/sasrec-256-distillation.md).
