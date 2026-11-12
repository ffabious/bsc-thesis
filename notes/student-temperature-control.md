# Проверка температуры дистилляции

12 ноября 2026. Статус: выполнен.
[Результаты](../reports/experiments/student-temperature-control.md).
[Протокол](../configs/experiments/student-temperature-control.json).
Основа: [контроль с учителем CE50](../reports/experiments/student-teacher-control.md).

## Вопрос и единственное изменение

Проверить, изменит ли снижение T баланс между верхом ранжирования и Recall@100.
Teacher — прежний frozen/eval CE50 best4, alpha0, lambda1.
**T2 →T1**; формула CE +lambda×T²×KL(teacher||student) прежняя, включая
T² scaling. При T1 multiplier равен1 вместо4. Это один фактор принятого KD loss,
а не отдельная абляция только формы soft targets при одинаковом множителе.

Пилот256, train739571/model_valid29313, history50, dim64, 2blocks/heads,
dropout0.2, batch256, Adam0.001/(0.9,0.98), clip5, seed42, dummy negatives16,
полный каталог114125 и прежний OOT/positive target. Все ID-векторы trainable.
С нуля, без warm start, initial state/CPU/MPS RNG совпадают с T2 и CE50.
Random epoch0 не кандидат; best и patience2 только по model_valid NDCG,
без лимита эпох. Один новый MPS FP32 run, 4 CPU-потока, fallback0.

Исходники trainer, SASRec, teacher и loss побайтово совпадают с T2.
Перепроверены начальные29 тензоров и initial validation ranks. Фиксированный
CE50 rank proof на29313 validation targets переиспользован с проверенными
источниками; teacher probabilities T1 дополнительно проверены на32 MPS inputs
и в CPU/MPS test suite. В training loop initial state/RNG проверяются ещё раз.

## Оценка

Сначала validation-остановка и фиксация best SHA, затем те же CT/PV targets,
queries и каталог. CE50 и T2 не переобучать. Их готовые ranks переиспользовать;
проверить target/user/event_id alignment и связь ranks с выбранными checkpoints.
Для нового checkpoint пересчитать model_valid и CT/PV. Не подбирать другие
T/alpha/lambda по результатам. Test не читать.

Сохранить NDCG/Recall/Hit1, paired whole-user CI к обеим готовым моделям,
last/second-last/earlier-in50/outside50, train popularity и last после top10000,
rank transitions, users/days, training curves, время и memory snapshots.
PV ранее изучен; сравнение исследовательское, один seed не оценивает training
variance. Срезы не меняют правило checkpoint selection.

## Воспроизведение

В чистом output восстановить опубликованный `fixed-teacher.json` по config SHA;
данные и оба готовых контроля должны находиться локально. Setup proofs и
`tests.txt` сохраняются перед экспортом.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_initialization.py --config configs/experiments/student-temperature-control.json --output data/runs/student-temperature-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 1 --config configs/experiments/student-temperature-control.json --output data/runs/student-temperature-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py evaluate --config configs/experiments/student-temperature-control.json --output data/runs/student-temperature-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_student_teachers.py --config configs/experiments/student-temperature-control.json --input data/runs/student-temperature-control --output reports/experiments/student-temperature-control --device mps
```

Checkpoints и per-event arrays остаются в игнорируемом `data/`; config,
небольшие JSON/CSV/графики и отчёт публикуются в Git.

## Результат

4 эпохи, best 2, trainer 16.35 мин; у T2 также 4 эпохи и best 2.
На PV T1: 0.055750 / 0.272036 NDCG@10 / Recall@100 против
0.053899 / 0.265966 у T2 и 0.055093 / 0.252201 у CE50.
К T2 прирост +3.43% NDCG / +2.28% Recall, оба paired CI выше нуля.
К CE50: +1.19% NDCG (CI включает ноль), +7.86% Recall (CI выше нуля).
Повторы последнего трека не восстановились; выигрыш преимущественно
вне окна истории. Оба выбранных checkpoint прошли по 5778 updates.
80 тестов и artifact checks пройдены, test закрыт. Одним seed нельзя
обосновать общее превосходство над CE50; старый cache baseline сохраняется.
