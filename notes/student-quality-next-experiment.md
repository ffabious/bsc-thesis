# Проверка источника ошибок student

Статус: выполнен. [Результаты](../reports/experiments/student-teacher-control.md).
Основа:
[разбор качества](../reports/experiments/sasrec-256-distillation-audit.md).
[Зафиксированный протокол](../configs/experiments/student-teacher-control.json).

## Вопрос

Связана ли потеря повторов/менее популярных треков с добавлением CE2 к учителю,
либо она сохраняется при soft-target KD только от CE50? На проблемном PV-срезе
сам teacher хуже CE50, а student ещё хуже. На готовом student часть потери
снимается коротким входом. Ни один из этих фактов отдельно не устанавливает
причину в training. Нужен контроль источника soft targets при том же KD loss.

## Единственное изменение

Teacher — **готовый CE50 best4**, alpha короткой ветви **0.4 →0**.
T **2**, lambda **1** прежние. Loss: CE +T²×KL(CE50||student), batchmean
по batch примеров, сумма по114125 классам. Teacher frozen/eval, raw CE50 logits:
при alpha0 их возвращает текущая реализация. Это фиксированный контроль,
без повторного выбора alpha. Recent teacher не участвует в forward.

Пилот256, тот же train739571/model_valid29313, каталог, OOT, positive target
и строгий timestamp-prefix. History50, dim64, 2blocks/heads, dropout0.2,
batch256, Adam0.001/(0.9,0.98), clip5, seed42, dummy negatives16.
Случайная инициализация совпадает с восстановленной CE50; проверить все
state/CPU/MPS RNG hashes до update. Все ID-векторы trainable. Локальный
MPS FP32, 4 CPU-потока, fallback0. Без лимита эпох, patience2 по model_valid
NDCG; random epoch0 не кандидат. Готовые CE50 и student со смесью не переобучать.

## Что зафиксировать до CT/PV

Фиксированный источник teacher targets и checkpoint по model_valid NDCG.
Teacher alpha0 на model_valid обязан воспроизвести метрики готового CE50.
Сохранить init/data/checkpoint/source hashes. Кроме общих NDCG/Recall/Hit1:

- last item; second-last excluding last; earlier in50 excluding last2; outside50;
- train popularity top1000 /1001–10000 /после10000, их пересечения с last item;
- top1/top10/top100 переходы и paired whole-user CI, user/day stability.

## Как интерпретировать

Если teacher-only student восстанавливает repeat/tail NDCG относительно
student со смесью, это поддержит связь потери с добавлением CE2/смешиванием,
включая возможную сложность переноса смеси в один encoder. Это ещё не
отделит качество teacher targets от ограничений их переноса.

Если потеря остаётся и с CE50 teacher, смесь не является необходимым условием:
следует отдельно проверять сам KD objective/его strength или temperature,
не запускать новые варианты автоматически. Один опыт с общим seed не
оценивает training variance. Все срезы — диагностика, не checkpoint selection.
Для замены общего baseline отдельно нужны NDCG выше CE50, Recall не ниже
и перенос на более поздние события.

PV уже участвовал в диагностике и не является новым независимым подтверждением;
test оставить для финальной зафиксированной постановки. Никакого alpha/T/lambda
grid, нового CE-контроля или изменений данных/архитектуры. Serving student
прежний; training использует одну teacher-ветвь вместо двух, фактическое
время фиксируется по завершении.

## Воспроизведение

Источник alpha0 — сохранённый endpoint прежней model_valid curve; он
фиксируется до training, без нового поиска. Initializer восстановлен тем же
скриптом; все29 начальных тензоров и CPU/MPS RNG совпали с прошлым student.
Teacher logits на всех29313 validation targets точно совпали с raw CE50.
Проверки не читают CT/PV/test. В training loop state/RNG проверяются ещё раз.

```sh
mkdir -p data/runs/student-teacher-control
cp reports/experiments/student-teacher-control/fixed-teacher.json data/runs/student-teacher-control/fixed-teacher.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_initialization.py --config configs/experiments/student-teacher-control.json --output data/runs/student-teacher-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 1 --config configs/experiments/student-teacher-control.json --output data/runs/student-teacher-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py evaluate --config configs/experiments/student-teacher-control.json --output data/runs/student-teacher-control --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_student_teachers.py --config configs/experiments/student-teacher-control.json --input data/runs/student-teacher-control --output reports/experiments/student-teacher-control --device mps
```

`fixed-teacher.json` и его SHA закреплены в config; teacher selection и
initialization proof находятся в локальном output. Данные и checkpoints —
в игнорируемом `data/`; отчёт, небольшие JSON/CSV и графики — в Git.
Команды рассчитаны на чистый output: existing training/evaluation не перезаписываются.
Перед экспортом в output сохраняются `tests.txt`, `teacher-check.json` и
`matched-setup.json`; опубликованные копии содержат зафиксированные проверки.

## Результат

4 эпохи, best2, trainer16.10мин. Model_valid0.072193 /0.321939.
PV0.053899 /0.265966 против0.055093 /0.252201 у CE50 и
0.053961 /0.264512 у прошлого student. К CE50 NDCG−2.17% (CI включает0),
Recall+5.46% (CI выше0); к прошлому student обе общие CI включают0.
PV repeat NDCG по точечной оценке вырос0.4924 →0.5185, но остаётся ниже
CE50 0.5695; на последнем треке после train top10000 остаётся значимая
условная потеря. Смесь не является необходимым условием этой потери в этом seed.
80 тестов, matched setup/init/data/checkpoint/source проверки пройдены;
test закрыт. Следующие варианты автоматически не запускаются.
