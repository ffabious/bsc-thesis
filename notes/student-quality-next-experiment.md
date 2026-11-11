# Проверка источника ошибок student

Статус: предложение, training не запускался. Основа:
[разбор качества](../reports/experiments/sasrec-256-distillation-audit.md).

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
