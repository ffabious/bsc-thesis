# H6: условия точного KV reuse

14 ноября 2026. Naive KV reuse **не эквивалентен** текущему SASRec при
append с правым выравниванием или eviction из sliding window. Основная
серия refresh-политик сохраняет полный пересчёт encoder на каждом refresh.

## Проверка

Реальные frozen weights student43, CPU FP32, dropout выключен,2 слоя,
dim64,2 heads, learned absolute positions, окно50. Используются только
синтетические последовательности item IDs,5 фиксированных seeds;
dataset events и test не читаются. Допуск зафиксирован до запуска:
torch.allclose(atol=1e−5,rtol=1e−5).

Прототип действительно сохраняет projected K/V каждого слоя; query берётся
из LayerNorm текущего input, keys/values — из input, residual/FF соответствуют
нашей реализации SASRec. Full reference вычисляет все состояния заданного
окна с теми же tokens, positions, causal mask и weights. Отдельная проверка
подтверждает точное совпадение reference с текущим model.forward.

| Сценарий | Результат5 seeds | Max absolute difference |
| --- | --- | ---: |
| Append, фиксированные позиции, без eviction;8 prefixes/seed | Совпадает в допуске |8.58e−6|
| Append, текущие right-aligned positions; reuse старых K/V | Не совпадает |3.01–5.32|
| Окно50: eviction+сдвиг позиций; reuse оставшихся K/V | Не совпадает |2.40–4.99|
| После invalidation полностью пересобрать текущие K/V | Совпадает в комбинированном допуске |1.14e−5|
| Eviction, позиции сохранены, только первый слой | Совпадает в допуске |5.72e−6|
| Eviction, позиции сохранены, оба слоя | Не совпадает |0.64–10.90|

Все70 assertions прошли:55 ожидаемых совпадений и15 ожидаемых
несовпадений. Weights/source hashes сохранены, encoder state не изменён.
Максимальная абсолютная разница1.14e−5 в rebuild проходит именно
**комбинированный absolute+relative** допуск, а не absolute-only1e−5.

## Почему reference имеет значение

В текущем encoder последние n токенов используют positions50−n…49.
Добавление события сдвигает позиции прежних tokens. В полном окне после
eviction retained tokens также смещаются; старые K/V первого слоя уже
не представляют новое окно.

Фиксация позиций устраняет только эту причину. В2-layer контроле inputs
второго слоя оставшихся tokens уже содержат информацию об удалённом
событии. Полный пересчёт текущего окна эту информацию исключает, поэтому
простого удаления первого KV entry недостаточно. Однослойный positive
контроль изолирует эту причину: его K/V зависят непосредственно от
неизменных token+position inputs.

Точный positive control с append применим к неизменным позициям и полному
префиксу без eviction, пока хватает learned positions. Он меняет serving
контракт текущей модели и не доказывает сохранение её recommendation
quality. Streaming reference с доступом к более старой истории также
нельзя молча считать полным пересчётом последнего окна50.

Под **full recomputation** в основной серии понимается: взять последние
50 строго прошлых events, заново сформировать left padding/right-aligned
positions и causal mask, выполнить все слои, сохранить итоговый vector.
Reuse projection или cache invalidation допустимы только при доказанном
совпадении с этим reference. Для текущего сдвига требуется перестроение
состояний; работа не заявляет точного дешёвого incremental update.

Projected K/V payload при50 tokens:2(K/V)×2layers×50×64×4B =51200B/user
(50KiB), против256B/user vector. Это200× только по payload, без cache
metadata/allocator/history. Скорость не измеряется: корректность дешёвого
update в текущем контракте не прошла проверку, а timing другого reference
не ответил бы на вопрос основного эксперимента.

H6 закрывает технический риск для нашей реализации; это не утверждение
о невозможности KV reuse во всех архитектурах. Другая positional/history
семантика требует отдельной архитектуры, обучения и своего reference.

[Все70 cases и provenance](kv-reuse-feasibility/results.json),
[прототип](../../scripts/diagnostics/check_kv_reuse.py),
[конфиг](../../configs/experiments/kv-reuse-feasibility.json).
