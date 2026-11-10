# Более редкие refresh-политики на 256 CE

Расширение по запросу пользователя от 10 ноября 2026. До нового replay:

- Time TTL: **5m, 30m, 1h, 2h, 3h, 6h, 12h, 24h**. 5m и 2h сохранены для
  прямого сравнения с предыдущей сеткой; остальные запрошенные интервалы включены.
- Event thresholds: **2, 5, 10, 20, 25, 50, 60, 120, 240, 480**. Новая основная
  сетка 10/20/60/120/240/480 — условные 30m/1h/3h/6h/12h/24h при 3 минутах на трек,
  **не** оценка длительности треков в датасете и не эквивалентность политик.
  Счётчик учитывает все сохранённые прослушивания, включая короткие и повторы.
- Always/never и все OR-сочетания: **100 политик**, включая все прежние 31.
  Та же frozen 256 CE модель, CT/PV, все requests/targets и полный каталог.
  On-request TTL, strict-past timestamp groups, reset между CT/PV, lazy init
  и прочие правила [исходного протокола](sasrec-256-policies.md) сохранены.
- Переиспользуются проверенные по SHA предыдущие `(snapshot,target)→rank` пары.
  Перед reuse все stream arrays и прежние schedules должны совпасть точно;
  checkpoint/data hashes также совпадают. Новые пары оцениваются на MPS локально.
- Стоимость ветвей и decision cost прежних 31 политик сохраняются из исходного
  замера, чтобы сравнение сеток не менялось из-за timing noise. Decision cost
  новых политик измеряется локально, mixed-workload checks выполняются заново.
  Все initialization/refresh/hit считаются по полному реальному расписанию.
- CT по прежнему выбирает максимум NDCG при лимитах 60/75/90% always-cost,
  до новых PV-метрик. Это расширение после предыдущего validation-анализа,
  не независимая blind replication. **Test не открывается**.
- Дополнительные таблицы: refresh **count и % всех запросов**, cache hit count/%
  на CT/PV для каждого порога; отдельно из refresh выделяется initialization.
  Графики качества по порогам и новый quality–cost frontier, все гибриды в CSV.

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_sasrec_policies.py --device mps --config configs/experiments/sasrec-256-policies-wide.json --output data/runs/sasrec-256-policies-wide
.venv/bin/python scripts/plot_sasrec_policies.py --input data/runs/sasrec-256-policies-wide/results.json --output reports/experiments/sasrec-256-policies-wide
```

Стоимость B=1 — не вычислительный batch replay (там 256), а последовательный
serving benchmark: один запрос за раз. Полные ветви refresh/write/serve и
cache-hit/serve измерены отдельно; их времена взвешиваются по точным числам
выполнений, плюс CPU decisions. Это оценка локальной общей cache-схемы,
без сети, чтения хранилища истории или сравнения с direct-fresh без cache write.

Результаты: все 100 политик рассчитаны, 31 прежняя точка сохранена точно.
PV refresh rate: 30m 15.03%, 1h 10.31%, 3h 6.43%, 6h 4.94%, 12h 3.58%,
24h 2.79%; 10/20/60/120/240/480 событий — 11.62/6.08/2.31/1.40/0.96/0.77%.
При 480 на PV только 15 повторных обновлений сверх 216 инициализаций.
56 тестов, fresh/CPU oracle и отдельный CPU check новых пар пройдены.
Более длинная cost-перепроверка: расхождение медиан до 1.69% CT / 7.58% PV;
стоимость сравнения оставлена на общей исходной калибровке.
[Таблицы, quality/frequency curves, frontier и ограничения](../reports/experiments/sasrec-256-policies-wide.md).
