# Fresh/stale replay на 256 CE

10 ноября 2026. Локальный диагностический replay с фиксированным seed-42 checkpoint `decfa67…` из выбранного 256 full-CE baseline. Веса не обучаются. История 50, полный каталог 114 125; только `controller_train` и `policy_valid` и предшествующая им история. Файл событий test не читается.

- Event lags: 0, 1, 5, 10, 25, 50 новых прослушиваний. Состояние строится по prefix, из которого исключены как минимум N последних наблюдавшихся событий. Если граница попадает внутрь timestamp-группы, она округляется назад до начала группы. Публикуются фактические event/time ages.
- Time lags: 0, 5 минут, 30 минут, 2 часа, сутки, 7 суток. Для target в t encoder видит только события с timestamp строго меньше t−lag. Если новых взаимодействий за этот интервал нет, stale и fresh получают одинаковый контекст.
- Один фиксированный paired cohort для всех лагов и обеих осей: события, для которых каждый snapshot имеет непустую прошлую историю. Все исключения и coverage публикуются; полный fresh score до фильтрации показывается отдельно. Target labels не выбирают snapshot.
- Frozen encoder пересчитывает каждый уникальный `(user, cutoff)` с нуля; KV reuse не применяется. Уникальные пары `(snapshot, target)` оцениваются один раз. Это оптимизация offline replay, а не измерение deployment refresh frequency.
- Основные метрики NDCG@10 / Recall@100, парные stale−fresh разницы с 2 000 user-bootstrap перевыборок. Сегменты определены по fresh-окну и неизменны при лагах. Активность — число строго прошлых прослушиваний за сутки, интервалы 0–9 / 10–49 / 50+. Групповые метрики описательные.
- Fresh должен воспроизвести существующие `examples/ranks`; дополнительно проверяются CPU oracle и неизменность весов/исходных файлов. Тесты покрывают timestamp ties, cross-split history, отсутствие будущих/test событий, fixed cohort и эквивалентность naive recomputation.
- Отдельный cost benchmark: pre-tokenized histories на CPU, модель/каталог на MPS, локальный CPU-cache user vectors, top-100 с возвратом индексов на CPU. Batch 1 и 256, warm-up и три повтора, синхронизация каждого batch. Encoding, scoring, lookup/transfer, refresh/write и end-to-end измеряются отдельно. Компоненты не обязаны складываться из-за накладных расходов.
- Амортизация `hit_cost + refresh_cost / R` иллюстрирует раздельные background refresh и serving, где все запросы читают cache. R — число предсказаний между обновлениями, а не фиксированный лаг; качество–cost frontier конкретных политик требует отдельного replay этих политик. Внешний cache/network не моделируется.

Результат: paired cohort 52 172 CT / 23 394 PV targets. На PV NDCG при минимальном event lag 1 ниже fresh на 21.5%, при time lag 5 минут — на 17.5%, сутки — на 51.5%. Fresh oracle воспроизведён, 47 тестов пройдены. [Отчёт и графики](../reports/experiments/sasrec-256-replay.md).

## Команды

```sh
.venv/bin/python -m pip install -r requirements-replay.txt
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_sasrec.py --device mps --config configs/experiments/sasrec-256-replay.json --output data/runs/sasrec-256-replay
.venv/bin/python scripts/plot_sasrec_replay.py --input data/runs/sasrec-256-replay/results.json --output reports/experiments/sasrec-256-replay
```

Для повторного replay выбирается новая output-директория. Per-event snapshots/ranks остаются в локальном `data/`, в Git сохраняются код, конфиг, агрегаты и PNG/SVG/CSV. Этот этап измеряет чувствительность к возрасту представления и компоненты стоимости; он не выбирает refresh-политику или контроллер по test.
