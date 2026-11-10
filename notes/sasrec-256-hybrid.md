# Смесь long CE и свежей recent-only модели

10 ноября 2026. План до запуска. Все вычисления локальны; нового обучения нет.
Используются текущий frozen **256 CE, история 50**, и готовый frozen **256 joint
BCE, последние 2 события**. У моделей независимые обученные ID-таблицы.
Recent checkpoint сохраняет ширину 50 и маскирует первые 48 позиций;
его вычислительная дешевизна относительно длинной модели здесь не предполагается.

- Один запрос перед целой timestamp-группой, все прослушивания в истории,
  прежние positive targets и полный каталог 114 125. Читаются CT/PV replay
  и строго предшествующие данные, **test events не открываются**.
- Длинная ветвь использует snapshots прежних 100 refresh-политик. Короткая
  всегда видит последние два **строго прошлых** события текущего запроса,
  независимо от возраста long-cache. Это score ensemble, не совместно
  обученная сеть и не OR-политика из прежних таблиц.
- Для каждой ветви и запроса отдельно: `z=(score−mean)/max(population_std,1e−8)`
  по полному каталогу; затем `s=(1−alpha)*z_long + alpha*z_recent`.
  Нормировка без labels согласует масштабы, не калибрует вероятности.
  Alpha=0/1 оцениваются по исходным logits для точных single-model endpoints.
- Alpha из 0/0.1/…/1 выбирается **для каждой фиксированной long-политики на CT**
  по максимуму NDCG@10, tie — меньший alpha. Записывается selection.json до
  scoring смеси на PV. На PV каждый вес неизменен; это policy-specific константы,
  не обученный per-request controller. Повторный validation-анализ пилота
  исследовательский, не blind replication. Checkpoints не подбираются заново.
- Сравнения: recent-only, long-only fresh, long-only с каждой cache-политикой,
  hybrid fresh и hybrid с каждой cache-политикой. NDCG@10/Recall@100, парные
  2 000 user-bootstrap CI, общие repeat/other-in-window/outside-window сегменты.
  CT-метрики смеси — на выборке подбора alpha; основной результат — PV.
- Переиспользуются все прошлые long-ranks и schedules с проверкой SHA;
  новый scoring dedup по `(fresh request, long snapshot, target)`. Полные
  catalog scores существуют только для batch. Сохраняются ranks и агрегаты.
- Проверки: свежая recent ветвь совпадает с исходным evaluator, alpha endpoints
  и независимые item-таблицы корректны, CPU/MPS смеси/маски, frozen weights,
  неизменные data/reference hashes, selection фиксирован до PV.
- Здесь сравнивается **качество**, без переноса long-only cost на гибрид:
  добавляются второй encoder, item-table scoring и нормировка. Экономия полной
  стоимости такой смесью не заявляется; controller не обучается.

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_hybrid_sasrec.py --device mps --config configs/experiments/sasrec-256-hybrid.json --output data/runs/sasrec-256-hybrid
```

`--resume` переиспользует сохранённые mixture grids после проверки ключей,
входов/alpha и совпадения CT selection; подходит для продолжения проверки без
повторного scoring. В этом run так завершена CPU-проверка после исправления
приведения UInt32 target IDs к Long. Исходные alpha не менялись.

Результат PV: recent-only **0.039578 / 0.216979**, fresh long CE
**0.055093 / 0.252201**, fresh hybrid **0.058224 / 0.273190** при alpha=0.3
(+5.7% NDCG / +8.3% Recall к long). При TTL 24h alpha=0.7:
**0.042870 / 0.239591**, выше stale long и recent-only по обеим метрикам,
но ниже fresh long. 62 теста, свежий recent oracle и CPU mixture check пройдены;
test закрыт, стоимость гибрида не измерялась.
[Отчёт, все 100 политик, CI и графики](../reports/experiments/sasrec-256-hybrid.md).
