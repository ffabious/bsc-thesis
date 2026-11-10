# Реальные refresh-политики на 256 CE

10 ноября 2026. Протокол зафиксирован до запуска. Frozen seed-42 full-CE
checkpoint `decfa67cb…`, история 50, полный каталог 114 125. Только локальные
CPU/MPS; веса не обучаются, test не открывается.

- Один запрос перед каждой timestamp-группой **всех** сохранённых прослушиваний,
  включая короткие. Все события группы добавляются только после прогноза.
  Метрики — по прежним положительным targets; стоимость — по всем запросам.
  Это явно заданный offline proxy потока запросов, не production traffic.
- История проходит через границы split. Кэш каждой политики независимо
  сбрасывается в начале CT/PV и лениво инициализируется на первом запросе
  пользователя полным строго прошлым контекстом. Инициализация — оплаченный
  refresh. У пользователей без запросов кэш не создаётся.
- `always`: refresh перед каждым запросом; `never`: только инициализация.
  Events: refresh при >= N новых прослушиваниях после прошлого refresh,
  N = 2/5/10/25/50. Time: on-request TTL >= 5m/30m/2h/1d.
  Hybrid: **OR** обоих условий, полная сетка 5×4. Нет фоновых refresh в простое.
  Счётчик включает все наблюдённые события, не только положительные labels.
  Пороги сбрасываются после refresh; вся текущая timestamp-группа недоступна.
- Encoder полностью пересчитывает последнее окно; cache hit переиспользует
  готовый user vector. KV reuse не применяется. Offline dedup экономит только
  исследовательское время; все логические refresh учитываются в deployment cost.
- Одинаковые targets и кандидаты для 31 политики, без фильтрации по lag.
  Парные policy−always CI: 2 000 bootstrap целых пользователей, seed 42.
  Метрики event-weighted, а стоимость/request и refresh rate group-weighted.
- Точные числа refresh/hit/initialization дополняются локальным B=1 microbench:
  **refresh→CPU cache write→scoring/top100→CPU result** и
  **CPU cache lookup→scoring/top100→CPU result**. Refresh-ветвь использует новый
  vector напрямую; отдельного повторного чтения и двойного refresh-charge нет.
  Замеряются полные ветви, а не сумма изолированных компонентов.
- На каждом split: общие случайные запросы для обеих ветвей, warmup 8,
  3 серии × 128 запросов, синхронизация каждого запроса. Основная стоимость —
  **оценка** `refresh_rate × measured_refresh_ms + hit_rate × measured_hit_ms
  + measured_policy_decision_ms`; includes initialization. Это локальное
  последовательное обслуживание, не p50/p99 и не измерение полного сервиса.
  Истории токенизированы; сеть и storage истории не моделируются.
- Frontier строится отдельно для NDCG и Recall на обоих split; это point
  estimates, не утверждение статистического превосходства близких точек.
  CT выбирает максимум NDCG при бюджетах 60/75/90% стоимости always
  (tie: Recall, затем меньшая стоимость). Имена фиксируются до оценки PV.
  PV frontier всех заранее заданных политик описательный; test остаётся закрыт.
- Проверка модели стоимости: для CT-selected и always/never измеряются
  полные смешанные ветви на случайных **реальных** запросах со snapshot/refresh
  решениями этих политик на CT/PV. Подготовка уже существующего cache state
  за пределами таймера; refresh/write внутри. Выводим отличие от оценки,
  не подменяя полный replay коротким benchmark.

## Команды

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_sasrec_policies.py --device mps --config configs/experiments/sasrec-256-policies.json --output data/runs/sasrec-256-policies
.venv/bin/python scripts/plot_sasrec_policies.py --input data/runs/sasrec-256-policies/results.json --output reports/experiments/sasrec-256-policies
```

Per-request schedule и per-target ranks сохраняются локально в `data/`;
агрегаты, графики PNG/SVG, CSV и код публикуются в Git. Frozen/fresh oracle,
CPU/MPS, strict-past/timestamp groups, nonpositive events, split reset,
отсутствие test/future dependencies и точный cost accounting проверяются.

Результат: 31 политика, 62 129 CT / 29 914 PV запросов, все 75 585 targets.
Выбранные на CT при лимитах 60/75/90% always-cost: hybrid 25/2h, 5/30m, 2/5m.
На PV 2/5m: −18.4% расчётной полной стоимости, −8.6% NDCG, −0.9% Recall.
Fresh/oracle и 55 тестов пройдены. Максимальное отличие mixed-cost медианы
от оценки 2.48%. [Результаты, frontier и ограничения](../reports/experiments/sasrec-256-policies.md).
