# Fresh/stale сравнение пар энкодеров

12 ноября 2026. Готовые CE42/student42 T1 и CE43/student43 T1, пилот256.
Без обучения: frozen best checkpoints из парного повтора. Всё локально,
MPS FP32, fallback0, 4 CPU-потока, batch256.

## Фиксированный протокол

- Переиспользуются исходные `data/runs/sasrec-256-replay/replay.npz` и CE42
  ranks с проверкой SHA. Входы и сетка не пересобираются.
- Event lags: 0,1,5,10,25,50. Time lags: 0,5мин,30мин,2ч,1сутки,7суток.
  Event lag исключает как минимум N событий; timestamp-группы не разрезаются.
  Time snapshot видит только события строго до target_time−lag.
- Общий cohort: CT52172, PV23394. Те же19 CT-событий исключены из всех условий,
  потому что не имеют непустой истории для каждого лага. У PV исключений нет.
- History50, каталог114125 без seen-item exclusion. User vector пересчитывается
  из snapshot prefix полностью; KV reuse и новые политики refresh не добавляются.
- Для трёх новых моделей encoding/scoring выполняется один раз для каждого
  уникального snapshot и snapshot/target. CE42 не пересчитывается.
- Fresh сверяется с сохранёнными оценками четырёх моделей по event_id,
  user, target, split и history. Полный свежий CT до фильтрации остаётся
  доступен в предыдущем отчёте; его нельзя подменять общим cohort52172.

## Что сравниваем

1. NDCG@10, Recall@100 и Hit1 каждой модели при каждом лаге; fresh−stale
   потеря, доля сохранённого качества и прежние repeat/activity срезы.
2. Student−CE при одинаковом лаге, отдельно для каждого seed.
3. Снижение потери от устаревания:
   `(CE_fresh−CE_stale)−(student_fresh−student_stale)`.
   Положительный знак означает меньшую абсолютную потерю у student.
   Это также изменение преимущества student над CE относительно fresh.

Парные 95% CI: 2000 перевыборок целых пользователей, event-weighted,
bootstrap seed42 для обеих пар; model seeds42/43 не меняются. Сравнения
условны на fixed checkpoints; два training seeds не дают надёжной оценки
training variance. CI pointwise, срезы исследовательские. Относительная
потеря и удержание качества показываются описательно, без отдельного CI.

Проверяются frozen state, hashes checkpoint/data/source, causal cutoffs,
fresh alignment, CPU oracle на256 одинаковых смешанных lag/target парах
для каждой новой модели. Test не читается. Это fixed-lag ablation;
новый serving benchmark и quality–cost frontier не нужны для этой задачи.

## Команды

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m unittest tests.test_replay tests.test_staleness_comparison -v
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_encoder_pairs.py --device mps
```

Конфигурация: `configs/experiments/paired-staleness-replay.json`.
Per-event ranks сохраняются в `data/runs/paired-staleness-replay/`,
компактные JSON/CSV/PNG/SVG и отчёт — в Git.

## Результат

Replay завершён за143.61с. Recall student выше собственного CE при всех лагах
в обеих парах и периодах (pointwise CI выше0), но снижение абсолютной
потери от устаревания не воспроизводится систематически. PV сутки: NDCG
0.026732/0.026832/0.024478/0.026852 для CE42/student42/CE43/student43;
относительная потеря примерно51–55% у всех четырёх моделей.10тестов
и независимая проверка352CI прошли, test закрыт.
[Отчёт и кривые](../reports/experiments/paired-staleness-replay.md).
