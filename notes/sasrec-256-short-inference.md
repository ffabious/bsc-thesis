# Короткий inference recent2 и обновлённый frontier

11 ноября 2026. Локальный CPU/MPS, прежние checkpoints и выбранные на CT alpha.

1. Передавать recent только последние два token IDs, сохраняя learned positions
   48/49, padding, causal mask и все веса. Полный вход длины 50 сохраняет прежний
   расчёт. Обучение и подбор alpha не повторяются.
2. Проверить эквивалентность в eval FP32: все CT/PV представления, scores на
   фиксированной выборке запросов по полному каталогу, recent ranks и выбранные
   mixture ranks на всех targets. CE и BCE проверяются отдельно; CPU/MPS тоже.
   До запуска допуски vectors/scores: atol 2e-5, rtol 1e-4. Rank agreement и
   разницы NDCG/Recall/Hit@1 публикуются отдельно, точное FP32 совпадение не
   предполагается. Для frontier использовать фактическое качество короткого пути.
3. Повторить полный serving benchmark B=1/256 и memory probes прежним скриптом,
   переключив только ширину recent. CPU/MPS CE и MPS BCE, те же окна/повторы;
   warmup, sync, direct-fresh long, cache init/refresh/hit, полный каталог,
   z-score/mix и final top-100 остаются включёнными. Старые замеры — исторический
   контроль, новый frontier использует новую общую long-калибровку.
4. Построить quality–cost frontier и выбрать бюджетные настройки только на CT;
   PV — отдельная оценка. Сохранить raw JSON, CSV, PNG/SVG, проверки и команды.
   Test не открывать. После проверок commit и push в origin/main.

Ресурсные конфиги:
[CE](../configs/experiments/sasrec-256-hybrid-ce-short.json),
[BCE](../configs/experiments/sasrec-256-hybrid-bce-short.json).
`recent_input_length=2` влияет только на serving; configs и weights checkpoint
не переписываются. Width 50 остаётся исходным режимом по умолчанию.

Команды исходных запусков (для повторения выбрать новые output paths и
согласовать `hybrid_run` resource configs):

```sh
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m unittest discover -s tests -v

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_short.py --hybrid-run data/runs/sasrec-256-hybrid-ce --device cpu --output data/runs/sasrec-256-short-check-cpu
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_short.py --hybrid-run data/runs/sasrec-256-hybrid-ce --device mps --output data/runs/sasrec-256-hybrid-ce-short --full-replay
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_short.py --hybrid-run data/runs/sasrec-256-hybrid --device mps --output data/runs/sasrec-256-hybrid-bce-short --full-replay

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-ce-short.json --device cpu --output data/runs/sasrec-256-short-resources/cpu.json
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-ce-short.json --device mps --output data/runs/sasrec-256-short-resources/mps.json
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-bce-short.json --device mps --output data/runs/sasrec-256-short-resources/bce-mps.json

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/benchmark_sasrec_short.py --config configs/experiments/sasrec-256-hybrid-ce-short.json --device cpu --output data/runs/sasrec-256-short-resources/paired-cpu.json
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/benchmark_sasrec_short.py --config configs/experiments/sasrec-256-hybrid-ce-short.json --device mps --output data/runs/sasrec-256-short-resources/paired-mps.json

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/calibrate_sasrec_hybrids.py --ce-profile data/runs/sasrec-256-short-resources/mps.json --bce-profile data/runs/sasrec-256-short-resources/bce-mps.json --output data/runs/sasrec-256-short-resources/interleaved

.venv/bin/python scripts/plot_sasrec_frontier.py --input data/runs/sasrec-256-short-resources/cpu.json data/runs/sasrec-256-short-resources/interleaved/ce.json --bce-reference data/runs/sasrec-256-short-resources/interleaved/bce.json --output reports/experiments/sasrec-256-short-inference
```

Корректировка по результатам замеров: одинаковый long-контроль отдельных MPS
профилей дал 2.40 и 1.91 мс. Для общего frontier эти timings напрямую не
объединяются: повторяется только serving-калибровка CE/BCE с чередованием
путей в одном процессе, одним long-контролем и новыми mixed-workload checks.
Отдельные profiles сохраняются как диагностика и источник memory probes.

## Результат

Native2 сохраняет все три метрики обеих смесей на всех 100 политиках CT/PV.
На CPU все 92 043 vectors и проверенные scores совпадают побитово; на MPS
CE max abs vector error 1.14e-5, BCE 3.34e-6, допуски соблюдены.
Точные tail ranks могут отличаться; в mixtures максимум 3 / 2 позиции,
без изменений NDCG@10, Recall@100 или Hit@1. 71 тест и serving oracles пройдены.

Парный encoder speedup при B=256: CPU 11.9× / MPS 2.8×; полный hybrid
1.07× / 1.03×. При B=1 recent-only быстрее на CPU на 7.4%, медленнее на MPS
на 8.3%; hybrid почти не ускоряется. Новый frontier включает 201 CPU /
302 MPS варианта на split. Mixed-workload отклонения cost model достигают
17.3%, поэтому близкие cost-точки не считаются надёжно различимыми.
Test не открыт. [Отчёт и артефакты](../reports/experiments/sasrec-256-short-inference.md).
