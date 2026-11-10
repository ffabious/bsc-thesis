# Recent full CE, новая смесь и измеренный frontier

10 ноября 2026. Автономный локальный эксперимент по запросу пользователя.
Test не открывается; весь compute — Apple M3 Pro, CPU/MPS, FP32, fallback отключён.

1. **Обучение recent CE2.** Копия прежнего recent BCE2 config с единственным
   изменением `objective=full_ce`. История по-прежнему 2 видимых события в tensor
   длины 50, dim 64, 2 blocks/heads, dropout 0.2, batch 256, Adam 0.001,
   seed 42, те же train/validation prefixes, patience 2 по model_valid NDCG.
   Лимита эпох нет. 16 dummy negatives сохраняют прежний RNG stream, в loss
   full CE не входят. Checkpoint — максимум model_valid, не CT/PV.
2. **Смесь CE50+CE2.** Длинная модель `decfa67cb…` остаётся прежней. Полный
   каталог 114 125 и все CT/PV запросы/targets прежние. Scores нормируются
   per-query z-score и смешиваются линейно. Для каждой из 100 long-политик
   alpha из 0/0.1/…/1 выбирается по CT NDCG, tie — меньший alpha, фиксируется
   до PV mixtures. Сравнить recent CE с BCE и новую смесь со старой CE50+BCE2
   на тех же событиях с парными user-bootstrap CI. Основные метрики —
   NDCG@10/Recall@100; Hit@1 показывается как вторичная метрика.
3. **Ресурсы после оценки качества.** Измерить encoder/scoring и полные local
   serving paths: fresh long, recent-only, cached long и hybrid refresh/hit.
   Полный hybrid включает свежую recent, long encode при refresh, CPU cache
   write/lookup, обе item-таблицы, обе нормировки, смешивание, один final top-100
   и CPU result. Alpha endpoints пропускают неиспользуемую ветвь. Direct-fresh
   без бесполезной записи cache — отдельный контроль. Истории токенизированы;
   сеть/storage/конкурентный online traffic не моделируются.
4. **Quality–cost frontier CT/PV.** Для всех политик учесть точные init/refresh/hit
   counts, CPU decisions и measured serving paths; старую long-only стоимость
   на смесь не переносить. Сравнить long-only, recent-only и hybrid на одной
   машине/калибровке, отдельно NDCG и Recall. Point-estimate frontier,
   неопределённость повторных cost-серий и paired quality CI публикуются.
   Графики показывают стоимость полного serving и encoder refresh counts.
   Выбор политик при бюджетах — только CT; PV frontier описательный.

CPU/MPS oracle, неизменность данных и frozen checkpoints после обучения,
selection до PV, hashes и meaningful tests обязательны. Отдельно сохраняются
время и память обучения, параметры моделей и ресурсные измерения inference.
Короткая ветвь пока не оптимизируется в native tensor length 2: сначала
измеряется существующая архитектура, чтобы не смешивать эффект loss и inference.

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-256-recent-2-full-ce.json --output data/runs/sasrec-256-recent-2-full-ce/train-mps
```

## Выполнено

9 эпох / best 7, 18.70 мин. Единственное изменение к BCE2 — objective;
initial validation идентична. Fresh CE50+CE2 на PV: 0.063086 NDCG@10 /
0.259853 Recall@100, alpha=0.4. К CE50+BCE2: +8.4% / −4.9%, оба paired CI
исключают ноль; прежняя смесь сохранена в MPS frontier.

B=1 fresh long / fresh hybrid: CPU 1.04 / 2.37 мс, MPS 1.55 / 2.90 мс.
Full serving учитывает обе таблицы и scoring/normalization, поэтому masked
recent2 пока не даёт дешёвого real-time encoder. Сохранены CPU/MPS B=256
throughput, память, mixed-workload checks, CT-budget selection и PV frontier.
68 тестов, rank-oracles, frozen state и hashes проверены. Test не открыт.
[Итоговый отчёт](../reports/experiments/sasrec-256-hybrid-ce.md).

Команды исходных запусков ниже. Для нового повторения нужны новые output paths
и согласованные `hybrid_run` в resource configs; существующие результаты
не перезаписываются. Checkpoints и большие NPZ находятся локально в `data/runs/`.

```sh
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_hybrid_sasrec.py --config configs/experiments/sasrec-256-hybrid-ce.json --device mps --output data/runs/sasrec-256-hybrid-ce

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-ce-resources.json --device cpu --output data/runs/sasrec-256-hybrid-ce-resources/cpu.json
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-ce-resources.json --device mps --output data/runs/sasrec-256-hybrid-ce-resources/mps.json
env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/profile_sasrec_hybrid.py --config configs/experiments/sasrec-256-hybrid-bce-resources.json --device mps --output data/runs/sasrec-256-hybrid-ce-resources/bce-mps.json

.venv/bin/python scripts/plot_sasrec_hybrid.py --input data/runs/sasrec-256-hybrid-ce/results.json --selection data/runs/sasrec-256-hybrid-ce/selection.json --output reports/experiments/sasrec-256-hybrid-ce
.venv/bin/python scripts/plot_sasrec_frontier.py --input data/runs/sasrec-256-hybrid-ce-resources/cpu.json data/runs/sasrec-256-hybrid-ce-resources/mps.json --bce-reference data/runs/sasrec-256-hybrid-ce-resources/bce-mps.json --output reports/experiments/sasrec-256-hybrid-ce

env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m unittest discover -s tests -v
```
