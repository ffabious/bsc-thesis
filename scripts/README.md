# Скрипты

Текущие скрипты: локальная сборка LaTeX и подготовка Yambda. Содержимое `legacy/` импортировано из предыдущей работы.

| Путь | Назначение |
| --- | --- |
| `build-thesis.sh` | Сборка диплома через `latexmk`, pdfLaTeX и Biber; результат — `build/latex/thesis.pdf` |
| `prepare_yambda.py` | Проверенная загрузка Yambda-50M, аудит, глобальный chronological split и train-only словари; результаты только в локальном `data/` |
| `prepare_audio_embeddings.py` | Загрузка закреплённых готовых аудиовекторов с SHA-256, индекс покрытия и строгие локальные таблицы без fallback; `prepare_yambda.py --audio-items` пересобирает все части |
| `check_yambda.py` | Независимая проверка выгруженных Parquet: SHA-256, порядок, пересечение split, строгая прошлая история и соответствие словарю |
| `sasrec.py` | SASRec, строго прошлые prefixes, negative sampling и полные ранги каталога; короткий suffix сохраняет исходные right-aligned learned positions |
| `run_sasrec.py` | Локальные CPU/MPS benchmark, обучение с выбором по model_valid и оценка checkpoint; `epochs: null` оставляет только остановку по validation; `loss_reduction` выбирает separate/joint mean BCE, `context_keep` маскирует ранние токены без сдвига позиций; поддерживает frozen-аудио и перенос ID-таблицы |
| `compare_sasrec.py` | Оценка на исходных событиях/targets и каталоге прежнего пилота; сопоставление item_id, входная маска checkpoint и парные пользовательские интервалы |
| `blend_sasrec.py` | Нормировка scores по кандидатам запроса, выбор веса 50/2 только на validation и отдельная test-оценка замороженной смеси; без обучения |
| `replay_sasrec.py` / `plot_sasrec_replay.py` | Диагностические fixed-lag fresh/stale snapshots, локальные component costs, кривые и CSV; не actual refresh schedules |
| `replay_encoder_pairs.py` | Frozen CE/student42/43 на исходных fixed-lag snapshots; reuse CE42, fresh alignment, CPU oracle, own staleness loss и парная разница потерь, JSON/CSV/PNG/SVG. Без обучения, test и serving benchmark |
| `replay_sasrec_policies.py` / `plot_sasrec_policies.py` | Реальные on-request always/never/time/events/OR schedules на CT/PV; paired quality, initialization/refresh counts, измеренные ветви стоимости, frontier и CT-budget selection; test запрещён |
| `extend_refresh_grid.py` | Заранее заданные простые правила в разрывах сетки, reuse старых ranks, общая fixed cost calibration, CT-only comparator и парные сравнения контроллера |
| `refresh_controller.py` / `train_refresh_controller.py` | CPU ridge/MLP immediate-gain controller: causal features, request-balanced labels, CT10/4 или8/2/4 fit/early-stop/tuning, fixed PV и полная стоимость; без обучения энкодера и test |
| `mlp_refresh_controller.py` / `report_controller_models.py` | MLP1377 параметров с CPU early stopping, NumPy scalar serving, matched ridge и два seed; одинаковые labels, dense106 baseline, paired CI и expected-mixture reference |
| `benchmark_refresh_controller.py` | Совместные interleaved B=1 plain/controller hit/refresh и mixed workloads: CPU признаки/решение внутри MPS pipeline, без двойного начисления CPU |
| `controller_horizon.py` / `prepare_controller_horizon.py` / `report_controller_horizon.py` | Фиксированный mean held-cache gain по5 запросам, fit/early boundaries, reuse рангов; сравнение с immediate MLP, скалярный label audit и paired CI |
| `list_cache.py` / `prepare_list_cache.py` / `benchmark_list_cache.py` / `report_list_cache.py` | Canonical top100, полный quality audit и парные direct vector/list workloads109 политик; direct mode заменяет не прошедшую проверку additive cost model; CT-only budget selection |
| `prepare_top100_policies.py` / `check_policy_serving.py` / `report_policy_transfer.py` | Grid106 на готовых student42/CE1k: один top100 на snapshot, explicit outside sentinel; matched MLP, own-fresh paired comparisons и прямые serving checks выбранных на CT правил |
| `freeze_user_holdout.py` / `final_holdout.py` / `prepare_user_holdout.py` / `replay_user_holdout.py` / `report_user_holdout.py` | Train-only user exclusion/selection, committed protocol guard, frozen vocabulary retokenization, single final replay106+2 и prespecified bootstrap family; generic request reader отделён от CT/PV-only wrapper |
| `report_student_maintenance.py` | Сравнение student43/CE42 на100 политиках, CT-selected controllers и matched simple baselines; ожидание CT-выбранных смесей, paired CI, CSV и общий frontier |
| `replay_hybrid_sasrec.py` / `plot_sasrec_hybrid.py` | Смесь cached long / fresh recent scores с per-query z-score, alpha на CT для каждой политики и frozen PV evaluation; endpoints, CPU oracle, проверяемый resume mixture grids, CSV и quality curves; test запрещён |
| `profile_sasrec_hybrid.py` / `plot_sasrec_frontier.py` | Полные CPU/MPS serving paths long/recent/hybrid, B=1/256, память, реальные refresh counts и смешанные workload checks; CT-budget selection и frontier по измеренной полной стоимости, включая прежнюю BCE-смесь |
| `check_sasrec_short.py` | Проверка native recent2 против padded50: все CT/PV vectors, sample full-catalog scores, все recent ranks; `--full-replay` пересчитывает все выбранные смеси с прежними alpha |
| `benchmark_sasrec_short.py` | Парные замеры короткого/padded encoder, recent-only и fresh hybrid на одних CT запросах; CPU/MPS B=1/256, перемешанный порядок, sync, memory stage snapshots |
| `calibrate_sasrec_hybrids.py` | Сравнительная калибровка CE/BCE в одном MPS-процессе при дрейфе отдельных профилей: чередует все пути, разделяет общий long-контроль и пересчитывает стоимость и mixed-workload checks |
| `distill_sasrec.py` / `summarize_sasrec_distillation.py` | Учитель CE50+CE2 с alpha на model_valid, matched warm-start CE/KD, полный KL batchmean, выбор weight до CT/PV и парные serving benchmarks; JSON/CSV, training curves и отчёт. `run_sasrec.py` поддерживает `initial_checkpoint` с best epoch0 |
| `check_sasrec_initialization.py` / `summarize_sasrec_distillation_cold.py` | Восстановление случайной CE50-инициализации из закреплённого Git source, проверки state/RNG/initial validation; один KD student с нуля и готовый CE-контроль, отчёт и стоимость. Cold state/RNG guard в `run_sasrec.py` не делает epoch0 кандидатом |
| `compare_student_teachers.py` | Готовые CE50 и reference student против нового student; reference ranks могут поступать из прежнего сравнения с проверкой событий и checkpoint. Фиксированные model_valid/CT/PV, paired CI, history/popularity срезы, users/days, rank transitions, training curves и явные model labels. Без test; `distill_sasrec.py` загружает single teacher без recent-ветви |
| `prepare_seed_pair.py` / `compare_seed_pairs.py` | Парный повтор CE50/student с новым seed: сохранённая random initialization, собственный frozen teacher, одинаковые события/каталог и сравнение с готовой парой seed42. Общие метрики, paired CI внутри seed, history/popularity срезы и validation curves; без test |
| `diagnostics/sage_fd_check.py` | Самостоятельная численная диагностика опубликованного псевдокода SAGE; требует NumPy |
| `diagnostics/check_kv_reuse.py` | CPU FP32 projected-KV prototype на frozen SASRec: fixed append, right-aligned append, eviction, one-layer control и полный rebuild;70 ожидаемых исходов без dataset events |
| `diagnostics/audit_pilot_scale.py` | Проверка общей когорты и строго прошлых histories, естественные prefixes и сегменты качества 256/1k |
| `diagnostics/audit_joint_repeats.py` | Локальные inference-абляции выбранных повторов joint-BCE моделей, tracing encoder и полный validation-контроль короткого контекста; без обучения |
| `diagnostics/audit_cold_distillation.py` / `check_student_context.py` / `summarize_student_quality.py` | Разбор student/CE50/teacher на model_valid и сохранённых CT/PV: типы истории, популярность, пользователи, rank transitions, paired CI и input-ablation длинного контекста. Готовые веса, без test |
| `diagnostics/audit_pilot_training.py` | Воспроизведение validation/initial model, scoring-абляции и проверка gradient weighting BCE без optimizer updates |
| `legacy/literature/` | Сборка каталога и HTML-обзоров, добавление источников и прежние проверки |
| `legacy/conferences/` | Первоначальная сборка и обновление таблицы конференций |
| `legacy/registry/dcn/` | Историческое добавление DCN V2 |
| `legacy/registry/reinforce/` | Историческое добавление работ по REINFORCE |

В `legacy/` сохранены абсолютные пути прежнего окружения `/workspace/scratch/...`, старые расположения `output/` и `tmp/`, а также зависимости от Playwright и `@oai/artifact-tool`. Перед использованием нужны адаптация путей и проверка зависимостей. Некоторые скрипты изменяют другие скрипты и перезаписывают реестр; их запуск не является способом восстановить текущий комплект из 78 работ. Старые проверки рассчитаны на 75 записей.

Входные данные и свидетельства находятся в `research/literature/`, предыдущие версии реестра — в `research/registry/`. Сохранённый результат диагностики SAGE: `research/literature/sage-diagnostics.json`. При прямом запуске диагностический скрипт запишет новый `sage-diagnostics.json` рядом с собой.
