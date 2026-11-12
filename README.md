# BSc thesis

**A Study of Efficient User Representation Computation and Maintenance in Sequential Recommender Systems**

Материалы диплома Кирилла Грешнова. Текущая постановка сохранена в [пропозале от 9 октября 2026](docs/proposal/proposal-2026-10-09.pdf) и его [текстовом снимке](docs/proposal/proposal-snapshot.md): эффективное вычисление и обновление пользовательских представлений в последовательных рекомендательных системах, офлайн-эксперименты на Yambda-50M.

Репозиторий: [ffabious/bsc-thesis](https://github.com/ffabious/bsc-thesis).

## Структура

| Папка | Содержимое |
| --- | --- |
| `thesis/` | LaTeX-исходники: `thesis.tex`, главы, иллюстрации, библиография и титульный лист |
| `docs/proposal/` | Пропозал, извлечённый текст и снимок постановки |
| `docs/guidelines/` | Оригинальные требования к оформлению, извлечённый текст и HTML-памятка |
| `notes/` | Первоначальные идеи и обсуждение темы |
| `references/` | Актуальный реестр литературы в CSV/JSON и библиография BibTeX |
| `reports/literature/` | HTML-обзор, презентация и разборы Jev/SAGE |
| `reports/conferences/` | Таблица конференций и воркшопов, источники и изображения предпросмотра |
| `research/` | Собранные свидетельства, промежуточные данные и предыдущие версии реестра |
| `scripts/` | Подготовка и проверка данных, сборка диплома, диагностика и исторические скрипты |
| `configs/data/` | Закреплённый источник Yambda и параметры хронологического split |
| `data/` | Локальные raw и подготовленные Parquet; исключены из Git |
| `reports/data/` | Отчёт подготовки и агрегированные manifests без исходных событий |
| `tests/` | Проверки протокола подготовки данных |
| `assets/` | Исходное загруженное изображение |
| `archive/` | Оригинальные вложенные ZIP, распакованные исторические материалы и карта импорта |

## С чего начать

- [Пропозал](docs/proposal/proposal-2026-10-09.pdf) — текущая постановка; ранние заметки отражают историю обсуждения.
- [План экспериментов](notes/experiment-plan.md) — последовательность работ, критерии перехода и полностью локальные вычисления; [результаты подготовки данных](reports/data/preparation.md).
- [Реестр статей](references/papers.csv) — 84 записи по состоянию на 1 ноября 2026; [JSON](references/papers.json) и [BibTeX](references/references.bib) лежат рядом. Файлы для импорта в Marginalia: [Attention Is All You Need](references/imports/attention-is-all-you-need.bib) (`P081_AttentionIsAllYouNeed`); [YouTube 2016, COLD и RecFlow](references/imports/multi-stage-recommenders.bib) (`P082_YouTubeDNN`, `P083_COLD`, `P084_RecFlow`).
- [Аудит надёжности всех 84 источников](notes/bibliography-reliability-audit.md) — проверка первичных метаданных, публикационного статуса и допустимых выводов от 1 ноября 2026; без независимой репликации экспериментов.
- [Черновик семи секций Literature Review](notes/literature-review-draft.md) — краткий текст со ссылками у конкретных механизмов; по результатам аудита и выбору автора используются только источники категории `established`. [План и карта исходного покрытия](notes/introduction-literature-review-plan.md) лежат рядом.
- [Обзор литературы](reports/literature/literature-review.html) и [презентация](reports/literature/literature-brief.html) — открыть локально в браузере. Часть обзоров ещё описывает прежнюю подборку из 75 работ; актуальный состав находится в реестре.
- [Конференции и воркшопы](reports/conferences/conferences-and-workshops-2026.xlsx) — сохранённый снимок, согласно исходному описанию обновлённый 19 октября 2026. Даты не обновляются автоматически.
- [LaTeX-шаблон](thesis/thesis.tex) — содержит демонстрационные главы и титульный лист. Библиография `thesis/ref.bib` пока является частью шаблона; реестр литературы находится в `references/references.bib`.

## Работа с текстом диплома

Главный файл — `thesis/thesis.tex`. Главы находятся в `thesis/chapters/`, изображения — в `thesis/figs/`. Шаблон можно загрузить в Overleaf целиком папкой `thesis/`; исходная [памятка шаблона](thesis/README.md) сохранена.

На macOS локальный компилятор и обработчик библиографии устанавливаются через Homebrew:

```sh
brew install texlive biber
```

### VS Code

1. Открыть корень репозитория в VS Code и установить рекомендованное расширение **LaTeX Workshop** (`James-Yu.latex-workshop`).
2. Открыть `thesis/thesis.tex` или файл главы из `thesis/chapters/`.
3. Сохранение `.tex` запускает сборку. Вручную: **LaTeX Workshop: Build LaTeX project**, на macOS — **⌘⌥B**.
4. Для PDF справа: **LaTeX Workshop: View LaTeX PDF file**, на macOS — **⌘⌥V**. Включён SyncTeX для переходов между исходником и PDF.

Та же сборка из терминала:

```sh
bash scripts/build-thesis.sh
```

PDF находится в `build/latex/thesis.pdf`. `latexmk` автоматически запускает Biber и повторяет LaTeX до согласования ссылок и оглавления. Настройки VS Code и скрипт учитывают Homebrew на Apple Silicon и Intel; менять глобальные настройки редактора не требуется. Служебные файлы, PDF сборки, локальные окружения и кэши исключены через `.gitignore`. Исходные PDF, HTML, таблицы и изображения входят в историю.

## Локальные данные для экспериментов

Подготовка, обучение и оценка выполняются на локальной машине. Облачные вычислительные ресурсы не используются. Требуется Python 3.11+.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python scripts/prepare_yambda.py --download
.venv/bin/python scripts/prepare_yambda.py --users 256 --output data/processed/yambda50m-pilot
.venv/bin/python scripts/check_yambda.py data/processed/yambda50m
.venv/bin/python -m unittest discover -s tests -v
```

Первый запуск загружает закреплённый официальный файл Yambda-50M и проверяет SHA-256. Подготовка выполняется локально, по умолчанию в 4 потока с лимитом памяти DuckDB 4 GB. Существующие результаты не перезаписываются: для повторного запуска задайте новый `--output`. Разделение и словари сохраняются в `data/processed/`, параметры и определения полей описаны в [плане](notes/experiment-plan.md). Данные и `.venv` не публикуются. Локальный пилот обучения описан ниже.

## SASRec на пилоте

Локальные CPU/MPS training-step benchmarks, обучение и полная оценка каталога (Python 3.12+):

```sh
.venv/bin/python -m pip install -r requirements-training.txt
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device cpu --output data/runs/sasrec-pilot/benchmark-cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --output data/runs/sasrec-pilot/benchmark-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --output data/runs/sasrec-pilot/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device cpu --checkpoint data/runs/sasrec-pilot/train-mps/best.pt --output data/runs/sasrec-pilot/evaluation-cpu.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --checkpoint data/runs/sasrec-pilot/train-mps/best.pt --output data/runs/sasrec-pilot/evaluation-mps.json
```

Настройки находятся в [sasrec-pilot.json](configs/experiments/sasrec-pilot.json), результаты — в [отчёте пилота](reports/experiments/sasrec-pilot.md). Checkpoint и предсказания остаются в локальном `data/`. Существующий training output не перезаписывается. Для сравнения устройств используются одинаковые batch/FP32-настройки и один выбранный checkpoint; MPS fallback на CPU запрещён.

## Замороженные аудиовекторы

Текущий вариант использует пересобранные наборы `yambda50m-audio` и `yambda50m-pilot-audio`: все пять частей содержат только train-known треки с готовыми аудиовекторами. Реальных событий с UNK и подстановок пропущенных векторов нет. [Подготовка и команды](reports/data/audio-preparation.md), [результаты обучения](reports/experiments/sasrec-frozen-audio.md).

```sh
.venv/bin/python -m pip install -r requirements-audio.txt
HF_XET_HIGH_PERFORMANCE=1 .venv/bin/python scripts/prepare_audio_embeddings.py --download --index-only
.venv/bin/python scripts/prepare_yambda.py --audio-items data/raw/yambda-audio/items.parquet --output data/processed/yambda50m-audio
.venv/bin/python scripts/prepare_yambda.py --users 256 --audio-items data/raw/yambda-audio/items.parquet --output data/processed/yambda50m-pilot-audio
.venv/bin/python scripts/prepare_audio_embeddings.py --data data/processed/yambda50m-audio
.venv/bin/python scripts/prepare_audio_embeddings.py --data data/processed/yambda50m-pilot-audio
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-frozen-audio.json --output data/runs/sasrec-frozen-audio/train-mps
```

Frozen-аудио таблица 128D используется с обучаемыми проекциями 128→64 и 64→128; обучаются 70 144 параметра ранкера и позиций. Контрольный запуск на тех же данных: `configs/experiments/sasrec-audio-baseline.json`. Данные, исходный файл аудиовекторов и checkpoints хранятся локально и исключены из Git; публикуются код, конфигурации и агрегированные результаты.

## Замороженные ID-векторы

Отдельный запуск переносит только обученную 64D-таблицу треков из контрольного SASRec на том же пересобранном пилоте. Все остальные веса инициализируются заново; обучаются 53 760 параметров. Аудиопризнаки в модель не подаются. [Протокол и результаты](reports/experiments/sasrec-frozen-id.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-frozen-id.json --output data/runs/sasrec-frozen-id/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-frozen-id.json --checkpoint data/runs/sasrec-frozen-id/train-mps/best.pt --output data/runs/sasrec-frozen-id/evaluation-mps.json
```

Источник в конфиге закреплён по SHA-256 и должен соответствовать завершённому обучению и тому же manifest данных. Для воспроизведения с заново обученным источником обновите путь и `item_checkpoint_sha256` под этот checkpoint и выберите новую output-директорию. Стоимость предварительного обучения учитывается отдельно от нового запуска.

## Контрольный SASRec до остановки по validation

На том же пересобранном пилоте контроль обучается с прежними настройками и seed 42. `epochs: null` снимает лимит эпох; `patience: 2` останавливает запуск после двух эпох без улучшения validation NDCG@10. Test оценивается после выбора checkpoint. [Конфигурация](configs/experiments/sasrec-control-early-stop.json), [результаты](reports/experiments/sasrec-control-early-stop.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-control-early-stop.json --output data/runs/sasrec-control-early-stop/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-control-early-stop.json --checkpoint data/runs/sasrec-control-early-stop/train-mps/best.pt --output data/runs/sasrec-control-early-stop/evaluation-mps.json
```

Запуск начинается с нуля: прежний checkpoint не содержит Adam/RNG state для точного продолжения. Первые пять эпох повторяются в том же запуске, затем обучение продолжается с сохранением optimizer/RNG в памяти. Предыдущие checkpoints и результаты не перезаписываются.

## SASRec на 1 000 пользователей

Пилот увеличен до 1 000 пользователей с тем же train-фильтром и seed 42. Модель и правило остановки прежние. [Данные, обучение и сравнение с 256 пользователями](reports/experiments/sasrec-1k.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-1k-early-stop.json --output data/runs/sasrec-1k-early-stop/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-1k-early-stop.json --checkpoint data/runs/sasrec-1k-early-stop/train-mps/best.pt --output data/runs/sasrec-1k-early-stop/evaluation-mps.json
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_sasrec.py --device mps --checkpoint data/runs/sasrec-1k-early-stop/train-mps/best.pt --reference-evaluation data/runs/sasrec-control-early-stop/evaluation-mps.json --output data/runs/sasrec-1k-early-stop/comparison-256.json
```

Основное сравнение использует ровно прежние test-события, histories и каталог 114 125 треков, с сопоставлением словарей по item_id. Полная оценка нового каталога 244 194 треков сохраняется отдельно: её метрики относятся к другому набору кандидатов и пользователей. Checkpoints и ranks остаются локальными.

Причины ухудшения исследованы в [аудите данных, обучения и сравнения](reports/experiments/sasrec-1k-audit.md); связанные первичные работы и различие нормировок BCE разобраны в [литературной заметке](notes/sasrec-scaling-literature.md). Диагностические скрипты используют готовые checkpoints; отдельный короткий контроль ограничен одной эпохой.

## BCE с общим усреднением на 1 000 пользователей

Меняется только `loss_reduction: joint_mean`: сумма positive и 16 negative ошибок делится на общее число logits. Прежний режим `separate_mean` остаётся default для старых конфигов. Данные, sampler, seed, модель, Adam, clipping и patience 2 сохранены. [Конфигурация](configs/experiments/sasrec-1k-joint-bce.json), [результаты сравнения](reports/experiments/sasrec-1k-joint-bce.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-1k-joint-bce.json --output data/runs/sasrec-1k-joint-bce/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-1k-joint-bce.json --checkpoint data/runs/sasrec-1k-joint-bce/train-mps/best.pt --output data/runs/sasrec-1k-joint-bce/evaluation-mps.json
```

Основное A/B-сравнение использует все те же 175 831 test-targets и каталог из 244 194 треков. Отдельно сохраняется оценка на фиксированной когорте и каталоге прежнего пилота на 256. Выбор checkpoint производится среди обученных эпох по validation NDCG@10, как в контроле.

## BCE с общим усреднением на 256 пользователей

Тот же пилот и настройки `sasrec-control-early-stop`, единственное изменение — `loss_reduction: joint_mean`. Обучение локально на MPS до прежнего patience 2 по validation NDCG@10. [Конфигурация](configs/experiments/sasrec-256-joint-bce.json), [сравнение со старым 256 и новым 1k](reports/experiments/sasrec-256-joint-bce.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config configs/experiments/sasrec-256-joint-bce.json --output data/runs/sasrec-256-joint-bce/train-mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config configs/experiments/sasrec-256-joint-bce.json --checkpoint data/runs/sasrec-256-joint-bce/train-mps/best.pt --output data/runs/sasrec-256-joint-bce/evaluation-mps.json
```

Все три модели сравниваются на одинаковых 45 711 test-событиях, histories и 114 125 кандидатах. Для 1k переиспользуются сохранённые ранги `data/runs/sasrec-1k-joint-bce/comparison-256.npz`; полная 1k-оценка с другим каталогом в эту таблицу не включается.

Причины оставшейся потери на повторах исследованы в [аудите joint-BCE моделей](reports/experiments/sasrec-joint-repeat-audit.md). Короткий контекст восстанавливает верхние попадания, но при inference ухудшает общий Recall. [Обучающий контроль короткого контекста](notes/sasrec-next-repeat-experiment.md) выполнен для обоих размеров; результаты приведены ниже.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/diagnostics/audit_joint_repeats.py --device mps --full-validation
```

## Обучение с двумя последними событиями

Контроли на 256 и 1 000 пользователей используют `context_keep: 2`: ранние токены маскируются при train, validation и test. `length: 50`, learned positions и остальные настройки сохранены. [Сравнение четырёх моделей](reports/experiments/sasrec-recent-2.md).

```sh
for pilot_size in 256 1k; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --device mps --config "configs/experiments/sasrec-${pilot_size}-recent-2.json" --output "data/runs/sasrec-${pilot_size}-recent-2/train-mps"
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py evaluate --device mps --config "configs/experiments/sasrec-${pilot_size}-recent-2.json" --checkpoint "data/runs/sasrec-${pilot_size}-recent-2/train-mps/best.pt" --output "data/runs/sasrec-${pilot_size}-recent-2/evaluation-mps.json"
done
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_sasrec.py --device mps --checkpoint data/runs/sasrec-1k-recent-2/train-mps/best.pt --reference-evaluation data/runs/sasrec-256-joint-bce/evaluation-mps.json --output data/runs/sasrec-1k-recent-2/comparison-256.json
```

Сравнение использует одни исходные события, targets и кандидатов; видимый контекст намеренно различается у моделей 50 и 2. `compare_sasrec.py` применяет маску из checkpoint и отражает это в metadata. Полная native-оценка 1k сохраняется отдельно.

## Смешивание оценок моделей с историей 50 и 2

Готовые joint-BCE модели заморожены. Для каждого запроса scores каждой модели приводятся к среднему 0 и стандартному отклонению 1 по текущему candidate set; нормировка не использует labels. Смесь: `(1-alpha) * z50 + alpha * z2`. Это согласование шкал для ранжирования. [Протокол](configs/experiments/sasrec-score-blend.json), [результаты](reports/experiments/sasrec-score-blend.md).

Для каждого размера пилота alpha выбирается отдельно по максимуму NDCG@10 на полной `model_valid`; сетка 0–1 с шагом 0.1, при равенстве выбирается меньший alpha. Обе selection фиксируются до test. На общей старой когорте 1k используется тот же alpha, который выбран на native validation.

```sh
for pilot_size in 256 1k; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/blend_sasrec.py select --pilot "$pilot_size" --device mps --output "data/runs/sasrec-score-blend/$pilot_size"
done
for pilot_size in 256 1k; do
  PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/blend_sasrec.py evaluate --pilot "$pilot_size" --device mps --output "data/runs/sasrec-score-blend/$pilot_size"
done
```

`select` читает только `model_valid` и предшествующий train. `evaluate` читает сохранённую selection и не изменяет коэффициент. Скрипт проверяет hashes моделей, воспроизведение validation endpoints и согласованность test endpoints с отдельными моделями. Полные матрицы scores хранятся только для текущего batch; локально сохраняются ranks.

## Увеличенные SASRec на 256 и 1k

Локальный последовательный эксперимент с item embedding 64, hidden 256, четырьмя
блоками / восемью головами и FFN 1024. Для 256 обе истории, 50 и 2, обучены до
validation-остановки; 1k отменена пользователем. Прежние данные, OOT, общий BCE и batch 256 сохранены.
Короткое окно физически имеет длину 2. [Протокол](notes/sasrec-large-pilots.md),
[замеры и результаты](reports/experiments/sasrec-large.md).

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py benchmark --device mps --config configs/experiments/sasrec-256-large-long.json --output data/runs/sasrec-256-large-long/benchmark-mps.json
caffeinate -i .venv/bin/python scripts/run_sasrec_pairs.py
.venv/bin/python scripts/summarize_sasrec_pairs.py
```

Для остальных benchmark меняются `256/1k` и `long/recent` в конфигурации и output.
Orchestration по умолчанию обрабатывает только `256-large`; `last.pt` включает optimizer и RNG,
индивидуальная train-команда принимает `--resume`. Alpha выбирается по
validation до test. Отменённые 1k и full-data обучение автоматически не запускаются.

## Контроль авторских гиперпараметров SASRec

Один локальный запуск на прежних 256 пользователях с историей 50: hidden/item dim 50,
два блока, одна голова, dropout 0.5, batch 128 и один negative. Наши OOT, примеры,
полный каталог и patience 2 сохранены. Это перенос гиперпараметров в существующую
реализацию, а не точное воспроизведение TensorFlow-training авторов.
[Протокол и команды](notes/sasrec-author-params-experiment.md),
[конфигурация](configs/experiments/sasrec-256-author-params.json),
[результаты сравнения](reports/experiments/sasrec-256-author-params.md).

## Dim 50 / одна голова с прежним обучением

От успешной 256 joint-BCE меняются только dim 64 → 50 и головы 2 → 1;
история 50, 16 negatives, общий BCE, batch 256 и dropout 0.2 сохранены.
[Команды](notes/sasrec-256-d50-h1.md), [результаты](reports/experiments/sasrec-256-d50-h1.md).
Дальнейшие контроли выбираются по validation в [автономном цикле baseline](notes/sasrec-baseline-search.md).

## Сильный encoder baseline: полный CE

На прежнем 256-пилоте проверены размерности/головы, dropout, sampled CE16 и полный
CE. Выбран полный CE при исходных dim 64, двух головах/блоках, истории 50 и dropout
0.2. Средний test двух seed: NDCG@10 0.045531 / Recall@100 0.230732 против
0.026218 / 0.217202 у BCE. Архитектура и обычный inference сохранены.
[Отчёт и команды](reports/experiments/sasrec-baseline-search.md),
[выбранные checkpoint и hashes](reports/experiments/sasrec-baseline-search/selected-baseline.json),
[основной конфиг](configs/experiments/sasrec-256-full-ce.json).

`run_sasrec.py evaluate --split model_valid` сохраняет validation ranks для отбора
рецептов; по умолчанию evaluate использует test. `objective` выбирает `bce`,
`sampled_ce` или `full_ce`; без этого поля сохранено прежнее поведение BCE.

## Полный CE на 1 000 пользователей

Тот же рецепт перенесён на прежний 1k-набор; конфиг отличается от 256 full CE
только путём данных. 4 эпохи, best 2, 52.66 мин локально. Native NDCG выросла
на 65.0% относительно 1k BCE, Recall снизился на 3.3%. На точных событиях и
кандидатах 256 новая 1k-модель уступает 256 CE по обеим метрикам.
[Команды](notes/sasrec-1k-full-ce.md), [результаты и проверки](reports/experiments/sasrec-1k-full-ce.md).

## Replay fresh/stale на 256 CE

Frozen encoder сравнивается на одних событиях controller_train / policy_valid
при лагах по новым прослушиваниям и времени. Timestamp-группы неделимы;
test не читается. На policy_valid минимальный event lag 1 снижает NDCG на 21.5%,
суточный time lag — на 51.5%. Отдельно измерены encoder, scoring, CPU-cache
и end-to-end cost для batch 1/256. Фиксированный лаг не является refresh-политикой.
[Команды и зависимости](notes/sasrec-256-replay.md),
[отчёт, PNG/SVG и CSV](reports/experiments/sasrec-256-replay.md).

## Refresh-политики на 256 CE

31 on-request политика (always/never, time, events, hybrid OR) проверена на всех
62 129 CT / 29 914 PV timestamp-группах. Quality–cost frontier включает точные
счётчики initialization/refresh и измеренные локальные B=1 ветви refresh/write/serve
и hit/serve. На PV гибрид 2 события OR 5 минут: −18.4% расчётной полной стоимости,
−8.6% NDCG, −0.9% Recall; настройка выбрана на CT. Fresh/CPU oracle и 55 тестов
прошли, test не читается. [Протокол](notes/sasrec-256-policies.md),
[отчёт, PNG/SVG, CSV и ограничения стоимости](reports/experiments/sasrec-256-policies.md).

Сетка расширена до **100 политик** с TTL 30m/1h/3h/6h/12h/24h и
10/20/60/120/240/480 новыми прослушиваниями, сохранены прежние контроли.
На PV TTL 1h даёт refresh на 10.31% запросов, 6h — на 4.94%, 24h — на 2.79%.
Добавлены refresh/hit counts и проценты для CT/PV, качество по порогам и новый
frontier. Прежние 31 quality/cost точки переиспользованы без изменений;
56 тестов и CPU oracle новых пар прошли, test не открыт.
[Расширенный отчёт и статистика](reports/experiments/sasrec-256-policies-wide.md),
[протокол и команды](notes/sasrec-256-policies-wide.md).

## Смесь long CE и свежей recent-only модели на 256

Готовые CE50 и BCE2 checkpoints смешаны после per-query z-score; recent всегда
свежая, long — по прежним 100 refresh-политикам. Для каждой политики alpha
выбирается только на CT и фиксируется до расчёта смеси на PV. Fresh hybrid:
**0.058224 / 0.273190 NDCG@10 / Recall@100** против **0.055093 / 0.252201**
у fresh long (+5.7% / +8.3%), alpha=0.3. Recent-only: 0.039578 / 0.216979.
При суточном long-cache hybrid даёт 0.042870 / 0.239591, alpha=0.7;
потеря к fresh уменьшается, но остаётся. Нового обучения нет, test не открыт,
62 теста и CPU oracle пройдены. Стоимость двухветвевого serving не измерялась.
[Отчёт и графики](reports/experiments/sasrec-256-hybrid.md),
[протокол](notes/sasrec-256-hybrid.md).

## Recent full CE и измеренная стоимость смеси на 256

Recent2 переобучена с единственной сменой BCE → full CE: 9 эпох, best 7,
18.70 мин на MPS. На policy_valid свежая CE50+CE2 смесь (alpha=0.4 с CT)
даёт **0.063086 / 0.259853 NDCG@10 / Recall@100**. К прежней CE50+BCE2:
**+8.4% NDCG, −4.9% Recall**; оба парных интервала исключают ноль.
К fresh long: +14.5% / +3.0%. Прежняя смесь сохраняет лучший Recall.

Полный B=1 serving fresh long / fresh hybrid: CPU **1.04 / 2.37 мс**,
MPS **1.55 / 2.90 мс**. Masked recent2 пока обрабатывает tensor длины 50
и не дешевле длинного encoder. Frontier учитывает обе item-таблицы,
нормировку, смешивание, cache init/refresh/hit и final top-100; CPU и MPS
показаны отдельно, прежняя BCE-смесь добавлена в MPS-сравнение.
Test не открыт; 68 тестов, независимые rank-oracles и hashes проверены.
[Отчёт, ресурсы, графики и CSV](reports/experiments/sasrec-256-hybrid-ce.md),
[протокол и команды](notes/sasrec-256-hybrid-ce.md).

## Native recent2 inference

Recent получает настоящий tensor длины 2 с прежними learned positions 48/49;
checkpoints и alpha сохранены. На всех CT/PV targets и 100 политиках обеих
смесей NDCG/Recall/Hit@1 совпали с padded50. CPU vectors точны побитово,
MPS — в допусках FP32. 71 тест пройден, test не открыт.

Парный batch256-контроль: encoder быстрее **11.9× CPU / 2.8× MPS**, полная
смесь — только **1.07× / 1.03×**. При B=1 recent-only на CPU быстрее на 7.4%,
на MPS медленнее на 8.3%; общего ускорения hybrid B=1 нет. Полный frontier
пересчитан, CE/BCE MPS paths чередуются в одном процессе из-за дрейфа отдельных
замеров; близкие cost-точки требуют осторожной интерпретации.
[Отчёт, память, paired timings и frontier](reports/experiments/sasrec-256-short-inference.md),
[команды](notes/sasrec-256-short-inference.md).

## Дистилляция CE50+CE2 на 256

Alpha учителя заново выбран на model_valid: **0.4**. Control и KD weights
0.25/1 дообучены из одного CE50 checkpoint с новым Adam до patience2.
Все три запуска остановились после двух эпох; по NDCG лучшими остались
исходные веса (epoch0). Дистилляция в этой warm-start постановке baseline
не улучшила; общий дополнительный training занял около **26.3 минуты**.

У последней KD1 модели validation Recall вырос на 2.7%, NDCG снизился на 0.7%.
При зафиксированном отборе по NDCG выбранные модели на PV сохраняют
**0.055093 / 0.252201**, учитель даёт **0.063086 / 0.259853**.
Стоимость одиночного inference прежняя; KD training примерно втрое дороже
CE-дообучения. 77 тестов и artifact checks пройдены, test не открыт.
[Отчёт, кривые и стоимость](reports/experiments/sasrec-256-distillation.md),
[протокол и команды](notes/sasrec-256-distillation.md).

## Student с нуля на 256

Фиксированные alpha=0.4, T=2, lambda=1; один новый локальный KD run,
готовый CE50 переиспользован. Начальные 29 тензоров и validation ranks
совпали с восстановленной CE50-инициализацией; state/RNG guard пройден
в фактическом training loop. 8 эпох, best6, **46.94 мин** против исторических
12.63 мин CE50. На model_valid обе метрики выше контроля.

На PV student даёт **0.053961 / 0.264512 NDCG@10 / Recall@100**:
**−2.05% / +4.88%** к CE50. NDCG CI включает ноль, Recall CI выше нуля;
перенос выигрыша NDCG не подтверждён. Fresh serving практически одинаков:
CPU B1 1.33мс, MPS B1 1.87мс. 79 тестов, artifact checks, test закрыт.
Исходные cache/frontier результаты продолжают относиться к CE50.
[Отчёт, CI, кривые и ресурсы](reports/experiments/sasrec-256-distillation-cold.md),
[протокол и команды](notes/sasrec-256-distillation-cold.md).

## Разбор качества student

Преимущество model_valid NDCG имеет парный CI, включающий ноль. Student уже
там хуже на повторах последнего трека; на PV потери усиливаются, а выигрыш
на популярных треках уменьшается. Повторы дают −0.001367 общего PV NDCG
при итоговой разнице −0.001131; из288 дополнительных Recall hits283
приходятся на targets вне окна. Общая пользовательская когорта и смена весов
не объясняют знак. Input-ablation последних двух токенов закрывает60.9%
repeat-разрыва при прежних весах, но не весь разрыв.

Контроль с teacher только CE50 выполнен; его результаты приведены ниже.
[Разбор, срезы, CI и проверки](reports/experiments/sasrec-256-distillation-audit.md),
[план](notes/student-quality-next-experiment.md).

## Контроль источника знаний student

Один новый student с teacher CE50, alpha0/T2/lambda1, той же случайной
инициализацией и training recipe. 4 эпохи, best2, **16.10 мин** на локальном MPS;
готовые CE50 и student со смесью переиспользованы. На PV новый student:
**0.053899 / 0.265966 NDCG@10 / Recall@100** против 0.055093 / 0.252201 у CE50
и 0.053961 / 0.264512 у прошлого student. Recall выше CE50 на 5.46%, общий
NDCG-прирост не подтверждён; отличия от прошлого student по обеим метрикам
имеют CI, включающие 0. Repeat/tail потеря сохраняется без смеси учителя,
исходный CE50 остаётся baseline. 80 тестов и artifact checks; test закрыт.
[Отчёт, срезы, ресурсы и кривые](reports/experiments/student-teacher-control.md).

## Контроль температуры student

При teacher CE50, alpha0/lambda1 изменена только T2 → T1; исходники обучения,
initialization/RNG и остальные настройки сохранены. 4 эпохи, best2, **16.35 мин**
на локальном MPS; у готового T2 те же число эпох и выбранных updates.
На PV T1: **0.055750 / 0.272036 NDCG@10 / Recall@100**.
К T2: **+3.43% / +2.28%**, оба paired CI выше нуля. К CE50:
**+1.19% NDCG** (CI включает ноль), **+7.86% Recall** (CI выше нуля).
Repeat/tail потеря сохраняется; общий выигрыш приходит преимущественно
вне окна истории. 80 тестов и artifact checks; test закрыт, CE50 cache baseline
прежний. [Отчёт и срезы](reports/experiments/student-temperature-control.md),
[протокол и команды](notes/student-temperature-control.md).

## Происхождение материалов

Импорт выполнен 20 октября 2026 из `thesis-project-2026-10-20.zip` и двух заметок, уже находившихся в этой папке. Заметки совпали с копиями в архиве и сохранены в одном экземпляре.

Все 86 файлов внешнего архива учтены в [карте импорта](archive/import-2026-10-20/import-map.json), включая контрольные суммы и новые пути. Три вложенных ZIP сохранены и распакованы: идентичные шаблоны сведены к `thesis/`, прежний комплект литературы находится в `archive/import-2026-10-20/snapshots/`. В актуальных HTML-отчётах исправлены относительные ссылки после переноса файлов.

Оригинальное [описание комплекта](archive/import-2026-10-20/README.txt) и [манифест](archive/import-2026-10-20/MANIFEST.json) сохранены для справки. Исторические скрипты описаны в [scripts/README.md](scripts/README.md).
