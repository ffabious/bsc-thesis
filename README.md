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

## Происхождение материалов

Импорт выполнен 20 октября 2026 из `thesis-project-2026-10-20.zip` и двух заметок, уже находившихся в этой папке. Заметки совпали с копиями в архиве и сохранены в одном экземпляре.

Все 86 файлов внешнего архива учтены в [карте импорта](archive/import-2026-10-20/import-map.json), включая контрольные суммы и новые пути. Три вложенных ZIP сохранены и распакованы: идентичные шаблоны сведены к `thesis/`, прежний комплект литературы находится в `archive/import-2026-10-20/snapshots/`. В актуальных HTML-отчётах исправлены относительные ссылки после переноса файлов.

Оригинальное [описание комплекта](archive/import-2026-10-20/README.txt) и [манифест](archive/import-2026-10-20/MANIFEST.json) сохранены для справки. Исторические скрипты описаны в [scripts/README.md](scripts/README.md).
