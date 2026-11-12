# Парный повтор обучения

12 ноября 2026. Статус: оба запуска завершены по validation patience.
Цель — повторить пользу T1 относительно CE50 на другом seed, сохранив данные
и весь рецепт. Это два новых последовательных MPS FP32 запуска, 4 CPU-потока,
fallback0. Сравнение с готовой парой seed42; без новой настройки по CT/PV.

## Постановка

Датасет и те же256 пользователей, train739571/model_valid29313, OOT,
словарь114125 и target не меняются. Seed43 относится к model initialization,
dropout, train shuffle и dummy negative draws; выборка пользователей не пересобирается.
History50, dim64, 2blocks/heads, dropout0.2, batch256, Adam0.001/(0.9,0.98),
clip5, dummy negatives16, все ID-векторы trainable, patience2 по model_valid
NDCG, без лимита эпох. Random epoch0 не кандидат.

1. Сохранить random `initial.pt` seed43, state/CPU/MPS RNG hashes и guarded
   CE-config. Проверить данные/source/runtime относительно прежнего запуска.
2. Обучить CE50 seed43 с нуля до validation-остановки. Best teacher выбирать
   только по model_valid; после этого weights frozen/eval.
3. Сформировать фиксированный teacher CE43, alpha0/T1/lambda1, проверить
   его validation ranks и подготовить student config. Начальные веса student
   побитово совпадают с сохранённой CE43 random initialization; trained teacher
   weights не являются student initializer. Guard срабатывает до первого update.
4. Обучить student43 с нуля до того же patience. Зафиксировать оба checkpoints
   до CT/PV. Готовые CE42 и student42 T1 не переобучать.
5. На тех же model_valid/CT/PV событиях сравнить четыре модели и paired effects
   student−CE внутри каждого seed. Сохранить cross-seed differences, history/
   popularity срезы, user/day stability, training curves и ресурсы. Test не читать.

CE43 должен получать собственные оценки на saved replay queries. Прежние
CE42 ranks переиспользуются только для того же checkpoint, не как alias CE43.
Teacher alpha0 использует ranks именно своего CE43 контроля.

CE43 остановился после7 эпох, best5, trainer15.59мин;
model_valid0.072448 /0.314229. Учитель зафиксирован по этому checkpoint.
Все29 saved random tensors и CPU/MPS RNG сверены с CE43 и student initializer;
teacher probabilities T1 проверены на32 MPS examples. Learning AST совпадает
с seed42; изменена только логика повторного использования ranks в evaluation.

Student43 остановился после11 эпох, best9, учтённый trainer49.61мин,
model_valid0.076497 /0.331048. Фактические начальные state/CPU/MPS RNG совпали
с CE43. Все параметры T1/alpha0/lambda1 и остальной рецепт сохранены.

## Проверки и пределы вывода

Learning recipe не меняется. Новые expected-state/RNG поля — проверки
инициализации, не warm start. Bootstrap seed42 сохраняется для CI и отделён
от training seed43. User-bootstrap CI условны на фиксированных weights;
два training seeds дают первую проверку устойчивости, не надёжную оценку
общей training variance. PV уже исследовался, test остаётся закрытым.

Student был прерван вместе с чатом на третьем проходе. Возобновление — из
`last.pt` конца epoch2 с optimizer/CPU/MPS/NumPy RNG; незавершённый проход
повторяется. Trainer elapsed не включает отброшенную часть epoch3 и простой.
Исходный и resumed журналы сохраняются отдельно; это не новый вариант обучения.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/prepare_seed_pair.py initial
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/run_sasrec.py train --config configs/experiments/paired-repeat-ce50.json --output data/runs/paired-seed-repeat/ce50/train-mps --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/prepare_seed_pair.py teacher
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/check_sasrec_initialization.py --config configs/experiments/paired-repeat-student.json --output data/runs/paired-seed-repeat/student --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py train --weight 1 --config configs/experiments/paired-repeat-student.json --output data/runs/paired-seed-repeat/student --device mps
# При прерывании этого же запуска добавить --resume.
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/distill_sasrec.py evaluate --config configs/experiments/paired-repeat-student.json --output data/runs/paired-seed-repeat/student --device mps
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/compare_seed_pairs.py --device mps
```

Команды предназначены для чистого output; существующие runs не перезаписываются.
Checkpoints и per-event arrays остаются в `data/`, компактные доказательства,
JSON/CSV/графики и отчёт публикуются в Git.

Итог PV: CE43 0.054199 /0.258271 →student43 0.057996 /0.271095;
+7.01% NDCG /+4.97% Recall, оба paired CI выше0. Recall-прирост повторился
в обоих seeds; tail-repeat NDCG потеря сохраняется.
[Отчёт и артефакты](../reports/experiments/paired-seed-repeat.md).
