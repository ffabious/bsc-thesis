# H8: содержимое новых действий как сигнал refresh

14 ноября 2026. Первая frozen user-holdout серия опубликована в ddb4f39.
Её users/outcomes не используются для выбора этого варианта; дальнейшая
подтверждающая оценка потребует новой user cohort.

Причина проверки: исходные8 признаков описывают возраст и активность,
но не различают повтор последнего трека и появление другого. Это пробел
исходной optional ветки H2. Проверяем один компактный content-aware вариант,
не начинаем перебор архитектур/семантических признаков.

Добавляются4 строго прошлых поля: latest item совпадает с cached latest;
latest есть в cached history50; cosine frozen ID embeddings latest/cached
latest; log1p(train count latest). Это trained ID table CE1k, не audio features.
Ни feedback текущего target, ни новые encoder outputs не используются.
Веса encoder/item table заморожены; normalized table и log counts — static
CPU setup, общий для всех пользователей. Их размер публикуется отдельно.

CE1k dataset/grid106/top100 references — готовые. MLP12→32→32→1,
1505 параметров, seed42, прежние optimizer/loss/request weights/CT8/2/4/
early stopping. Порог выбора не меняется. Референс — готовый8-feature
MLP42 CE1k на тех же labels/ранних validation contexts.

**Gate до дорогого replay:** weighted early-validation MSE должна снизиться
хотя бы на1% относительно повторно вычисленного MSE frozen8-feature model.
Если нет — сохранить probe/curve и закончить вариант без нового PV replay.
Если да — выполнить прежний CT-only threshold selection, closed-loop PV
и полную CPU-feature/prediction+MPS cost calibration. Никакой настройки по
ранее открытому user-holdout. Новая metadata отмечает gate outcome.

Отдельные repeated CPU rollout timings отключены: в текущем драйвере они
уже не участвуют в primary cost, потому что CPU features/prediction входят
в complete coupled serving. Сами trajectories считаются один раз; branch
и mixed serving benchmarks сохраняют3 repeats. Это устраняет лишние полные
обходы343738 requests без изменения решений, обучения и основного cost scope.

Tests проверяют scalar/matrix feature agreement, repeat/novelty/cosine
значения и отсутствие зависимости от будущих queries/labels. Source/checkpoint/
item-count hashes и frozen encoder guard обязательны. Если development
результат не оправдает дальнейшую проверку, ветку закрываем; старый holdout
не используется повторно ни для подтверждения, ни для выбора.
