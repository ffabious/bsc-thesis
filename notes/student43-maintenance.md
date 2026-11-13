# Student43: политики и контроллер refresh

13 ноября 2026. Согласованный сетап дополнен обучением контроллера и его
сравнением с простыми политиками. Все вычисления локальны; энкодер frozen.

## Основной frontier

Student43 T1/alpha0/lambda1, best9, checkpoint `a6549493…`, history50,
dim64, 2blocks/heads. Полный каталог114125 без seen-item exclusion.
Готовые100 расписаний CE42 переносятся с проверкой hashes, всех requests,
targets, refs и refresh flags. Чужие ranks и branch timings не переносятся.
Качество: все52191 CT/23394 PV targets; расходы: все62129/29914 requests.
Инициализации232/216 оплачиваются, кэш сбрасывается отдельно в CT/PV.
Current timestamp group не видна при решении; в простое refresh нет.

Время5/30мин,1/2/3/6/12/24ч; события2/5/10/20/25/50/60/120/240/480;
always/never и80 OR-гибридов. Основной NDCG/cost frontier, Recall и Hit1
дополнительно. Budgets0.6/0.75/0.9 стоимости always, выбор по полному CT,
PV без перенастройки. Недостижимые бюджеты отмечаются явно.

Стоимость student измеряется заново: MPS FP32 B=1, локальный CPU-cache,
full catalog scoring/top100/CPU result, refresh/write/serve либо hit/serve,
плюс CPU decision.128 запросов×3 повтора, warmup8; mixed workload checks.
Offline scoring B=256, 4 CPU-потока; это не serving latency B=256.
CE42 сравнивается по качеству на тех же политиках, его старые timings
не выдаются за парный текущий performance benchmark.

## Контроллер

Один weighted ridge regressor, lambda0.001, CPU.8 признаков: log1p числа
новых событий и возраста cache; log1p активности1ч/24ч, последней паузы и
длины истории; два взаимодействия возраста/новых событий с активностью24ч.
Все значения строго прошлые. История времён строится из уже готовых request
групп и только нужного суточного prefix из model_valid.

Train — первые10 дней CT, tuning — последние4. Датасет — разные stale
состояния существующих100 политик до cutoff. Повторяющиеся(request,snapshot)
убираются, суммарный вес каждого запроса равен1. Запросы без positive target
сохраняются с gain0. Label — сумма `fresh NDCG−cached NDCG` по positive targets
текущей группы; отрицательные gains сохраняются. Это myopic immediate-benefit
модель, не оценка всей будущей пользы refresh.

Пороги — weighted fit-prediction quantiles0/.1/.25/.5/.75/.9/.95/.99/1;
сетка фиксируется без PV. Каждый порог replay на всём CT с собственным
состоянием кэша. Качество/compute для выбора берутся только на последних4 днях,
кэш на внутренней границе не сбрасывается. Общие CT/PV границы прежние.
Контроллер и простые baseline для этого сравнения выбираются на одном tuning
интервале. Simple comparator имеет tuning cost не выше cost выбранного
контроллера; фактические PV costs публикуются отдельно, точное равенство
при дискретных политиках и изменении нагрузки не предполагается.

После selection.json исполняются только выбранные threshold на PV.
Refit на полном CT не делается. Решения принимаются на всех requests,
включая nonpositive; initial refresh обязательный. Затраты CPU включают
scalar history searches, feature extraction, prediction и state bookkeeping.
Online history/network loading исключены, как и в основном benchmark.
Итоговые CPU+MPS ветви измеряются совместно с перемешанным порядком.
Отдельная CPU цифра сохраняется как диагностика и повторно не начисляется.
Первичная аддитивная оценка оказалась нестабильной (mixed error до46.22%);
калибровка изменена только по CT timings, при тех же весах и пороговой сетке.
Итоговый maximum mixed error2.12%.

CI:2000 whole-user bootstrap, event-weighted, seed42, fixed weights.
Политики не выбираются по PV; итоговый test закрыт.

## Команды

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python -m unittest tests.test_policies tests.test_replay tests.test_refresh_controller -v
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/python scripts/replay_sasrec_policies.py --config configs/experiments/student43-policies.json --output data/runs/student43-policies --device mps
caffeinate -i env PYTORCH_ENABLE_MPS_FALLBACK=0 VECLIB_MAXIMUM_THREADS=4 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv/bin/python scripts/train_refresh_controller.py --output data/runs/student43-controller-calibrated --device mps
.venv/bin/python scripts/plot_sasrec_policies.py --input data/runs/student43-controller-calibrated/policy-results.json --output reports/experiments/student43-maintenance
.venv/bin/python scripts/report_student_maintenance.py
```

Configs, модели контроллера, метрики, выбор и отчёты публикуются в Git;
raw arrays и checkpoints энкодера остаются в локальном `data/`.

## Итог и дополнительный cost-matched reference

Student43 frontier: CT-selected simple policies50events/1h,5events/30m,
2events/5m. PV costs59.94%/66.94%/82.09% always; NDCG0.033800/0.044054/0.053455.

Контроллер q90/q50/q25: PV costs58.48%/73.18%/84.84%,
NDCG0.031648/0.048884/0.055476. Две NDCG-точки добавлены к дискретному
frontier, Recall ниже simple comparator. Из-за cost gaps дополнительно
на CT tuning выбраны смеси двух готовых политик при стоимости контроллера.
Пара и вероятность фиксированы для PV; считаются ожидаемые quality/cost,
а не новая randomized serving implementation. Assignment overhead и variance
не измеряются. Все три NDCG CI controller−expectation включают0.
23теста и824 независимых paired CI прошли; test закрыт.
[Отчёт, артефакты и выводы](../reports/experiments/student43-maintenance.md).
