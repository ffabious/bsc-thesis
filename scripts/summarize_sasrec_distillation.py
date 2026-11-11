#!/usr/bin/env python3
"""Export the fixed distillation comparison and its validation/serving plots."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from distill_sasrec import label,validation
from plot_sasrec_replay import plt,save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args();root,out = args.input,args.output;out.mkdir(parents=True,exist_ok=True)
    read = lambda p:json.loads(p.read_text())
    teacher = read(root/"teacher/selection.json");selection = read(root/"selection.json");evaluation = read(root/"evaluation.json")
    runs = {r["weight"]:read(root/label(r["weight"])/"train-mps/training.json") for r in selection["candidates"]}
    profiles = {dev:read(root/f"serving-{dev}.json") for dev in ["cpu","mps"]}
    for name,source in [("teacher-selection.json",root/"teacher/selection.json"),("teacher-check.json",root/"teacher/implementation-check.json"),
                         ("selection.json",root/"selection.json"),("evaluation.json",root/"evaluation.json"),("tests.txt",root/"tests.txt")]:
        shutil.copyfile(source,out/name)
    for weight,r in runs.items():(out/f"training-{label(weight)}.json").write_text(json.dumps(r,indent=2)+"\n")
    for dev in profiles:shutil.copyfile(root/f"serving-{dev}.json",out/f"serving-{dev}.json")
    if (root/"tests-before-training.txt").exists():shutil.copyfile(root/"tests-before-training.txt",out/"tests-before-training.txt")
    if (root/"source-history.json").exists():
        shutil.copyfile(root/"source-history.json",out/"source-history.json")
        (out/"training-source").mkdir(exist_ok=True)
        for source in (root/"training-source").glob("*.py"):shutil.copyfile(source,out/"training-source"/source.name)
    rows=[]
    for weight,r in runs.items():
        for epoch in [{"epoch":0,"validation":r["initial_validation"]}]+r["epochs"]:
            components=epoch.get("loss_components",{})
            rows.append({"weight":weight,"epoch":epoch["epoch"],"best":epoch["epoch"]==r["best_epoch"],
                **epoch["validation"],"ce":components.get("ce",epoch.get("loss")),"kl_t2":components.get("kl_t2"),
                "total_loss":epoch.get("loss"),"train_seconds":epoch.get("train_seconds"),
                "validation_seconds":epoch.get("validation_seconds"),"peak_rss_mib":epoch.get("process_peak_rss_mib")})
    quality=[{"split":split,"model":name,**v} for split,b in evaluation["results"].items() for name,v in b["metrics"].items()]
    serving=[{"device":dev,"batch_size":int(batch),"model":name,"median_ms_per_request":v["median_ms_per_request"],
              "min_repeat":min(v["repeat_mean_ms_per_request"]),"max_repeat":max(v["repeat_mean_ms_per_request"])}
             for dev,p in profiles.items() for batch,cases in p["measurements"].items() for name,v in cases.items()]
    for name,data in [("training.csv",rows),("metrics.csv",quality),("serving.csv",serving)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator="\n");writer.writeheader();writer.writerows(data)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    colors=["#c36b22","#26906b","#8653a6"]
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    teacher_metrics=next(p["metrics"] for p in teacher["curve"] if p["alpha"]==teacher["alpha"])
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        for (weight,r),color in zip(runs.items(),colors):
            values=[r["initial_validation"]]+[e["validation"] for e in r["epochs"]]
            ax.plot(range(len(values)),[v[metric] for v in values],"o-",color=color,label="CE control" if weight==0 else f"KD weight {weight:g}",markersize=4)
            ax.scatter([r["best_epoch"]],[validation(r)[metric]],s=85,facecolors="none",edgecolors=color,zorder=4)
        ax.axhline(teacher_metrics[metric],linestyle=":",color="#555555",label="Selected teacher")
        ax.set(xlabel="Additional epoch (0 = original checkpoint)",ylabel=metric,title="model_valid")
        ax.set_xticks(range(max(len(r["epochs"]) for r in runs.values())+1))
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,"validation")
    fig,axes=plt.subplots(2,2,figsize=(10,6.7))
    names=["original","control","student","teacher"]
    for ax,(dev,batch) in zip(axes.flat,[("cpu","1"),("cpu","256"),("mps","1"),("mps","256")]):
        cases=profiles[dev]["measurements"][batch];values=[cases[n]["median_ms_per_request"] for n in names]
        ax.bar(names,values,color=["#888888",colors[0],colors[1],colors[2]],zorder=3)
        ax.errorbar(names,values,yerr=[[v-min(cases[n]["repeat_mean_ms_per_request"]) for n,v in zip(names,values)],
                    [max(cases[n]["repeat_mean_ms_per_request"])-v for n,v in zip(names,values)]],fmt="none",ecolor="#333333",capsize=3)
        ax.set(title=f"{dev.upper()} — B={batch}",ylabel="Full serving ms/request")
    fig.text(.5,.01,"B=256: batch duration / 256, not individual latency. Bars: repeat cost range, not confidence intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,out,"serving")
    pv=evaluation["results"]["policy_valid"];delta=pv["student_minus"]["control"]
    summary={"teacher_alpha":teacher["alpha"],"distillation_weight":selection["student"]["weight"],"temperature":2,
        "training":{label(w):{"epochs":len(r["epochs"]),"best_epoch":r["best_epoch"],"validation":validation(r),
            "last_validation":r["epochs"][-1]["validation"],
            "elapsed_seconds":r["elapsed_seconds"],"total_run_seconds":r["total_run_seconds"],
            "peak_rss_mib":max(e["process_peak_rss_mib"] for e in r["epochs"]),
            "maximum_saved_mps_driver_mib":max(e["mps_driver_mib"] for e in r["epochs"])} for w,r in runs.items()},
        "validation_acceptance_guard":selection["student"]["metrics"]["ndcg@10"]>selection["control"]["metrics"]["ndcg@10"] and
            selection["student"]["metrics"]["recall@100"]>=selection["control"]["metrics"]["recall@100"],
        "policy_valid":pv["metrics"],"student_minus_control":delta,"serving":serving,"no_test_events_read":True}
    all_initial=all(r["best_epoch"]==0 for r in runs.values())
    summary["all_best_checkpoints_retain_initial_weights"]=all_initial
    summary["total_additional_training_minutes"]=sum(r["total_run_seconds"] for r in runs.values())/60
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    train_table="\n".join(f"| {label(w)} | {len(r['epochs'])} | {r['best_epoch']} | {validation(r)['ndcg@10']:.6f} | {validation(r)['recall@100']:.6f} | {r['total_run_seconds']/60:.2f} | {max(e['process_peak_rss_mib'] for e in r['epochs']):.1f} |" for w,r in runs.items())
    quality_table="\n".join(f"| {split} | {name} | {m['ndcg@10']:.6f} | {m['recall@100']:.6f} | {m['hit@1']:.6f} |" for split,b in evaluation["results"].items() for name,m in b["metrics"].items())
    cost_table="\n".join('| '+name+' | '+' | '.join(f"{profiles[dev]['measurements'][batch][name]['median_ms_per_request']:.4f}" for batch in ['1','256'] for dev in ['cpu','mps'])+' |' for name in names)
    memory_table="\n".join(f"| {label(w)} | {max(e['mps_tensor_mib'] for e in r['epochs']):.1f} | {max(e['mps_driver_mib'] for e in r['epochs']):.1f} |" for w,r in runs.items())
    ci_table="\n".join(f"| {metric} | {d['difference']:+.6f} | [{d['cluster_bootstrap_95ci'][0]:+.6f}; {d['cluster_bootstrap_95ci'][1]:+.6f}] |" for metric,d in delta.items())
    outcome=(f"**В проверенной постановке дистилляция не улучшила одиночный baseline.**\n"
        f"Control и оба KD-запуска остановились после двух дополнительных эпох;\n"
        f"по validation NDCG лучшими во всех случаях остались исходные веса (epoch0).\n"
        f"Формальный выбор weight{selection['student']['weight']:g} — разрешение ничьей двух epoch0 candidates,\n"
        f"а не свидетельство лучшего KL coefficient. Веса выбранных checkpoints побитово\n"
        f"совпадают с исходным CE50, поэтому их CT/PV качество одинаково.") if all_initial else (
        f"Выбран weight{selection['student']['weight']:g}, best epoch{selection['student']['best_epoch']};\n"
        f"validation acceptance guard: {summary['validation_acceptance_guard']}.\n"
        f"Независимое от этого выбора CT/PV сравнение приведено ниже.")
    last_weight=max(runs);last=runs[last_weight]["epochs"][-1]["validation"];initial=runs[last_weight]["initial_validation"]
    report=f'''# Дистилляция CE50+CE2 в одиночный SASRec50

11 ноября 2026. Пилот 256, локальные Apple M3 Pro / MPS FP32, 4 CPU-потока,
fallback отключён. Учитель и initializer готовы из прежних запусков;
текущий эксперимент не повторяет их pretraining.
[Протокол и команды](../../notes/sasrec-256-distillation.md),
[сводка](sasrec-256-distillation/summary.json).

{outcome}

## Протокол

Учитель — frozen CE50 epoch4 + CE2 epoch7. Alpha **{teacher['alpha']:.1f}** выбран
по model_valid NDCG из 0/0.1/…/1, tie — меньший alpha. Model_valid учителя:
**{teacher_metrics['ndcg@10']:.6f} / {teacher_metrics['recall@100']:.6f}**.
Native recent2 сохраняет learned positions48/49; scaling soft targets проверен
на всех 29 313 validation targets и сохраняет выбранные метрики.
[Teacher selection](sasrec-256-distillation/teacher-selection.json),
[implementation check](sasrec-256-distillation/teacher-check.json).

Teacher logits = z-score mixture × population std исходных long logits.
Это положительное affine scaling, сохраняющее ranking и возвращающее logits
в шкалу CE50; вероятности прослушивания не калибруются. T=2 фиксирована.
Loss = CE(student raw logits, target) + weight × T² × KL(teacher || student),
полный каталог **114 125** items. Вес CE остаётся 1 во всех вариантах.
KL суммируется по классам и усредняется по batch (`batchmean`), не по B×catalog.
Учитель eval/frozen, без current feedback или градиентов; dense soft targets
вычисляются на лету, не сохраняются на диск.

Control weight0 и students weight0.25/1 начинают из одинакового CE50 checkpoint,
Adam новый. Прежние train **739 571** / model_valid **29 313** prefixes,
history50, dim64, 2 blocks/heads, dropout0.2, batch256, lr0.001,
seed42, dummy negatives16 и patience2 сохранены. Все ID-векторы student trainable.
Лимита эпох нет, epoch0 участвует в best selection. Inference student — обычный
SASRec50, **7 357 888 параметров**, без teacher modules или второго scoring.

## Обучение и выбор

| Запуск | Эпох | Best дополнительная эпоха | model_valid NDCG | Recall | Полный запуск, мин | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{train_table}

Best epoch0 означает сохранение исходных весов после неудачного дообучения.
Nonzero weight **{selection['student']['weight']:g}** выбран только по best
model_valid NDCG, tie — меньший weight; после этого выбор зафиксирован до
чтения CT/PV. Критерий принятия: NDCG выше контроля, Recall не ниже.
[Locked selection](sasrec-256-distillation/selection.json),
[все эпохи и CE/KL components](sasrec-256-distillation/training.csv).

У последней фактически обученной модели weight{last_weight:g} validation Recall
**{last['recall@100']:.6f}** ({100*(last['recall@100']/initial['recall@100']-1):+.1f}% к initializer), NDCG
**{last['ndcg@10']:.6f}** ({100*(last['ndcg@10']/initial['ndcg@10']-1):+.1f}%).
Это описание model_valid; дополнительные unselected weights на CT/PV
не подбирались и не оценивались.

![Validation curves](sasrec-256-distillation/validation.png)

Полное время включает setup, сбор данных, обучение, validation, checkpoints
и финальные проверки. В raw training JSON отдельно `setup_seconds`,
`elapsed_seconds` trainer и train time каждой эпохи. Peak RSS — процесс;
MPS tensor/driver snapshots сохранены по эпохам и не суммируются с RSS.

| Training run | Max saved MPS tensors, MiB | Max saved MPS driver, MiB |
| --- | ---: | ---: |
{memory_table}

Это snapshots после эпох и allocator residency, не гарантированный пик
временных buffers внутри forward/backward kernels.

## Качество на фиксированных CT/PV событиях

Использованы прежние 52 191 CT / 23 394 PV targets и одинаковый полный каталог.
Original ranks переиспользованы из проверенного fresh replay; control/student/
teacher оценены на тех же строго прошлых histories. Teacher использует alpha
с model_valid, а не прежний CT selection. Test не читается.

| Split | Модель | NDCG@10 | Recall@100 | Hit@1 |
| --- | --- | ---: | ---: | ---: |
{quality_table}

Student минус continued control на PV, 2 000 bootstrap целых пользователей,
seed42, event-weighted, fixed checkpoints. Интервалы pointwise, без поправки
за множественные сравнения; training variance не включена.

| Метрика | Разница | 95% CI |
| --- | ---: | --- |
{ci_table}

[Все paired CI и repeat/window strata](sasrec-256-distillation/evaluation.json),
[metrics CSV](sasrec-256-distillation/metrics.csv).

## Полная стоимость inference

Original/control/student выполняют один encoder50 и один full-catalog scoring.
Teacher использует нужные ветви, native recent2, per-query z-score/mix и один
final top-100. Передача token IDs и результата на CPU включены; input уже
токенизирован, сеть/storage/queues исключены. Модели чередуются в одном процессе.
B=1: 5×128 запросов, warmup16; B=256: 3×8 batch, warmup2; MPS sync.

Медианы серий, **мс/запрос**. B=256 — throughput (batch duration/256),
не latency одного online запроса.

| Модель | CPU B=1 | MPS B=1 | CPU B=256 | MPS B=256 |
| --- | ---: | ---: | ---: | ---: |
{cost_table}

![Serving cost](sasrec-256-distillation/serving.png)

[Все повторы и memory probes CPU](sasrec-256-distillation/serving-cpu.json),
[MPS](sasrec-256-distillation/serving-mps.json),
[serving CSV](sasrec-256-distillation/serving.csv).
Малые timing differences одиночных моделей не означают изменение сложности:
архитектура и число параметров одинаковы. Profiling process держит все четыре
модели для чередования; его RSS не является памятью standalone student.
MPS memory stage snapshots не гарантируют захват внутренних kernel peaks.

## Проверки и ограничения

77 CPU/MPS тестов: KL direction/gradients, frozen teacher, native2, warm start,
epoch0 fallback, default trainer/resume. Проверяются SHA checkpoints, train/
model_valid files, protocol и source; одинаковые initial CPU/MPS RNG states.
Teacher и weight selection выполняются на model_valid, модельные checkpoint
hashes фиксируются до CT/PV. Test events не читались.

После обучения исправлен экспорт относительного output path и уточнено имя
data-read metadata flag. Формулы и веса не менялись; фактически выполненный
training source сохранён с исходным SHA в
[source history](sasrec-256-distillation/source-history.json).

Один seed и небольшой pilot; это проверка конкретного KD loss/температуры/
двух nonzero weights, а не общий предел дистилляции. Текущие cost и quality
относятся к fresh inference; cache replay для новых весов не выполнялся.
[Проверки артефактов](sasrec-256-distillation/verification.json).
'''
    out.with_suffix('.md').write_text(report)
    print(json.dumps({"alpha":teacher['alpha'],"weight":selection['student']['weight'],"policy_valid":pv['metrics']},indent=2))


if __name__=="__main__":main()
