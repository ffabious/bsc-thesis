#!/usr/bin/env python3
"""Export the fixed cold-start student comparison with its reused CE control."""

import argparse
import csv
import json
import shutil
from pathlib import Path

from distill_sasrec import validation
from plot_sasrec_replay import plt,save
from run_sasrec import ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();root,out=args.input,args.output;out.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads(p.read_text())
    protocol=read(args.config);audit=read(root/"initialization.json");selection=read(root/"selection.json")
    teacher=read(root/"teacher/selection.json");evaluation=read(root/"evaluation.json")
    runs={name:read(ROOT/selection[name]["run"]/"training.json") for name in ["control","student"]}
    profiles={device:read(root/f"serving-{device}.json") for device in ["cpu","mps"]}
    teacher_metrics=next(p["metrics"] for p in teacher["curve"] if p["alpha"]==teacher["alpha"])
    for destination,source in [("protocol.json",args.config),("initialization.json",root/"initialization.json"),
            ("teacher-selection.json",root/"teacher/selection.json"),("selection.json",root/"selection.json"),
            ("evaluation.json",root/"evaluation.json"),("tests.txt",root/"tests.txt")]:
        shutil.copyfile(source,out/destination)
    shutil.copytree(root/"initialization-reference",out/"initialization-reference",dirs_exist_ok=True)
    for name,r in runs.items():(out/f"training-{name}.json").write_text(json.dumps(r,indent=2)+"\n")
    for device in profiles:shutil.copyfile(root/f"serving-{device}.json",out/f"serving-{device}.json")
    for source in root.glob("*.log"):shutil.copyfile(source,out/source.name)
    names=["control","student","teacher"];colors=["#c36b22","#26906b","#8653a6"]
    curves=[]
    for name,r in runs.items():
        for epoch in [{"epoch":0,"validation":r["initial_validation"]}]+r["epochs"]:
            components=epoch.get("loss_components",{})
            curves.append({"model":name,"epoch":epoch["epoch"],"best":epoch["epoch"]==r["best_epoch"],
                **epoch["validation"],"ce":components.get("ce",epoch.get("loss")),"kl_t2":components.get("kl_t2"),
                "total_loss":epoch.get("loss"),"train_seconds":epoch.get("train_seconds"),
                "validation_seconds":epoch.get("validation_seconds"),"peak_rss_mib":epoch.get("process_peak_rss_mib")})
    quality=[{"split":split,"model":name,**b["metrics"][name]} for split,b in evaluation["results"].items() for name in names]
    serving=[{"device":dev,"batch_size":int(batch),"model":name,"median_ms_per_request":v["median_ms_per_request"],
        "min_repeat":min(v["repeat_mean_ms_per_request"]),"max_repeat":max(v["repeat_mean_ms_per_request"])}
        for dev,p in profiles.items() for batch,cases in p["measurements"].items() for name,v in cases.items()]
    for name,rows in [("training.csv",curves),("metrics.csv",quality),("serving.csv",serving)]:
        with (out/name).open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,"grid.alpha":.2})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for ax,metric in zip(axes,["ndcg@10","recall@100"]):
        for (name,r),color in zip(runs.items(),colors):
            values=[r["initial_validation"]]+[e["validation"] for e in r["epochs"]]
            ax.plot(range(len(values)),[v[metric] for v in values],"o-",color=color,label="CE50 (reused)" if name=="control" else "Student from scratch",markersize=4)
            ax.scatter([r["best_epoch"]],[validation(r)[metric]],s=85,facecolors="none",edgecolors=color,zorder=4)
        ax.axhline(teacher_metrics[metric],linestyle=":",color=colors[2],label="Fixed teacher")
        ax.set(xlabel="Epoch (0 = same random initialization)",ylabel=metric,title="model_valid")
        ax.set_xticks(range(max(len(r["epochs"]) for r in runs.values())+1))
    axes[0].legend(fontsize=8);fig.tight_layout();save(fig,out,"validation")
    fig,axes=plt.subplots(2,2,figsize=(10,6.7))
    for ax,(dev,batch) in zip(axes.flat,[("cpu","1"),("cpu","256"),("mps","1"),("mps","256")]):
        cases=profiles[dev]["measurements"][batch];values=[cases[n]["median_ms_per_request"] for n in names]
        ax.bar(names,values,color=colors,zorder=3)
        ax.errorbar(names,values,yerr=[[v-min(cases[n]["repeat_mean_ms_per_request"]) for n,v in zip(names,values)],
            [max(cases[n]["repeat_mean_ms_per_request"])-v for n,v in zip(names,values)]],fmt="none",ecolor="#333333",capsize=3)
        ax.set(title=f"{dev.upper()} — B={batch}",ylabel="Full serving ms/request")
    fig.text(.5,.01,"B=256: batch duration / 256. Bars: repeat range, not confidence intervals.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1));save(fig,out,"serving")
    training={name:{"reused":name=="control","epochs":len(r["epochs"]),"best_epoch":r["best_epoch"],
        "validation":validation(r),"elapsed_seconds":r["elapsed_seconds"],"full_run_seconds":r.get("total_run_seconds"),
        "train_seconds":sum(e["train_seconds"] for e in r["epochs"]),
        "peak_rss_mib":max(e["process_peak_rss_mib"] for e in r["epochs"]),
        "max_saved_mps_tensor_mib":max(e["mps_tensor_mib"] for e in r["epochs"]),
        "max_saved_mps_driver_mib":max(e["mps_driver_mib"] for e in r["epochs"])} for name,r in runs.items()}
    pv=evaluation["results"]["policy_valid"];delta=pv["student_minus"]["control"]
    guard=validation(runs["student"])["ndcg@10"]>validation(runs["control"])["ndcg@10"] and validation(runs["student"])["recall@100"]>=validation(runs["control"])["recall@100"]
    summary={"teacher_alpha":teacher["alpha"],"temperature":protocol["temperature"],"weight":selection["student"]["weight"],
        "initialization_verified":audit["same_all_tensors"],"parameters_single":audit["parameters"],"training":training,
        "validation_acceptance_guard":guard,"policy_valid":{name:pv["metrics"][name] for name in names},
        "student_minus_control":delta,"serving":serving,"new_training_runs":1,"no_test_events_read":True}
    summary["policy_valid_supported_improvements"]={metric:d["cluster_bootstrap_95ci"][0]>0 for metric,d in delta.items()}
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    train_table="\n".join(f"| {name} | {t['epochs']} | {t['best_epoch']} | {t['validation']['ndcg@10']:.6f} | {t['validation']['recall@100']:.6f} | {t['elapsed_seconds']/60:.2f} | {t['peak_rss_mib']:.1f} |" for name,t in training.items())
    quality_table="\n".join(f"| {row['split']} | {row['model']} | {row['ndcg@10']:.6f} | {row['recall@100']:.6f} | {row['hit@1']:.6f} |" for row in quality)
    cost_table="\n".join('| '+name+' | '+' | '.join(f"{profiles[dev]['measurements'][batch][name]['median_ms_per_request']:.4f}" for batch in ['1','256'] for dev in ['cpu','mps'])+' |' for name in names)
    ci_table="\n".join(f"| {metric} | {d['difference']:+.6f} | [{d['cluster_bootstrap_95ci'][0]:+.6f}; {d['cluster_bootstrap_95ci'][1]:+.6f}] |" for metric,d in delta.items())
    memory_table="\n".join(f"| {name} | {t['max_saved_mps_tensor_mib']:.1f} | {t['max_saved_mps_driver_mib']:.1f} |" for name,t in training.items())
    relative={metric:100*(pv['metrics']['student'][metric]/pv['metrics']['control'][metric]-1) for metric in ['ndcg@10','recall@100']}
    evidence="; ".join(f"{metric}: {'прирост подтверждён' if d['cluster_bootstrap_95ci'][0]>0 else 'прирост не подтверждён'}" for metric,d in delta.items())
    report=f'''# Student с нуля: фиксированный CE50+CE2 teacher

11 ноября 2026. Пилот 256, Apple M3 Pro / MPS FP32, 4 CPU-потока, fallback отключён.
Выполнен **один новый training run**; готовые CE50 и CE2 не переобучались.
[Протокол и команды](../../notes/sasrec-256-distillation-cold.md),
[машинная сводка](sasrec-256-distillation-cold/summary.json).

Student выбран по model_valid на эпохе **{training['student']['best_epoch']}**,
остановка после **{training['student']['epochs']}** эпох по patience2.
На policy_valid изменение к готовому CE50: NDCG **{relative['ndcg@10']:+.2f}%**,
Recall **{relative['recall@100']:+.2f}%**. Проверка model_valid «NDCG выше,
Recall не ниже»: **{guard}**. Это один seed и одна фиксированная KD-постановка.
На отдельном policy_valid по парным пользовательским интервалам: **{evidence}**.
Validation guard описывает model_valid и не заменяет проверку переноса.

## Инициализация и обучение

Исходный initial.pt CE50 не сохранялся. Его начальное состояние восстановлено
из source commit591575a, исходных config и seed42. SHA archived source совпадают
с реестром исходного эксперимента; AST constructors, sampling и CE update
проверены на совпадение. Все **{audit['tensors']} тензоров** восстановленной CE50
и нового student совпадают побитово, CPU RNG совпадает с сохранённым CE last.pt.
На **{audit['validation_examples']:,}** начальных validation targets ранги совпадают
точно и воспроизводят метрики исходного лога. В реальном training loop перед
первым update дополнительно проверены SHA весов и CPU/MPS RNG.
[Проверка и границы реконструкции](sasrec-256-distillation-cold/initialization.json).

Это случайная инициализация, без загрузки обученных CE50 weights; epoch0
не участвует в выборе, как у контроля. Все ID-векторы student trainable.
History50, dim64, 2 blocks/heads, dropout0.2, batch256, Adam lr0.001,
betas(0.9,0.98), clip5, seed42, dummy negatives16 сохранены.
Train **739 571** prefixes, model_valid **29 313**, полный каталог **114 125**.
Одиночный student: **{audit['parameters']:,} параметров**.

Teacher — frozen/eval CE50 epoch4 + native CE2 epoch7 (positions48/49).
Alpha короткой ветви **{teacher['alpha']:.1f}**, T **{protocol['temperature']:g}**, lambda **{selection['student']['weight']:g}** фиксированы;
alpha переиспользована из прежнего model_valid selection. Никаких новых grids.
Teacher logits = per-query z-score mixture × population std исходных long logits.
Loss = CE + lambda × T² × KL(teacher || student), `batchmean`; KL суммируется
по каталогу, не усредняется по B×items. Targets вычисляются на лету,
teacher gradients отсутствуют. Student inference не использует teacher.

| Запуск | Эпох | Best | model_valid NDCG@10 | Recall@100 | Trainer, мин | Peak RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{train_table}

Время и память контроля взяты из **исторического** запуска, не из нового парного
training benchmark. Student full run с setup/проверками:
**{training['student']['full_run_seconds']/60:.2f} мин**;
внутри trainer time включает train, validation и checkpoint I/O.
Исторический контроль такого outer-таймера не сохранял.

| Запуск | Max saved MPS tensors, MiB | Max saved MPS driver, MiB |
| --- | ---: | ---: |
{memory_table}

RSS — максимум процесса. MPS — snapshots после эпох, не гарантированный
внутренний пик kernels; это общая unified memory, значения не суммируются.

![Validation](sasrec-256-distillation-cold/validation.png)

[Все эпохи, CE и T²KL](sasrec-256-distillation-cold/training.csv).

## Fresh качество на прежних CT/PV событиях

Best checkpoint зафиксирован до чтения CT/PV. Прежние 52 191 CT / 23 394 PV
targets, полные одинаковые candidate sets, строго прошлые history перед
timestamp group; без исключения уже прослушанных items. Контрольные ranks
переиспользованы из проверенного replay, student/teacher рассчитаны заново.
Test не читается. Выбора по CT/PV нет.

| Split | Модель | NDCG@10 | Recall@100 | Hit@1 |
| --- | --- | ---: | ---: | ---: |
{quality_table}

Student минус CE50 на PV: 2 000 bootstrap целых пользователей, seed42,
event-weighted. Интервалы pointwise при фиксированных checkpoints;
training variance и поправки на множественные проверки не включены.

| Метрика | Разница | 95% CI |
| --- | ---: | --- |
{ci_table}

[Полные paired CI и last-item/in-window/outside strata](sasrec-256-distillation-cold/evaluation.json).

На PV student хуже распознаёт повторы последнего item: NDCG **{pv['strata']['last_item']['metrics']['student']['ndcg@10']:.4f}**
против **{pv['strata']['last_item']['metrics']['control']['ndcg@10']:.4f}** у CE50
({pv['strata']['last_item']['targets']} targets). На items вне текущего окна
Recall **{pv['strata']['outside_window']['metrics']['student']['recall@100']:.4f}** против
**{pv['strata']['outside_window']['metrics']['control']['recall@100']:.4f}**.
Это описательные срезы фиксированных моделей, не доказательство причины.

## Стоимость fresh serving

Обе одиночные модели: encoder50 + full-catalog scoring + final top100.
Teacher: две ветви, native recent2, z-score/mix и один top100.
Включены подготовка/передача token IDs и результат на CPU; input уже
токенизирован, без сети, storage и queues. Веса моделей чередуются в одном
процессе. B1: 5×128, warmup16; B256: 3×8, warmup2; MPS sync.

Медиана серий, мс/запрос. B256 — batch duration/256, **throughput**,
а не индивидуальная online latency.

| Модель | CPU B1 | MPS B1 | CPU B256 | MPS B256 |
| --- | ---: | ---: | ---: | ---: |
{cost_table}

![Serving](sasrec-256-distillation-cold/serving.png)

[CPU timings/memory](sasrec-256-distillation-cold/serving-cpu.json),
[MPS timings/memory](sasrec-256-distillation-cold/serving-mps.json).
Общий process держит три модели: control, student, recent teacher;
control делит long-ветвь учителя. RSS profiling process не равен памяти
standalone student. У control/student одинаковая архитектура и параметрический
размер; небольшая разница timings не означает изменение сложности.

## Проверки и пределы вывода

79 CPU/MPS тестов: loss/градиенты, frozen teacher, native2, default trainer,
warm start и cold state/RNG guard. Сохранены SHA source, data, checkpoint,
locked selection, исходники восстановленного initializer, logs/JSON/CSV/plots.
Test не открыт. Измерено fresh качество; stale replay и policy frontier
этого student не проводились. Прежние cache/frontier результаты относятся
к исходным CE50 weights. Один seed/256 пользователей ограничивают обобщение.
[Итоговые проверки артефактов](sasrec-256-distillation-cold/verification.json).
'''
    out.with_suffix(".md").write_text(report)
    print(json.dumps({"report":str(out.with_suffix('.md')),"training":training,"policy_valid":summary['policy_valid']},indent=2))


if __name__=="__main__":main()
