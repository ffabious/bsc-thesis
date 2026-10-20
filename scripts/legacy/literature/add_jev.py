from pathlib import Path
import json,csv,re
R=Path('/workspace/scratch/bece6597f7e0'); T=R/'tmp/research'; O=R/'output/thesis-literature-2026-10-11'
new=[
 ('2609.40241','Jev-Rec','Decision-Oriented Recommendation Reranking: An Empirical Study of Jev','Hanjia Lyu, Yinglong Xia','30 Sep 2026','direct','P1','Choice probabilities / semantic reranking',
  'Jev ранжирует кандидатов по вероятностям Choice, используя 10 последних взаимодействий и текстовые описания товаров. Проверка на трёх доменах Amazon, K=20–200.',
  'Опционально проверить semantic reranking поверх фиксированного retrieval; отдельно измерять качество всего pipeline и условное качество reranker.',
  'Target искусственно включён в кандидаты; оценены только пользователи с target в SASRec top-200. Hosted API и локальные GPU не сопоставлены по hardware.'),
 ('2609.37647','Jev-Benchmark','Evaluating and Benchmarking the System One Model Jev','Tobias Deußer, Lorenz Sparrenberg, Rafet Sifa','29 Sep 2026','evaluation','P2','Typed decisions / calibration / selective prediction',
  'Независимая проверка jev-1.13.0 на 37 наборах. Choice, Score и Noul имеют разные свойства; хорошее ранжирование не гарантирует подходящий бинарный порог.',
  'Для refresh controller оценивать calibration и regret при заданном бюджете; настраивать порог до теста, сравнивать с регрессией выигрыша.',
  'Это NLP benchmark, не музыкальная рекомендация. Закрытые training data не позволяют исключить contamination; промпты и baseline readout влияют на результат.'),
 ('2609.25845','Visual Jev','Visual Jev: Accurate and Efficient Decisions from Shared Visual Context','Guanxu Yu, Yuhang Yao','22 Sep 2026','adjacent','P1','Shared prefix KV / batched isolated suffixes',
  'Общий визуальный префикс кодируется один раз, независимые вопросы исполняются батчем. Факторный эксперимент отделяет reuse от batching.',
  'Для общего user-prefix сравнить reuse on/off × batching on/off; проверять score error, top-k agreement, group latency и peak memory.',
  'Визуальные задачи, не RecSys. Выигрыш рассчитан на совместно доступные вопросы; амортизированное время не равно задержке запроса. Mixed precision меняет отдельные ответы.'),
 ('2610.00831','AnyJev','AnyJev Technical Report','Jiamu Zhang, Tianze Yang, Yucheng Shi, Evan Chen, Zixiang Nie, Kelly Wan, Liangjie Hong, Ninghao Liu, Liang Wu','30 Sep 2026','adjacent','P2','Next-token option readout / debiasing / adaptive rotations',
  'Открытая LLM возвращает распределение по option tokens без генерации. Label prior и cyclic rotations уменьшают смещение; ранняя остановка сокращает число prefills.',
  'Если нужен текстовый baseline, сравнить single readout и rotations при равной стоимости; проверять перестановки кандидатов и зависимость KV от контекста.',
  'Несколько rotations требуют нескольких prefills. Сертификат ограничивает disagreement с полным вариантом, не ошибку относительно истины. KV блока после state нельзя переносить между разными state.'),
 ('2609.23986','Jev-Mem','Jev-Mem: System-One-Controlled Agentic Memory for Efficient AI Agents','Dongming Jiang, Yi Li, Bingzhe Li','21 Sep 2026','adjacent','P2','Typed memory controller / budget allocation / adaptive stopping',
  'Контроллер управляет записью и поиском агентной памяти: маршрутизацией, бюджетом графа, оценкой кандидатов и остановкой по достаточности и ожидаемой пользе поиска.',
  'Учесть как смежный prior art для benefit-based compute: перенести принцип на решение refresh/keep с явной ценой обновления.',
  'Agentic memory и retrieval на LoCoMo; не проверяет freshness user embeddings или refresh policies на Yambda. Применение в дипломе — гипотеза.'),
 ('2609.38850','OpenJev-RLCD','OpenJev-RLCD: A Working RLCD Implementation','Zhimin Gao, Pichao Wang','30 Sep 2026','adjacent','P3','Proper scoring / calibrate-then-reinforce',
  'Открытая реализация обучает вероятностный readout и затем reasoning, используя proper score для каждого rationale. Анализирует calibration и selective prediction.',
  'Для вероятности полезного refresh сначала проверить BCE/Brier + calibration; RLCD рассматривать лишь при самостоятельной задаче обучения decision model.',
  'Независимая реализация, не раскрытие закрытого алгоритма TypeSafe. Генерирует rationale; результат не доказывает одношаговый быстрый inference коммерческого Jev.'),
]
meta=json.loads((T/'metadata.json').read_text())
rows=list(csv.DictReader((T/'curated.tsv').open(),delimiter='\t')); fields=list(rows[0]); keys={r['key'] for r in rows}
for key,short,title,authors,date,tier,priority,method,summary,experiment,limitation in new:
 if key not in keys:
  rows.append(dict(zip(fields,[key,'JEV',short,'2026',method,priority,summary,experiment,limitation])))
 meta[key]={'title':title,'authors':authors,'comments':'Препринт; публикация на конференции не установлена','version':'v1','submitted':'Submitted on '+date,'abstract_fetched':True,'refs':[],'html_verified':['https://arxiv.org/html/'+key+'v1']}
with (T/'curated.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
(T/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
b=T/'build_catalog.py'; s=b.read_text()
s=s.replace(" 'TRANSFER': 'Смежные вычислительные методы: гипотезы переноса',", " 'TRANSFER': 'Смежные вычислительные методы: гипотезы переноса',\n 'JEV': 'Jev: structured decisions, общий контекст и управление памятью',")
s=s.replace("papers = []", "jev_tiers="+repr({p[0]:p[5] for p in new})+"\ndeep.update(jev_tiers)\npapers = []")
s=s.replace(" p['priority_meaning']=", " if p['group']=='JEV': p['relevance_tier']=jev_tiers[key]\n p['priority_meaning']=")
s=s.replace("'TRANSFER':'compute / memory / controller objective'", "'TRANSFER':'compute / memory / controller objective','JEV':'structured ranking / prefix reuse / calibrated controller'")
s=s.replace("len(papers)==68 and len({p['key'] for p in papers})==68", "len(papers)==74 and len({p['key'] for p in papers})==74")
s=s.replace("'dataset':1,'evaluation':5,'adjacent':5", "'dataset':sum(p['group']=='DATA' for p in papers),'evaluation':sum(p['relevance_tier']=='evaluation' for p in papers),'adjacent':sum(p['relevance_tier']=='adjacent' for p in papers)")
s=s.replace('для 9 опорных работ','для 15 работ (9 опорных + 6 Jev)')
b.write_text(s)
# Update existing presentation and keep its original slide order.
b=T/'build_brief.py';s=b.read_text()
s=re.sub(r'(?<![0-9])68(?![0-9])','74',s)
for a,z in [('57 модельных','58 модельных'),('57 по моделям','58 по моделям'),('5 по оценке + 5 смежных','6 по оценке + 9 смежных'),('Для 60 из 63','Для 66 из 69'),('разделами 9 опорных работ','разделами 15 работ'),('+32 уникальные','+26 уникальных')]:s=s.replace(a,z)
slide="""slide('Jev: что взять в диплом',table([
 [a('2609.40241'),'Semantic reranking; отдельная дополнительная ветвь'],
 [a('2609.25845'),'Разделить влияние prefix reuse и batching'],
 [a('2610.00831'),'Порядок options; корректность переноса KV'],
 [a('2609.23986'),'Prior art: управление памятью по ожидаемой пользе'],
 [a('2609.37647')+' · '+a('2609.38850'),'Calibration и пороги decision controller']
 ],['Работа','Связь с экспериментами'])+'<p class="takeaway">Для текущего пропозала сначала полезны reuse-контроли и benefit-based controller.</p><p class="small">Шесть свежих препринтов; перенос на Yambda ещё не проверен. JevEmbed — отдельный программный ресурс, в число статей не входит.</p>','Дополнение · Jev / System One')
"""
s=s.replace('assert len(slides)==11',slide+'assert len(slides)==12')
s=s.replace(" ['Смежные вычислительные идеи'", " ['Jev / structured decisions','6',a('2609.40241')+' · '+a('2609.25845')],\n ['Смежные вычислительные идеи'")
b.write_text(s)
b=T/'build_review.py';s=b.read_text()
s=s.replace('"TRANSFER":"Перенос вычислительных идей как отдельная гипотеза"','"TRANSFER":"Перенос вычислительных идей как отдельная гипотеза","JEV":"Structured ranking; reuse; калибровка и контроллер памяти"')
s=s.replace('68 работ. 57 по моделям и системам, 1 статья о Yambda, 5 по офлайн-оценке, 5 смежных вычислительных работ.','74 работы. 58 по моделям и системам, 1 статья о Yambda, 6 по оценке, 9 смежных работ.')
s=s.replace('9 опорных','15 выбранных').replace('32 новых','26 новых').replace('32 уникальные','26 уникальных')
s=s.replace('</nav>','</nav><p><a href="jev-brief.html">Дополнение: Jev — 6 статей и идеи переноса</a></p>',1)
b.write_text(s)
print('Appended six Jev entries, existing IDs preserved.')
