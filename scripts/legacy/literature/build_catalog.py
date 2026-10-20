from pathlib import Path
import csv, json, re, collections, html

ROOT = Path('/workspace/scratch/bece6597f7e0')
OUT = ROOT / 'output/thesis-literature-2026-10-11'
meta = json.loads((ROOT / 'tmp/research/metadata.json').read_text())
rows = list(csv.DictReader((ROOT / 'tmp/research/curated.tsv').open(), delimiter='\t'))
groups = {
 'HIST': 'До трансформеров: динамика, локальные энкодеры и память',
 'ATTN': 'Последовательные трансформеры и обучение представлений',
 'LONG': 'Длинная история, выбор событий, кэши и промышленный инференс',
 'EFFIC': 'Эффективные энкодеры: low-rank, linear attention, RNN, SSM, MLP',
 'DYNAMIC': 'Изменение истории, пользовательских состояний и весов',
 'TRAIN': 'Сильные baselines и обучение на большом каталоге',
 'DATA': 'Музыкальные данные и протокол Yambda',
 'EVAL': 'Офлайн-оценка и воспроизводимость',
 'TRANSFER': 'Смежные вычислительные методы: гипотезы переноса',
 'JEV': 'Jev: structured decisions, общий контекст и управление памятью',
}
non = {
 'FPMC': dict(title='Factorizing Personalized Markov Chains for Next-Basket Recommendation', authors='Steffen Rendle, Christoph Freudenthaler, Lars Schmidt-Thieme', url='https://www.ismll.uni-hildesheim.de/pub/pdfs/RendleFreudenthaler2010-FPMC.pdf', venue='WWW 2010', ref='turn47view2', doi='10.1145/1772690.1772773'),
 'SHAN': dict(title='Sequential Recommender System based on Hierarchical Attention Networks', authors='Haochao Ying, Fuzhen Zhuang, Fuzheng Zhang, Yanchi Liu, Guandong Xu, Xing Xie, Hui Xiong, Jian Wu', url='https://www.ijcai.org/proceedings/2018/546', venue='IJCAI 2018', ref='turn19view2', doi='10.24963/ijcai.2018/546'),
 'TiSASRec': dict(title='Time Interval Aware Self-Attention for Sequential Recommendation', authors='Jiacheng Li, Yujie Wang, Julian McAuley', url='https://cseweb.ucsd.edu/~jmcauley/pdfs/wsdm20b.pdf', venue='WSDM 2020', ref='turn47view1', doi='10.1145/3336191.3371786'),
 'LightSANs': dict(title='Lighter and Better: Low-Rank Decomposed Self-Attention Networks for Next-Item Recommendation', authors='Xinyan Fan, Zheng Liu, Jianxun Lian, Wayne Xin Zhao, Xing Xie, Ji-Rong Wen', url='https://www.microsoft.com/en-us/research/publication/lighter-and-better-low-rank-decomposed-self-attention-networks-for-next-item-recommendation/', venue='SIGIR 2021', ref='turn47view0'),
 'SampledMetrics': dict(title='On Sampled Metrics for Item Recommendation', authors='Walid Krichene, Steffen Rendle', url='https://research.google/pubs/on-sampled-metrics-for-item-recommendation/', venue='KDD 2020', ref='turn47view3', doi='10.1145/3394486.3403226'),
}
deep = {'2507.15994','2607.14331','2402.17152','2601.01712','2510.22049','2609.23677','1905.09248','2506.02267','2310.02367'}
jev_tiers={'2609.40241': 'direct', '2609.37647': 'evaluation', '2609.25845': 'adjacent', '2610.00831': 'adjacent', '2609.23986': 'adjacent', '2609.38850': 'adjacent'}
deep.update(jev_tiers)
deep.add('2510.02470')
papers = []
for i,row in enumerate(rows, 1):
 key=row['key']; p=dict(row); p['id']=f'P{i:03d}'; p['display_year']=int(p.pop('year')); p['group_name']=groups[p['group']]
 p['relevance_tier']='adjacent' if p['group']=='TRANSFER' else ('evaluation' if p['group']=='EVAL' else 'direct')
 if p['group']=='JEV': p['relevance_tier']=jev_tiers[key]
 p['priority_meaning']={'P1':'Читать сначала','P2':'Второй проход','P3':'Расширение покрытия'}[p['priority']]
 p['experiment_is_proposal']=True; p['reading_status']='screened_not_reproduced'; p['checked_date']='2026-10-11'
 if key in non:
  n=non[key]; p.update(title=n['title'],authors=n['authors'].split(', '),arxiv_id=None,arxiv_version=None,preprint_year=None,version_year=None,abstract_url=n['url'],html_url=None,preferred_url=n['url'],url_status='primary_source_fetched',venue_as_reported=n['venue'],source_refs=[n['ref']],doi=n.get('doi'),review_depth='abstract_and_metadata',bibliography_year=p['display_year'])
 else:
  m=meta[key]; title=m['title'].replace('S^3-Rec','S3-Rec'); version=m.get('version'); urls=m.get('html_verified',[])
  hu=next((u for u in reversed(urls) if version and u.endswith(version)), urls[-1] if urls else None)
  vy=re.search(r'last revised[^\n]*?(20\d\d)',m.get('submitted','')) or re.search(r'Submitted on[^\n]*?(20\d\d)',m.get('submitted',''))
  vyear=int(vy.group(1)) if vy else (p['display_year'] if key in {'2504.09596','2506.13315','2510.22049','2507.15994'} else 2000+int(key[:2]))
  p.update(title=title,authors=m['authors'].split(', '),arxiv_id=key,arxiv_version=version,preprint_year=2000+int(key[:2]),version_year=vyear,abstract_url=f'https://arxiv.org/abs/{key}',html_url=hu,preferred_url=hu or f'https://arxiv.org/abs/{key}',url_status='html_fetched' if hu else ('abstract_fetched_html_unavailable' if m.get('abstract_fetched') else 'primary_indexed_abstract_only'),venue_as_reported=m.get('comments','Не установлено по первичной странице'),source_refs=list(dict.fromkeys(m.get('refs',[]))),doi=f'10.48550/arXiv.{key}',review_depth='selected_method_sections' if key in deep else ('abstract_and_metadata' if m.get('abstract_fetched') else 'indexed_abstract'),bibliography_year=vyear)
 if key=='2510.02470':
  p['checked_date']='2026-10-12'
  p['local_diagnostic']={'status':'literal_pseudocode_checked','report_url':'sage-brief.html','script_url':'sage_fd_check.py','results_url':'sage-diagnostics.json','authors_experiments_reproduced':False,'fd_counterexample_spectral_error':200.0,'reference_fd_error':2.0,'fd_bound':2.0}
 p['experiment_axis']={'HIST':'recent encoder / architecture','ATTN':'representation learning / causality','LONG':'history selection / cache / serving','EFFIC':'encoder quality-cost / state size','DYNAMIC':'adaptive history / dynamic state / model drift','TRAIN':'training setup / large catalog','DATA':'chronological replay','EVAL':'evaluation validity','TRANSFER':'compute / memory / controller objective','JEV':'structured ranking / prefix reuse / calibrated controller'}[p['group']]
 if key=='2510.02470':p['experiment_axis']='training-data selection / gradient and sketch cost'
 p['tags']=[p['group'],p['method'],p['experiment_axis']]
 p.pop('source_refs',None)
 papers.append(p)
assert len(papers)==75 and len({p['key'] for p in papers})==75
assert all(p['title'] and p['authors'] and p['preferred_url'].startswith('https://') for p in papers)
catalog={'schema_version':1,'as_of':'2026-10-12','proposal_date':'2026-10-09','proposal_title':'A Study of Efficient User Representation Computation and Maintenance in Sequential Recommender Systems','scope':'Yambda-50M; offline-only; cached long-history + recent encoder; refresh policy; KV reuse','counts':{'total':len(papers),'direct_models_systems':sum(p['relevance_tier']=='direct' and p['group']!='DATA' for p in papers),'dataset':sum(p['group']=='DATA' for p in papers),'evaluation':sum(p['relevance_tier']=='evaluation' for p in papers),'adjacent':sum(p['relevance_tier']=='adjacent' for p in papers),'arxiv':sum(bool(p['arxiv_id']) for p in papers),'verified_arxiv_html':sum(bool(p['html_url']) for p in papers)},'groups':groups,'screening_note':'Это структурированный первый проход: метаданные и abstract проверены по первичным источникам, для 16 работ (9 опорных + 6 Jev + SAGE) прочитаны выбранные разделы методов/обслуживания. Полное чтение всех статей и воспроизведение авторских экспериментов не выполнены. Для SAGE проверена буквальная реализация псевдокода на локальном контрпримере. Для TransX и REPAIR доступен первичный индексированный abstract, но полная страница не загрузилась.','year_note':'display_year — год указанной публикации или актуальной версии. preprint_year и version_year хранятся отдельно. LinRec: SIGIR 2023, arXiv 2024; ARGUS: препринт 2025, KDD/v2 2026.','papers':papers}
(OUT/'papers.json').write_text(json.dumps(catalog,ensure_ascii=False,indent=2))
lines=['% Bibliography generated from verified primary metadata, 2026-10-12.','% ArXiv entries cite the selected preprint version. Conference metadata is only used when established.','% Primary article URLs are in url; verified arXiv HTML URLs are in howpublished.','']
for p in papers:
 key=p['id']+'_'+re.sub(r'[^A-Za-z0-9]','',p['short_name'])[:35]
 entry='misc' if p['arxiv_id'] else 'inproceedings'
 fields={'title':'{'+p['title']+'}','author':' and '.join(p['authors']),'year':str(p['bibliography_year']),'url':p['abstract_url']}
 if p['arxiv_id']:
  fields.update(eprint=p['arxiv_id']+(p['arxiv_version'] or ''),archivePrefix='arXiv')
  if p['html_url']:fields['howpublished']=r'\url{'+p['html_url']+'}'
  fields['note']='Selected arXiv version; first preprint '+str(p['preprint_year'])+'; accessed '+p['checked_date']
 else: fields['booktitle']=p['venue_as_reported']; fields['note']='Accessed 2026-10-11'
 if p.get('doi'):fields['doi']=p['doi']
 lines.append('@'+entry+'{'+key+',\n'+',\n'.join('  '+k+' = {'+v.replace('&',r'\&')+'}' for k,v in fields.items())+'\n}\n')
(OUT/'references.bib').write_text('\n'.join(lines))
# CSV is a supporting interchange export of the same catalog, not a modeled workbook.
with (OUT/'papers.csv').open('w',encoding='utf-8-sig',newline='') as f:
 fields=['id','short_name','title','display_year','preprint_year','version_year','group_name','method','priority','relevance_tier','summary','experiment','limitation','preferred_url','abstract_url','html_url','url_status','review_depth','authors']
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
 for p in papers:w.writerow({**p,'authors':'; '.join(p['authors'])})
print(json.dumps(catalog['counts'],ensure_ascii=False))
