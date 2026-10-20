from pathlib import Path
import json,csv,re,shutil
R=Path('/workspace/scratch/bece6597f7e0');T=R/'tmp/research';O=R/'output/thesis-literature-2026-10-11'
old=json.loads((O/'papers.json').read_text());(T/'pre-sage-catalog.json').write_text(json.dumps(old,ensure_ascii=False,indent=2))
rows=list(csv.DictReader((T/'curated.tsv').open(),delimiter='\t'));fields=list(rows[0])
new={'key':'2510.02470','group':'TRANSFER','short_name':'SAGE','year':'2025','method':'Training subset / gradient agreement / Frequent Directions','priority':'P2',
 'summary':'Отбирает обучающие примеры по cosine agreement градиентов в Frequent Directions скетче. Эксперименты с ResNet-18 на изображениях; перенос на RecSys не проверен.',
 'experiment':'Отдельная ось обучения: full data, random/stratified subset и исправленный FD agreement при равной доле данных и полном учёте отбора. Для refresh оставить общий фиксированный checkpoint.',
 'limitation':'Буквальный Algorithm 1 перезаписывает скетч: локальный контрпример нарушает FD bound. Для exact scores после consensus нужны сохранённые проекции либо дополнительный проход. Выигрыш Yambda не установлен.'}
assert '2510.02470' not in {p['key'] for p in rows}
rows.append(new)
with (T/'curated.tsv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)
m=json.loads((T/'metadata.json').read_text());m['2510.02470']={'title':'SAGE: Streaming Agreement-Driven Gradient Sketches for Representative Subset Selection','authors':'Ashish Jha, Salman Ahmadi-Asl','version':'v2','submitted':'Submitted on 2 Oct 2025; last revised 9 Oct 2025','abstract_fetched':True,'html_verified':['https://arxiv.org/html/2510.02470v2'],'comments':'Препринт; статус рецензируемой публикации не подтверждён','refs':['turn68view0','turn70view0','turn72view0'],'checked_date':'2026-10-12'};(T/'metadata.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
p=T/'build_catalog.py';s=p.read_text()
s=s.replace("deep.update(jev_tiers)","deep.update(jev_tiers)\ndeep.add('2510.02470')")
s=s.replace(" p['experiment_axis']=", " if key=='2510.02470':\n  p['checked_date']='2026-10-12'\n  p['local_diagnostic']={'status':'literal_pseudocode_checked','report_url':'sage-brief.html','script_url':'sage_fd_check.py','results_url':'sage-diagnostics.json','authors_experiments_reproduced':False,'fd_counterexample_spectral_error':200.0,'reference_fd_error':2.0,'fd_bound':2.0}\n p['experiment_axis']=")
s=s.replace(" p['tags']=", " if key=='2510.02470':p['experiment_axis']='training-data selection / gradient and sketch cost'\n p['tags']=")
s=s.replace('len(papers)==74 and len({p[\'key\'] for p in papers})==74','len(papers)==75 and len({p[\'key\'] for p in papers})==75')
s=s.replace("'as_of':'2026-10-11'","'as_of':'2026-10-12'")
s=s.replace('для 15 работ (9 опорных + 6 Jev)','для 16 работ (9 опорных + 6 Jev + SAGE)')
s=s.replace('Полное чтение всех статей и воспроизведение не выполнены.','Полное чтение всех статей и воспроизведение авторских экспериментов не выполнены. Для SAGE проверена буквальная реализация псевдокода на локальном контрпримере.')
s=s.replace('metadata, 2026-10-11.','metadata, 2026-10-12.')
s=s.replace("+'; accessed 2026-10-11'","+'; accessed '+p['checked_date']")
p.write_text(s)
# Presentation: retain original twelve slides, append one compact SAGE slide.
p=T/'build_brief.py';s=p.read_text()
s=re.sub(r'(?<![0-9])74(?![0-9])','75',s)
for a,z in [('6 по оценке + 9 смежных','6 по оценке + 10 смежных'),('Для 66 из 69','Для 67 из 70'),('разделами 15 работ','разделами 16 работ'),('+26 уникальных','+25 уникальных'),('Обзор: 11 октября 2026','Обновлено: 12 октября 2026'),('Диплом · литература · 11.10.2026','Диплом · литература · 12.10.2026')]:s=s.replace(a,z)
s=s.replace("['Смежные вычислительные идеи','5',", "['Смежные вычислительные идеи','6',")
extra="""slide('SAGE: отбор данных для обучения',f'<p class="lead">{a("2510.02470")} · gradient agreement + FD sketch</p><ul><li>Смежная ось: стоимость обучения рекомендателя.</li><li>Буквальный Algorithm 1 теряет накопленный скетч.</li><li>Exact scores требуют хранения проекций или дополнительного прохода.</li><li>Пилот: random/stratified subset против исправленного FD selection.</li></ul><p class="takeaway">Основные refresh-эксперименты по-прежнему используют общий фиксированный checkpoint.</p><p class="small"><a href="sage-brief.html">Разбор, контрпример и воспроизводимый diagnostic</a>. Авторские training results не воспроизведены.</p>','Дополнение · смежное обучение')
"""
s=s.replace('assert len(slides)==12',extra+'assert len(slides)==13')
p.write_text(s)
p=T/'build_review.py';s=p.read_text()
for a,z in [('74 работы.','75 работ.'),('6 по оценке, 9 смежных','6 по оценке, 10 смежных'),('11 октября 2026','12 октября 2026'),('около 26 <em>','около 25 <em>'),('Девять работ помечены','Десять работ помечены'),('Из 69 arXiv работ для 66','Из 70 arXiv работ для 67')]:s=s.replace(a,z)
s=s.replace('<h2>План расширения примерно до 100 статей</h2>', '<h2>SAGE: смежное направление обучения</h2><p>{ref("2510.02470")} добавлен как P075. <a href="sage-brief.html">Разбор и локальная проверка</a> содержат замечания к воспроизводимости. Метод относится к выбору training subset; включать его в основной refresh-policy benchmark пока не предлагается. Польза от отбора данных и польза от свежести состояния требуют отдельных сравнений.</p>\n<h2>План расширения примерно до 100 статей</h2>')
s=s.replace('<td>Какой минимальный объём истории сохраняет качество?</td><td>6</td>', '<td>Какой минимальный объём истории сохраняет качество?</td><td>5</td>')
p.write_text(s)
# The Jev note keeps its research date; only the link to the live aggregate changes.
p=O/'jev-brief.html';s=p.read_text().replace('счётчик 74 работ','счётчик статей').replace('Общий обзор: 74 работы','Общий обзор: 75 работ').replace('Общий обзор: 74 работы','Общий обзор: 75 работ');p.write_text(s)
p=O/'README.txt';s=p.read_text()
for a,z in [('12 коротких','13 коротких'),('74 работ','75 работ'),('74 статей','75 статей'),('74 библиографических','75 библиографических'),('P074.','P075.'),('6 по оценке + 9 смежных','6 по оценке + 10 смежных'),('69 работ','70 работ'),('для 66','для 67'),('разделами 15 работ','разделами 16 работ'),('около 26 новых','около 25 новых')]:s=s.replace(a,z)
s+='\nОбновление 12.10.2026: P075 SAGE, версия v2. sage-brief.html — разбор; sage_fd_check.py и sage-diagnostics.json — локальный diagnostic псевдокода (не репликация training benchmark).\n'
p.write_text(s)
p=O/'search-notes.json';d=json.loads(p.read_text());d['date']='2026-10-12';d['deduplication']=d['deduplication'].replace('P074','P075');d['next_pass']=d['next_pass'].replace('26','25');d['limitations']=d['limitations'].replace('15 papers (nine anchors and six Jev additions)','16 papers (nine anchors, six Jev additions and SAGE)');d['sage_addition']={'id':'P075','arxiv_id':'2510.02470','version':'v2','primary_url':'https://arxiv.org/html/2510.02470v2','date_checked':'2026-10-12','foundation_url':'https://arxiv.org/html/1501.01711v2','category':'adjacent training-data selection','review':'full SAGE v2 HTML, PDF Algorithm 1 and supplement; selected FD methods','local_diagnostic':'published pseudocode counterexample only; no author training experiment reproduction','numeric_result':{'literal_algorithm_error':200.0,'correct_FD_error':2.0,'bound':2.0},'followup_candidates_unreviewed':['GRAFT (2508.13653)','Market-Based Data Subset Selection (2510.02456)','GradMatch','GLISTER','CRAIG']};p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
shutil.copyfile(T/'sage_fd_check.py',O/'sage_fd_check.py');shutil.copyfile(T/'sage-diagnostics.json',O/'sage-diagnostics.json')
print('Added P075; existing paper IDs preserved.')
