from pathlib import Path
import json,csv,re,hashlib
from lxml import html
root=Path('/workspace/scratch/bece6597f7e0');out=root/'output/thesis-literature-2026-10-11'
d=json.loads((out/'papers.json').read_text());ps=d['papers']
assert len(ps)==75 and len({p['key'] for p in ps})==75
assert sum(p['relevance_tier']=='direct' for p in ps)==59
assert sum(bool(p['html_url']) for p in ps)==67
assert all(p['authors'] and p['title'] and p['summary'] and p['experiment'] and p['limitation'] for p in ps)
assert all(p['display_year']<=2026 for p in ps)
assert all(not p['html_url'] or p['arxiv_id'] in p['html_url'] for p in ps)
csvrows=list(csv.DictReader((out/'papers.csv').open(encoding='utf-8-sig')))
assert len(csvrows)==75 and [x['id'] for x in csvrows]==[x['id'] for x in ps]
bib=(out/'references.bib').read_text();assert len(re.findall(r'^@(?:misc|inproceedings)\{',bib,re.M))==75
depth=0
for c in bib:
 if c=='{':depth+=1
 if c=='}':depth-=1
 assert depth>=0
assert depth==0
doc=html.fromstring((out/'literature-brief.html').read_text())
assert len(doc.xpath('//section[contains(@class,"slide")]'))==13
assert len(doc.xpath('//article[contains(@class,"paper")]'))==75
assert not doc.xpath('//script[@src]|//link[@rel="stylesheet"]|//img')
embedded=json.loads(doc.get_element_by_id('catalogData').text)
assert embedded==d
dom={'slides':[],'papers':[],'elements':{}}
for el in doc.xpath('//*[@id]'):
 dom['elements'][el.get('id')]={'tagName':el.tag.upper(),'textContent':el.text or '','value':'','hidden':el.get('hidden') is not None,'className':el.get('class','')}
for el in doc.xpath('//section[contains(@class,"slide")]'):dom['slides'].append(el.get('id'))
for el in doc.xpath('//article[contains(@class,"paper")]'):
 dom['papers'].append({'hidden':False,'dataset':{k[5:]:v for k,v in el.attrib.items() if k.startswith('data-')}})
(root/'tmp/research/dom.json').write_text(json.dumps(dom))
for f in ['literature-review.html','literature-brief.html','jev-brief.html','sage-brief.html']:
 h=html.fromstring((out/f).read_text())
 for href in h.xpath('//a/@href'):
  if not href.startswith(('https://','#')):assert (out/href).is_file(),(f,href)
original=hashlib.sha256((root/'upload/document.pdf').read_bytes()).hexdigest()
assert hashlib.sha256((out/'proposal-2026-10-09.pdf').read_bytes()).hexdigest()==original
print(json.dumps({'papers':75,'html_sources':67,'slides':13,'csv':75,'bibtex':75,'proposal_sha256':original,'self_contained':True},indent=2))
