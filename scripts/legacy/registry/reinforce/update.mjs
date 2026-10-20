import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';
const root='/workspace/scratch/bece6597f7e0',dir=root+'/output/thesis-literature-2026-10-11',tmp=root+'/tmp/reinforce-registry';
const old=JSON.parse(await fs.readFile(dir+'/papers.json','utf8'));
assert.equal(old.papers.length,76);
const group='RL',group_name='Политики принятия решений: REINFORCE и off-policy learning';
const common={group,group_name,priority:'P2',priority_meaning:'Второй проход',experiment_is_proposal:true,reading_status:'screened_not_reproduced',checked_date:'2026-10-17',review_depth:'selected_method_sections'};
const added=[
 {...common,id:'P077',key:'Williams1992REINFORCE',short_name:'REINFORCE (Williams)',
 title:'Simple statistical gradient-following algorithms for connectionist reinforcement learning',authors:['Ronald J. Williams'],
 display_year:1992,preprint_year:null,version_year:null,bibliography_year:1992,
 method:'Score-function policy gradient / reward baseline / episodic REINFORCE',relevance_tier:'adjacent',
 summary:'Основополагающая работа о REINFORCE: обучает стохастический выбор действий по награде с помощью производной log-probability. Рассматривает baseline, эпизодическую награду и сочетание с backpropagation.',
 experiment:'Опциональная гипотеза: policy обновить/не обновлять кэш, награда — offline next-item quality минус стоимость вычислений. Сравнить с periodic refresh, порогами и supervised benefit predictor при фиксированных весах рекомендателя; controller train/tune до теста.',
 limitation:'Не работа о RecSys или cache refresh. Стохастические оценки могут быть шумными. Replay внутренних решений о кэше оценивает качество на фиксированных историях, а не причинный эффект новой политики показа на пользователей; новизна применения не установлена.',
 arxiv_id:null,arxiv_version:null,abstract_url:'https://link.springer.com/article/10.1007/BF00992696',html_url:null,
 preferred_url:'https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf',
 url_status:'publisher_metadata_and_fulltext_pdf_fetched',venue_as_reported:'Machine Learning 8, 229–256 (1992)',doi:'10.1007/BF00992696',
 experiment_axis:'refresh controller / discrete decisions / delayed quality-cost reward',
 tags:['RL','REINFORCE','policy gradient','reward baseline','refresh controller hypothesis'],
 source_sections:['General REINFORCE algorithms','5 Episodic REINFORCE algorithms','8.3 Choice of reinforcement baseline']},
 {...common,id:'P078',key:'1812.02353',short_name:'Top-K Off-Policy REINFORCE',
 title:'Top-K Off-Policy Correction for a REINFORCE Recommender System',
 authors:['Minmin Chen','Alex Beutel','Paul Covington','Sagar Jain','Francois Belletti','Ed Chi'],
 display_year:2019,preprint_year:2018,version_year:2021,bibliography_year:2019,
 method:'Policy-gradient recommender / learned behavior policy / importance weighting / top-K correction',relevance_tier:'direct',
 summary:'Применяет REINFORCE к YouTube-рекомендациям с миллионами объектов. Оценивает logging policy и корректирует обучение по историческим логам; вводит top-K correction и методы ограничения variance importance weights.',
 experiment:'Методологическая опора для разграничения cache-control replay и обучения новой политики показа. Если вводить RL для refresh, определить state/action/reward и сравнить с простыми контроллерами. Off-policy weighting для выбора items требует отдельного протокола и проверяемых предположений.',
 limitation:'Оптимизирует выбор рекомендуемых объектов, а не cache refresh. Приближённая коррекция и clipping меняют bias/variance; обычный NDCG-тест не заменяет off-policy оценку пользовательской награды. Результаты на YouTube не устанавливают выигрыш на Yambda.',
 arxiv_id:'1812.02353',arxiv_version:'v3',abstract_url:'https://arxiv.org/abs/1812.02353',html_url:'https://arxiv.org/html/1812.02353v3',
 preferred_url:'https://arxiv.org/html/1812.02353v3',url_status:'html_fetched',venue_as_reported:'WSDM 2019',doi:'10.1145/3289600.3290999',
 experiment_axis:'off-policy recommendation / logging bias / policy evaluation boundaries',
 tags:['RL','REINFORCE','off-policy','importance sampling','top-K','YouTube'],
 source_sections:['3 Reinforce Recommender','4 Off-Policy Correction','4.2 Estimating the behavior policy','4.3 Top-K Off-Policy Correction','4.4 Variance Reduction Techniques']}
];
for(const p of added)assert(!old.papers.some(q=>q.id===p.id||q.key===p.key||q.doi===p.doi));
const csv=await fs.readFile(dir+'/papers.csv','utf8'),bib=await fs.readFile(dir+'/references.bib','utf8');
for(const file of ['papers.csv','papers.json','references.bib'])await fs.copyFile(dir+'/'+file,tmp+'/before-'+file);
const wb=await Workbook.fromCSV(csv,{sheetName:'Registry'}),sh=wb.worksheets.getItemAt(0);
const keys=sh.getRange('A1:S1').values[0].map(v=>String(v).replace(/^\uFEFF/,''));
const previous=sh.getRange('A2:S77').values;
sh.getRange('A78:S79').values=added.map(p=>keys.map(k=>Array.isArray(p[k])?p[k].join('; '):p[k]??''));
await wb.recalculate();assert.deepEqual(sh.getRange('A2:S77').values,previous);
const esc=v=>{const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
const eol=csv.includes('\r\n')?'\r\n':'\n';
const newCsv=csv+(csv.endsWith('\n')?'':eol)+sh.getRange('A78:S79').values.map(r=>r.map(esc).join(',')).join(eol)+eol;
const check=await Workbook.fromCSV(newCsv,{sheetName:'Check'});assert.deepEqual(check.worksheets.getItemAt(0).getRange('A2:S77').values,previous);assert.equal(check.worksheets.getItemAt(0).getRange('A79').values[0][0],'P078');
const updated=structuredClone(old);updated.papers.push(...added);updated.groups[group]=group_name;
updated.counts.total+=2;updated.counts.adjacent+=1;updated.counts.direct_models_systems+=1;updated.counts.arxiv+=1;updated.counts.verified_arxiv_html+=1;
updated.screening_note=updated.screening_note.replace('для 17 работ (9 опорных + 6 Jev + SAGE + DCN-V2)','для 19 работ (9 опорных + 6 Jev + SAGE + DCN-V2 + REINFORCE + Top-K Off-Policy REINFORCE)');
updated.year_note+=' Top-K Off-Policy REINFORCE: WSDM 2019; первый arXiv 2018, выбранная версия v3 2021. Williams REINFORCE: журнальная публикация 1992, arXiv не указан.';
assert.equal(updated.papers.length,78);assert.equal(new Set(updated.papers.map(p=>p.id)).size,78);assert.deepEqual(updated.papers.slice(0,76),old.papers);
const entries=`
@article{P077_REINFORCE_Williams,
  title = {{Simple statistical gradient-following algorithms for connectionist reinforcement learning}},
  author = {Ronald J. Williams},
  journal = {Machine Learning},
  volume = {8},
  pages = {229--256},
  year = {1992},
  doi = {10.1007/BF00992696},
  url = {https://link.springer.com/article/10.1007/BF00992696},
  note = {Full-text PDF: https://gwern.net/doc/reinforcement-learning/model-free/1992-williams.pdf; accessed 2026-10-17}
}

@inproceedings{P078_TopK_OffPolicy_REINFORCE,
  title = {{Top-K Off-Policy Correction for a REINFORCE Recommender System}},
  author = {Minmin Chen and Alex Beutel and Paul Covington and Sagar Jain and Francois Belletti and Ed Chi},
  booktitle = {Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  series = {WSDM '19},
  year = {2019},
  doi = {10.1145/3289600.3290999},
  url = {https://arxiv.org/html/1812.02353v3},
  eprint = {1812.02353v3},
  archivePrefix = {arXiv},
  note = {Conference publication 2019; first arXiv preprint 2018; selected arXiv version v3 2021; accessed 2026-10-17}
}
`;
const qa=wb.worksheets.add('Added records');qa.getRange('A1:D3').values=[['ID','Работа','Метод','Связь с дипломом'],...added.map(p=>[p.id,p.short_name,p.method,p.experiment])];qa.getRange('A1:D3').format.font={name:'Arial',size:11};qa.getRange('A1:D3').format.wrapText=true;qa.getRange('A:A').format.columnWidth=10;qa.getRange('B:B').format.columnWidth=32;qa.getRange('C:C').format.columnWidth=48;qa.getRange('D:D').format.columnWidth=80;qa.getRange('A1:D3').format.rowHeight=150;qa.getRange('A1:D1').format.rowHeight=35;qa.getRange('A1:D1').format.fill='#E2E8F0';await wb.recalculate();
const png=await wb.render({sheetName:'Added records',range:'A1:D3',scale:1.2,format:'png'});await fs.writeFile(tmp+'/added-records.png',new Uint8Array(await png.arrayBuffer()));
await fs.writeFile(dir+'/papers.csv',newCsv);await fs.writeFile(dir+'/papers.json',JSON.stringify(updated,null,2)+'\n');await fs.writeFile(dir+'/references.bib',bib+entries);await fs.writeFile(root+'/upload/papers.csv',newCsv);
await fs.writeFile(tmp+'/upload-request.json',JSON.stringify({uploads:[
 {local_path:dir+'/papers.csv',purpose:'replace_library_file',library_file_id:'libfile_dec466b77b7c81919506b6ebee568968',expected_current_version:3,version_reason:'Add P077 REINFORCE and P078 Top-K Off-Policy REINFORCE'},
 {local_path:dir+'/papers.json',purpose:'replace_library_file',library_file_id:'libfile_443380236094819198e912e5e6a91d93',expected_current_version:3,version_reason:'Add two RL references and update counts'},
 {local_path:dir+'/references.bib',purpose:'replace_library_file',library_file_id:'libfile_9dd07cc4e704819181b9ed4ec4eb8837',expected_current_version:3,version_reason:'Add Williams 1992 and Chen et al. WSDM 2019'}
]},null,2));
console.log(JSON.stringify({count:updated.counts,added:added.map(p=>({id:p.id,title:p.short_name,priority:p.priority})),previous76Unchanged:true}));
