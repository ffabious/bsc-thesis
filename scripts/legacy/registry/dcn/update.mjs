import fs from 'node:fs/promises';
import assert from 'node:assert/strict';
import {Workbook} from '@oai/artifact-tool';

const root='/workspace/scratch/bece6597f7e0';
const dir=root+'/output/thesis-literature-2026-10-11';
const tmp=root+'/tmp/dcn-registry';
const csvPath=dir+'/papers.csv',jsonPath=dir+'/papers.json',bibPath=dir+'/references.bib';
const originalCsv=await fs.readFile(csvPath,'utf8');
const originalJson=await fs.readFile(jsonPath,'utf8');
const originalBib=await fs.readFile(bibPath,'utf8');
const catalog=JSON.parse(originalJson);
assert.equal(catalog.papers.length,75);
assert(!catalog.papers.some(p=>p.arxiv_id==='2008.13535'||/DCN.?V?2/i.test(p.short_name)));
for(const [name,body] of [['papers.csv',originalCsv],['papers.json',originalJson],['references.bib',originalBib]])await fs.writeFile(tmp+'/before-'+name,body);

const groupName='Взаимодействия признаков и объединение представлений';
const paper={
 key:'2008.13535',group:'FUSION',short_name:'DCN-V2',
 method:'Explicit feature crosses / full-rank and low-rank Cross Network / DCN-Mix',
 priority:'P1',
 summary:'Обучает явные взаимодействия признаков cross-слоями и объединяет их с MLP. Предлагает full-rank, low-rank и mixture-of-experts варианты; cross-блок можно использовать внутри другой модели.',
 experiment:'Baseline для объединения cached long-history и recent представлений: линейная проекция, MLP, full-rank CN и low-rank CN. Фиксировать энкодеры и refresh policy; сравнивать NDCG@10, Recall@100, полное время и память. Затем сравнивать refresh policies при одинаковых весах выбранной модели.',
 limitation:'Компонент feature interaction / ranking, а не самостоятельный последовательный энкодер или политика refresh. Само добавление CN не устанавливает новизну. Выигрыш на Yambda и способность компенсировать устаревание кэша не проверены.',
 id:'P076',display_year:2021,group_name:groupName,relevance_tier:'adjacent',
 priority_meaning:'Читать сначала',experiment_is_proposal:true,reading_status:'screened_not_reproduced',
 checked_date:'2026-10-17',
 title:'DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems',
 authors:['Ruoxi Wang','Rakesh Shivanna','Derek Z. Cheng','Sagar Jain','Dong Lin','Lichan Hong','Ed H. Chi'],
 arxiv_id:'2008.13535',arxiv_version:'v2',preprint_year:2020,version_year:2020,
 abstract_url:'https://arxiv.org/abs/2008.13535',html_url:'https://arxiv.org/html/2008.13535v2',
 preferred_url:'https://arxiv.org/html/2008.13535v2',url_status:'html_fetched',
 venue_as_reported:"Proceedings of the Web Conference 2021 (WWW '21)",
 doi:'10.1145/3442381.3450078',review_depth:'selected_method_sections',bibliography_year:2021,
 experiment_axis:'fusion architecture / feature interaction / rank-cost trade-off',
 tags:['FUSION','Cross Network','low-rank','DCN-Mix','ranking component','cached + recent fusion'],
 source_sections:['3.1 Embedding Layer','3.2 Cross Network','3.5 Cost-Effective Mixture of Low-Rank DCN','3.6 Complexity Analysis']
};

const wb=await Workbook.fromCSV(originalCsv,{sheetName:'Papers'});
const sheet=wb.worksheets.getItemAt(0);
const header=sheet.getRange('A1:S1').values[0].map(v=>String(v).replace(/^\uFEFF/,''));
const previous=sheet.getRange('A2:S76').values;
assert.equal(previous.length,75);assert.equal(previous.at(-1)[0],'P075');
const newRow=header.map(k=>Array.isArray(paper[k])?paper[k].join('; '):paper[k]??'');
sheet.getRange('A77:S77').values=[newRow];
await wb.recalculate();
assert.deepEqual(sheet.getRange('A2:S76').values,previous);
assert.equal(sheet.getRange('A77:S77').values[0][0],'P076');
// Serialize the authored row to CSV and retain the original 75 records byte-for-byte.
const csvEscape=v=>{const s=String(v??'');return /[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;};
const eol=originalCsv.includes('\r\n')?'\r\n':'\n';
const csv=originalCsv+(originalCsv.endsWith('\n')?'':eol)+sheet.getRange('A77:S77').values[0].map(csvEscape).join(',')+eol;
const parsed=await Workbook.fromCSV(csv,{sheetName:'Verify'});
assert.deepEqual(parsed.worksheets.getItemAt(0).getRange('A2:S76').values,previous);
assert.equal(parsed.worksheets.getItemAt(0).getRange('A77:S77').values[0][1],'DCN-V2');

catalog.papers.push(paper);catalog.groups.FUSION=groupName;catalog.as_of='2026-10-17';
catalog.counts.total+=1;catalog.counts.adjacent+=1;catalog.counts.arxiv+=1;catalog.counts.verified_arxiv_html+=1;
catalog.screening_note=catalog.screening_note.replace('для 16 работ (9 опорных + 6 Jev + SAGE)','для 17 работ (9 опорных + 6 Jev + SAGE + DCN-V2)');
catalog.year_note+=' DCN-V2: WWW 2021; arXiv/v2 2020.';
assert.deepEqual(catalog.papers.slice(0,75),JSON.parse(originalJson).papers);
assert.equal(catalog.counts.total,76);
assert.equal(new Set(catalog.papers.map(p=>p.id)).size,76);
const bibEntry=`\n@inproceedings{P076_DCNV2,
  title = {{DCN V2: Improved Deep \\& Cross Network and Practical Lessons for Web-scale Learning to Rank Systems}},
  author = {Ruoxi Wang and Rakesh Shivanna and Derek Z. Cheng and Sagar Jain and Dong Lin and Lichan Hong and Ed H. Chi},
  booktitle = {Proceedings of the Web Conference 2021},
  series = {WWW '21},
  year = {2021},
  doi = {10.1145/3442381.3450078},
  url = {https://arxiv.org/html/2008.13535v2},
  eprint = {2008.13535v2},
  archivePrefix = {arXiv},
  note = {Conference publication 2021; arXiv preprint and selected version v2 2020; accessed 2026-10-17}
}\n`;
assert(!originalBib.includes('P076_DCNV2'));

// The rendered view is internal QA only: CSV retains its original schema and format.
const preview=wb.worksheets.add('Added record');
preview.getRange('A1:B8').values=[['Поле','P076 · DCN-V2'],['Год','WWW 2021; arXiv/v2 2020'],['Группа',groupName],['Метод',paper.method],['Кратко',paper.summary],['Эксперимент (гипотеза)',paper.experiment],['Ограничения',paper.limitation],['HTML',paper.html_url]];
preview.getRange('A1:B8').format.font={name:'Arial',size:11};preview.getRange('A1:B8').format.wrapText=true;
preview.getRange('A:A').format.columnWidth=27;preview.getRange('B:B').format.columnWidth=100;
preview.getRange('A1:B8').format.rowHeight=65;preview.getRange('A6:B6').format.rowHeight=95;
preview.getRange('A1:B1').format={fill:'#263E5B',font:{color:'#FFFFFF',bold:true,size:11}};
await wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:'Papers',range:'A77:H77',maxChars:1200})).ndjson);
const png=await wb.render({sheetName:'Added record',range:'A1:B8',scale:1.2,format:'png'});
await fs.writeFile(tmp+'/added-record.png',new Uint8Array(await png.arrayBuffer()));

await fs.writeFile(csvPath,csv);
await fs.writeFile(jsonPath,JSON.stringify(catalog,null,2)+'\n');
await fs.writeFile(bibPath,originalBib+bibEntry);
await fs.writeFile(root+'/upload/papers.csv',csv);
await fs.writeFile(tmp+'/upload-request.json',JSON.stringify({uploads:[
 {local_path:csvPath,purpose:'replace_library_file',library_file_id:'libfile_dec466b77b7c81919506b6ebee568968',expected_current_version:2,version_reason:'Add P076 DCN-V2; preserve P001–P075'},
 {local_path:jsonPath,purpose:'replace_library_file',library_file_id:'libfile_443380236094819198e912e5e6a91d93',expected_current_version:2,version_reason:'Add P076 DCN-V2; update counts and method group'},
 {local_path:bibPath,purpose:'replace_library_file',library_file_id:'libfile_9dd07cc4e704819181b9ed4ec4eb8837',expected_current_version:2,version_reason:'Add DCN-V2 WWW 2021 bibliography entry'}
]},null,2));
console.log(JSON.stringify({total:catalog.counts.total,added:paper.id,previousRowsUnchanged:true,files:[csvPath,jsonPath,bibPath]}));
