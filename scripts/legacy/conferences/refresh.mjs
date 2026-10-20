import fs from 'node:fs/promises';
import {FileBlob, SpreadsheetFile} from '@oai/artifact-tool';

const dir='/workspace/scratch/bece6597f7e0/output/conferences-2026-10-14';
const path=dir+'/conferences-and-workshops-2026.xlsx';
const asof='2026-10-19';
const d=s=>s?Math.round((Date.parse(s+'T00:00:00Z')-Date.UTC(1899,11,30))/86400000):null;
const research=JSON.parse(await fs.readFile(dir+'/conference-research.json','utf8'));
const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(path));
console.log((await wb.inspect({kind:'workbook,sheet,table',maxChars:2500,tableMaxRows:2,tableMaxCols:4})).ndjson);
const guide=wb.worksheets.getItem('Начать здесь');
const ws=wb.worksheets.getItem('Воркшопы');
const cs=wb.worksheets.getItem('Конференции');
const patches={
 W01:{checked:asof},
 W02:{status:'CFP появляются; подбор открыт',extra:'https://llm4ecommerce.github.io/',notes:'Проверено 19.10: опубликован LLM4ECommerce (30.11), но он требует LLM/e-commerce вклада и слабо подходит текущей теме. Полного списка на проверенных страницах нет. Общий ориентир сайтов/CFP — 25.10. Дата 30.11 — верхняя граница, конкретный workshop может назначить раньше; notification тоже не позднее 18.12. Очная презентация.',checked:asof},
 W03:{status:'Ждать авторских CFP',notes:'19.10 — уведомление ОРГАНИЗАТОРОВ о принятии воркшопов. Это не дедлайн авторских статей. На странице Programme и в workshop call на 19.10 перечень конкретных CFP и сроки авторов не найдены.',extra:'https://www.ecir2027.co.uk/programme',checked:asof},
 W04:{checked:asof},W05:{checked:asof},
 C01:{checked:asof},C02:{checked:asof},
 C03:{status:'Abstract: 19.10 до 14:59 МСК',notes:'Abstract 18.10, 23:59 AoE = 19.10, 14:59 МСК. После этого полный текст 25.10 AoE доступен только с вовремя зарегистрированным abstract. Не считать срок закрытым по одной календарной дате без учёта AoE. Rebuttal 15–20 декабря.',checked:asof},
 C04:{checked:asof},C05:{checked:asof},
 C06:{status:'Сегодня; только с abstract',notes:'Paper deadline — сегодня, 19.10; час и зона в CFP не указаны. Рассматривать только с abstract, зарегистрированным до 12.10. Нужна независимая репликация опубликованной работы, а не только новая модель.',checked:asof},
 C13:{notes:'Повторная проверка 19.10: Full CFP по-прежнему помечает даты PROPOSED. Short CFP пока заглушка без сроков. До фиксации считать предварительными; abstract обязателен. Workshop CFP проверять отдельно.',checked:asof}
};
const fields={status:'C',paper:'D',abstract:'E',tz:'F',notification:'G',camera:'H',event:'I',place:'J',priority:'K',fit:'L',format:'M',angle:'N',notes:'O',source:'P',extra:'Q',checked:'R'};
const link=url=>`=IFERROR(HYPERLINK("${url}","${url}"),"${url}")`;
for(const sh of [ws,cs]){
 sh.getRange('C5').values=[['Статус на 19.10.2026']];
 sh.getRange('A3').values=[['Обновлено 19.10.2026. Открытые/ожидаемые CFP перепроверены; закрытые циклы сохранены. R — дата проверки источника; S — дни от 19.10.']];
 sh.getRange('S5').values=[['До даты D, дней от 19.10.2026']];
 const existing=sh.getRange('A6:A18').values.map(r=>r[0]);
 for(let i=0;i<existing.length;i++){
  const id=existing[i], row=i+6, patch=patches[id], rr=research.rows.find(r=>r.id===id);
  rr.checked=rr.checked||'2026-10-14';
  if(patch){
   Object.assign(rr,patch);
   for(const [key,value] of Object.entries(patch)){
    const target=sh.getRange(`${fields[key]}${row}`);
    if(key==='extra'||key==='source') target.formulas=[[link(value)]];
    else target.values=[[[ 'paper','abstract','notification','camera','checked'].includes(key)?d(value):value]];
   }
  }
 }
 sh.getRange('S6:S18').formulas=existing.map((_,i)=>[`=IF(D${i+6}="","",D${i+6}-'Начать здесь'!$B$3)`]);
 sh.getRange('S6:S18').setNumberFormat('0;[Red]-0;0');
}
guide.getRange('B3').values=[[d(asof)]];
guide.getRange('D3').values=[['26 вариантов. Действующие/ожидаемые CFP проверены 19.10; архив — с датами прежней проверки.']];
guide.getRange('E6:F6').values=[['Черновик и первые quality/cost результаты; проверить новые CFP к 25.10','Первые CFP появились (LLM4ECommerce); нужен профильный workshop. 30.11 — лишь верхняя граница']];
guide.getRange('E7:F7').values=[['После решения организаторам 19.10 ждать публикации списка и авторских CFP','На проверенных Programme / workshop call сроки авторских статей пока не найдены']];
guide.getRange('B21').values=[['AoE = UTC−12; 23:59 AoE = 14:59 следующего дня в Москве. WWW Full: abstract 18.10 AoE = 19.10, 14:59 МСК; далее paper только с вовремя поданным abstract. S считает календарные дни, без часа/зоны.']];
guide.getRange('A21:H21').format.rowHeight=54;
guide.getRange('B22').values=[['WSDM workshop 30.11 — верхняя граница; первые отдельные CFP появились. ECIR 19.10 — уведомление организаторов. EAIM: сайт 06.11, AAAI 04.11; EMC²: форматы расходятся. SIGIR Full: даты всё ещё PROPOSED.']];
guide.getRange('A22:H22').format.rowHeight=54;
guide.getRange('A28').values=[['Срочно 19.10']];
guide.getRange('B28:H28').merge();
guide.getRange('B28').values=[['C03 WWW Full: abstract до 14:59 МСК; paper 25.10 AoE только при регистрации abstract. C06 ECIR Replicability: paper сегодня только с abstract от 12.10; час/зона не опубликованы.']];
guide.getRange('A28:H28').format={wrapText:true,rowHeight:54,fill:'#FFF0CD'};
guide.getRange('A28').format.font={bold:true,color:'#80580D'};
research.checked=asof;
research.update_note='Перепроверены действующие и ожидаемые CFP; закрытые циклы сохраняют прежнюю дату проверки. Новая информация: LLM4ECommerce CFP. Число вариантов и ID сохранены.';
research.source_checks=[
 {date:asof,url:'https://llm4ecommerce.github.io/',finding:'Новый конкретный WSDM CFP: 30.11 AoE; слабое соответствие без LLM/e-commerce вклада'},
 {date:asof,url:'https://www.ecir2027.co.uk/programme',finding:'Конкретные workshop CFP не найдены на проверенной странице'},
 {date:asof,url:'https://www2027.thewebconf.org/important-dates/',finding:'WWW Full abstract: 18.10 AoE = 19.10 14:59 МСК; Short abstract 09.11 / paper 16.11'},
 {date:asof,url:'https://sigir2027.org/pages/submit-short.html',finding:'Заглушка без сроков; не добавлять неподтверждённый track'}
];
await wb.recalculate();
console.log((await wb.inspect({kind:'region',sheetId:'Конференции',range:'A6:H11',maxChars:4000,tableMaxRows:6,tableMaxCols:8})).ndjson);
console.log((await wb.inspect({kind:'region',sheetId:'Воркшопы',range:'P6:S10',maxChars:2500,tableMaxRows:5,tableMaxCols:4})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#VALUE!|#DIV/0!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20},maxChars:1500})).ndjson);
for(const [sh,range,file] of [[guide,'A1:G13','routes'],[ws,'A1:H10','workshops'],[cs,'A1:H11','conferences']]){
 const b=await wb.render({sheetName:sh.name,range,scale:1.1,format:'png'});
 await fs.writeFile(`${dir}/${file}-updated-preview.png`,new Uint8Array(await b.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);await output.save(path);
await fs.writeFile(dir+'/conference-research.json',JSON.stringify(research,null,2));
console.log(JSON.stringify({path,asof,updated:Object.keys(patches).length,rows:research.rows.length}));
