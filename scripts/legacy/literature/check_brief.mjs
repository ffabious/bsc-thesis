import fs from 'node:fs/promises';
import vm from 'node:vm';
import {chromium} from '/opt/codex/runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/index.mjs';
const base='/workspace/scratch/bece6597f7e0';
const path=base+'/output/thesis-literature-2026-10-11/literature-brief.html';
const source=await fs.readFile(path,'utf8');
const scripts=[...source.matchAll(/<script(?:\s[^>]*)?>([\s\S]*?)<\/script>/g)].map(m=>m[1]);
new vm.Script(scripts.at(-1));
const browser=await chromium.launch({headless:true,args:['--no-sandbox']});
const page=await browser.newPage({viewport:{width:1280,height:800},deviceScaleFactor:1});
let errors=[];page.on('pageerror',e=>errors.push(e.message));
await page.goto('file://'+path);
await page.waitForTimeout(100);
await fs.mkdir(base+'/tmp/qa',{recursive:true});
const geometry=[];
for(let i=0;i<11;i++){
 if(i)await page.locator('#next').click();
 const g=await page.locator('.slide.active').evaluate(el=>({id:el.id,width:el.scrollWidth,clientWidth:el.clientWidth,height:el.getBoundingClientRect().height,bottom:el.getBoundingClientRect().bottom}));
 geometry.push(g);await page.screenshot({path:base+'/tmp/qa/slide-'+(i+1)+'.png',fullPage:true});
}
await page.locator('#toggleCatalog').click();
if(await page.locator('.paper:visible').count()!==68)throw new Error('Catalog must contain 68 papers');
await page.locator('#search').fill('MuSeR');
if(await page.locator('.paper:visible').count()!==1)throw new Error('MuSeR search failed');
await page.locator('#search').fill('');await page.locator('#tier').selectOption('direct');
if(await page.locator('.paper:visible').count()!==58)throw new Error('Direct tier count incorrect');
await page.locator('#tier').selectOption('');await page.locator('#group').selectOption('LONG');
if(await page.locator('.paper:visible').count()!==21)throw new Error('Method filter count incorrect');
await page.locator('#group').selectOption('');
await page.locator('.paper').first().locator('summary').click();
await page.screenshot({path:base+'/tmp/qa/catalog.png',fullPage:false});
const [download]=await Promise.all([page.waitForEvent('download'),page.locator('#downloadJson').click()]);
await download.saveAs(base+'/tmp/qa/download-papers.json');
const saved=JSON.parse(await fs.readFile(base+'/tmp/qa/download-papers.json','utf8'));
if(saved.papers.length!==68)throw new Error('Downloaded JSON invalid');
await page.locator('#toggleCatalog').click();await page.keyboard.press('Home');
await page.setViewportSize({width:390,height:844});
const mobile=[];
for(let i=0;i<11;i++){
 if(i)await page.keyboard.press('ArrowRight');
 mobile.push(await page.evaluate(()=>({width:innerWidth,docWidth:document.documentElement.scrollWidth,id:document.querySelector('.slide.active').id})));
 if([0,1,5,9,10].includes(i))await page.screenshot({path:base+'/tmp/qa/mobile-'+(i+1)+'.png',fullPage:true});
}
await browser.close();
if(errors.length)throw new Error(errors.join('\n'));
if(geometry.some(g=>g.width>g.clientWidth+1)||mobile.some(g=>g.docWidth>g.width+1))throw new Error('Horizontal overflow');
console.log(JSON.stringify({slides:11,papers:68,desktop:geometry,mobile,errors},null,2));
