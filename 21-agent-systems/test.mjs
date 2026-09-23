import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const base=process.env.BASE_URL||'http://127.0.0.1:4173/21-agent-systems';
const paths=['observe-act','replanning','memory','context','verification','coordination'];
const browser=await chromium.launch({headless:true});
for(const p of paths){const page=await browser.newPage({viewport:{width:1280,height:800}});let errors=[];page.on('pageerror',e=>errors.push(String(e)));await page.goto(`${base}/${p}/`);await page.locator('#run').click();assert.equal(errors.length,0,`${p}: ${errors}`);assert.ok(await page.locator('.metric').count()>=3);assert.ok((await page.locator('#viz').innerText()).length>80);await page.close()}await browser.close();console.log(`PASS ${paths.length} demos`);
