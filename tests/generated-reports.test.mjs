import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import test from 'node:test';

const artifactRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const number=text=>Number(text.replaceAll('*','').replaceAll(',','').replaceAll('×',''));
const table=(markdown,heading)=>{
 const start=markdown.indexOf(heading+'\n');assert(start>=0,`Missing table: ${heading}`);
 const rest=markdown.slice(start+heading.length+1),next=rest.search(/\n#{2,3} /);
 return (next<0?rest:rest.slice(0,next)).split('\n').filter(line=>line.startsWith('| ')).slice(1).map(line=>line.split('|').slice(1,-1).map(value=>value.trim()));
};
const csvRecords=text=>{
 const rows=[];let row=[],cell='',quoted=false;
 for(let i=0;i<text.length;i++){
  const ch=text[i];
  if(ch==='"'){
   if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;
  }else if(!quoted&&(ch===','||ch==='\n')){
   row.push(cell);cell='';if(ch==='\n'){rows.push(row);row=[];}
  }else cell+=ch;
 }
 if(row.length||cell){row.push(cell);rows.push(row);}
 const [header,...data]=rows;return data.map(values=>Object.fromEntries(header.map((key,index)=>[key,values[index]])));
};
function verifyBests(rows,start,lowerColumns){
 for(let column=start;column<rows[0].length;column++){
  const values=rows.map(row=>number(row[column]));assert(values.every(Number.isFinite));
  const best=(lowerColumns.includes(column)?Math.min:Math.max)(...values);
  rows.forEach((row,index)=>assert.equal(/^\*\*.*\*\*$/.test(row[column]),values[index]===best,`Incorrect displayed best: column ${column}, ${row[0]}`));
 }
}

test('regenerated reports preserve manuscript scope, retained values and deterministic output',async t=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'generated-reports-'));
 t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
 for(const directory of ['aggregates','configs'])fs.cpSync(path.join(artifactRoot,directory),path.join(root,directory),{recursive:true});
 fs.mkdirSync(path.join(root,'scripts'));
 fs.copyFileSync(path.join(artifactRoot,'scripts/regenerate.mjs'),path.join(root,'scripts/regenerate.mjs'));
 const regenerate=timezone=>{
  const result=spawnSync(process.execPath,[path.join(root,'scripts/regenerate.mjs')],{cwd:os.tmpdir(),encoding:'utf8',env:{...process.env,TZ:timezone}});
  assert.equal(result.status,0,result.stderr);
  return Object.fromEntries(fs.readdirSync(path.join(root,'generated')).sort().map(name=>[name,fs.readFileSync(path.join(root,'generated',name),'utf8')]));
 };
 const output=regenerate('UTC'),markdown=output['tables.md'];
 const q=JSON.parse(fs.readFileSync(path.join(root,'aggregates/quality.json'),'utf8'));
 const checkpoints=JSON.parse(fs.readFileSync(path.join(root,'aggregates/checkpoints.json'),'utf8'));

 await t.test('all report files regenerate identically across runs and time zones',()=>{
  assert.deepEqual(Object.keys(output),['checkpoint-sweeps.svg','checkpoints.csv','concurrency-all-precisions.svg','concurrency.csv','concurrency.svg','decision-guide.csv','error-analysis.csv','hosted.csv','paired-comparisons.csv','per-field.csv','quality-intervals.svg','quality.csv','selection-stability.csv','serving-resources.csv','serving.csv','tables.md']);
  assert.deepEqual(regenerate('America/New_York'),output);
 });

 await t.test('main and appendix serving tables rank their own displayed configurations',()=>{
  const candidates=table(markdown,'### Gemma 4: validation-F1 checkpoints');
  const proxies=table(markdown,'### Llama 8B proxies: production architecture at three precisions');
  const prior=table(markdown,'### Prior 4B adapters: retained test-selected checkpoints');
  const supplementary=table(markdown,'### Supplementary configurations and block-matched proxies');
  assert.equal(candidates.length,12);assert.equal(proxies.length,6);assert.equal(prior.length,12);assert.equal(supplementary.length,12);
  assert(markdown.includes('## Prior 4B adapters (Appendix Table 6)'));
  assert.equal([...candidates,...proxies].some(row=>/Qwen|Gemma 3/.test(row[0])),false);
  assert.equal(prior.some(row=>/Nothink/.test(row[0])),false);
  assert.equal(supplementary.filter(row=>/Nothink/.test(row[0])).length,6);
  verifyBests([...candidates,...proxies],1,[2,3,4]);verifyBests(prior,1,[2,3,4]);
  assert.equal(proxies.filter(row=>row[2]==='**5.4**').length,2);
  assert.equal(candidates[8][6],'**74,143**');assert.equal(prior[8][6],'**81,313**');
 });

 await t.test('primary and field tables retain measured quality and support',()=>{
  const primary=table(markdown,'## Primary quality comparisons (Table 2)');assert.equal(primary.length,12);
  assert.equal(primary.filter(row=>row.at(-1)==='Yes').length,4);
  const modelNames={'12B JO':'gemma4-12b-jo','12B FT':'gemma4-12b-ft','E4B JO':'gemma4-e4b-jo','E4B FT':'gemma4-e4b-ft'};
  for(const row of primary){
   const parts=row[0].split(' '),precision=parts.pop(),model=modelNames[parts.join(' ')];
   const cell=q.cells.find(r=>r.model===model&&r.precision===precision&&r.selection==='validation_f1');assert(cell);
   assert.equal(number(row[1]),cell.f1_pct);assert.equal(number(row[2]),cell.raw_exact_pct);assert.equal(number(row[3]),cell.delta_f1_pp);
   assert.deepEqual(JSON.parse(row[4]),cell.ci95_f1_pp);assert.equal(row.at(-1)==='Yes',cell.meets_adjusted_margin);
  }
  const fields=table(markdown,'## Field-level quality (Table 7)');assert.equal(fields.length,10);
  for(const row of fields){
   assert.equal(number(row[1]),q.production.gold_nonempty_counts[row[0]]);assert.equal(number(row[2]),q.production.per_field_f1_pct[row[0]]);
   const expected=['gemma4-12b-jo','gemma4-e4b-jo'].flatMap(model=>['bf16','W4A16'].map(precision=>q.cells.find(r=>r.model===model&&r.precision===precision&&r.selection==='validation_f1').per_field_f1_pct[row[0]]));
   assert.deepEqual(row.slice(3).map(number),expected);
  }
  assert.equal(fields.find(row=>row[0]==='COUNTRY')[1],'8');
 });

 await t.test('selection table preserves retained contrasts instead of subtracting rounded F1',()=>{
  const rows=table(markdown,'## Checkpoint selection (Table 1)');assert.equal(rows.length,16);
  assert.equal(rows.filter(row=>row[1]==='Final checkpoint').length,4);
  assert.equal(rows.some(row=>/seed 43/.test(row[0])),false);
  const oracle=rows.find(row=>row[0]==='Gemma 4 E4B JO'&&row[1]==='Test oracle (descriptive)');
  assert.equal(oracle[4],'95.99');assert.equal(oracle[5],'-0.95');
 });

 await t.test('the second seed is a separate eight-row comparison with all twelve exported checkpoints',()=>{
  const rows=table(markdown,'## Second training seed (Table 5)');assert.equal(rows.length,8);
  assert.equal(rows.filter(row=>row[0]==='Gemma 4 E4B JO (seed 42)').length,4);
  const replicateRows=rows.filter(row=>row[0]==='Gemma 4 E4B JO (seed 43)');assert.equal(replicateRows.length,4);
  assert.deepEqual(replicateRows.map(row=>[row[1],...row.slice(2).map(number)]),[
   ['Validation F1',1200,96.87,95.97,-.98,0],
   ['Validation loss',800,96.39,94.96,-1.99,1.01],
   ['Final checkpoint',2298,96.78,95.01,-1.94,.96],
   ['Test oracle (descriptive)',1200,96.87,95.97,-.98,0]
  ]);
  assert(markdown.includes('exploratory unadjusted paired-record 95% CI [-1.68, -0.29]'));
  assert(markdown.includes('record uncertainty, not variation across training seeds'));
  const csv=csvRecords(output['checkpoints.csv']);assert.equal(csv.length,60);
  assert.equal(csv.filter(row=>row.seed==='42'&&row.replicate_of==='').length,48);
  assert.equal(csv.filter(row=>row.seed==='43'&&row.replicate_of==='gemma4-e4b-jo').length,12);
  for(const run of [...checkpoints.runs,...checkpoints.replicates]){
   const exported=csv.filter(row=>row.model===run.label);assert.equal(exported.length,12);
   for(const row of exported){
    const point=run.curve.find(value=>value.step===Number(row.step));assert(point);
    assert.equal(Number(row.seed),run.seed??42);assert.equal(row.replicate_of,run.replicate_of??'');
    assert.equal(Number(row.validation_strict_record_exact_count),point.validation_strict_record_exact_count);
    assert.equal(Number(row.test_F1_percent),point.test_f1_pct);
    assert.equal(row.selected_by_val_F1==='true',point.step===run.selected_by_validation_f1);
    assert.equal(row.selected_by_val_loss==='true',point.step===run.selected_by_validation_loss);
    assert.equal(row.test_oracle==='true',point.step===run.test_oracle_step);
    assert.equal(row.final_checkpoint==='true',point.step===run.final_checkpoint.step);
   }
   assert(output['checkpoint-sweeps.svg'].includes(run.label),`Missing checkpoint panel: ${run.label}`);
  }
  assert.equal(csvRecords(output['selection-stability.csv']).length,8);
 });

 await t.test('decision guide keeps its six-row scope and full-workload resource estimates',()=>{
  const rows=table(markdown,'## Decision guide (Table 4)');assert.equal(rows.length,6);
  verifyBests(rows,2,[4,5,7,8]);
  const llama=rows.find(row=>row[1]==='Llama 8B JO W4A16');assert.equal(llama[4],'**5.4**');assert.equal(llama[7],'14.0');
  const e4b=rows.find(row=>row[1]==='E4B JO W4A16');assert.equal(e4b[6],'**74,143**');assert.equal(e4b[7],'**12.0**');assert.equal(e4b[8],'**0.7**');
  assert.equal(output['selection-stability.csv'].trimEnd().split('\n').length,9);
  assert.equal(output['error-analysis.csv'].trimEnd().split('\n').length,19);
  assert.equal(output['serving-resources.csv'].trimEnd().split('\n').length,211);
 });
});
