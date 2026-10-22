import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {FIELDS,canonical,parseOutput,score,compare,quantile,seed42Generator} from '../lib/evaluation.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const rows=p=>fs.readFileSync(path.join(root,p),'utf8').trim().split('\n').map(JSON.parse);
const q=read('aggregates/quality.json'),s=read('aggregates/serving.json'),h=read('aggregates/hosted.json'),c=read('aggregates/checkpoints.json'),config=read('configs/evaluation.json');
const finite=x=>assert(Number.isFinite(x),'Expected finite number');
const bounds=(x,lo,hi)=>{finite(x);assert(x>=lo&&x<=hi);};
assert.equal(q.cells.length,41);assert.equal(q.production.test_records,1213);assert.equal(q.production.f1_pct,96.95);
const unique=new Set();for(const r of q.cells){const id=[r.model,r.selection,r.precision].join('|');assert(!unique.has(id));unique.add(id);assert.equal(r.test_records,1213);bounds(r.f1_pct,0,100);bounds(r.raw_exact_pct,0,100);r.ci95_f1_pp.forEach(finite);assert(r.ci95_f1_pp[0]<=r.ci95_f1_pp[1]);for(const field of FIELDS){bounds(r.per_field_f1_pct[field],0,100);finite(r.per_field_delta_pp[field]);}
 if(r.adjusted_lower_bound_pp!==null){finite(r.adjusted_lower_bound_pp);assert(r.adjusted_lower_bound_pp<=r.ci95_f1_pp[0]+0.01);assert.equal(r.meets_adjusted_margin,r.adjusted_lower_bound_pp>-config.bootstrap.margin_f1_pp);}else assert.equal(r.meets_adjusted_margin,null);
}
const primary=q.cells.filter(r=>r.selection==='validation_f1');assert.equal(primary.length,12);assert.equal(primary.filter(r=>r.meets_adjusted_margin).length,4);assert(primary.filter(r=>r.meets_adjusted_margin).every(r=>r.model.startsWith('gemma4-12b')));
assert.equal(h.settings.length,4);for(const r of h.settings){assert.equal(r.test_records,1213);bounds(r.f1_pct,0,100);assert(r.adjusted_lower_bound_pp<-1);assert(r.adjusted_lower_bound_pp<r.ci95_f1_pp[0]);}
assert.equal(c.runs.length,4);for(const r of c.runs){assert.equal(r.curve.length,12);assert.equal(new Set(r.curve.map(p=>p.step)).size,12);const loss=Math.min(...r.curve.map(p=>p.validation_loss)),vf=Math.max(...r.curve.map(p=>p.validation_f1_pct)),tf=Math.max(...r.curve.map(p=>p.test_f1_pct));assert.equal(r.curve.find(p=>p.step===r.selected_by_validation_loss).validation_loss,loss);assert.equal(r.curve.find(p=>p.step===r.selected_by_validation_f1).validation_f1_pct,vf);assert.equal(r.curve.find(p=>p.step===r.test_oracle_step).test_f1_pct,tf);assert(r.validation_f1_test_gain_pp>0);finite(r.training_runtime_seconds);const p=q.cells.find(x=>x.model===r.model&&x.selection==='validation_f1'&&x.precision==='bf16');assert.equal(p.f1_pct,r.primary_rerun_f1_pct);}
assert.equal(s.blocks.length,33);assert.equal(s.blocks.filter(r=>r.role==='candidate').length,27);assert.equal(s.blocks.filter(r=>r.role==='proxy').length,6);
for(const r of s.blocks){assert(r.complete);assert.equal(r.repetitions,3);assert.deepEqual(r.points.map(p=>p.concurrency),[1,8,16,32,64]);assert.deepEqual(r.points.map(p=>p.requests),[16,24,48,96,192]);assert(s.blocks.some(p=>p.role==='proxy'&&p.block===r.block&&p.prompt===r.prompt));for(const stat of [r.loaded_weights_gib,...r.points.flatMap(p=>[p.latency_p50_seconds,p.latency_p95_seconds,p.rows_per_hour])]){assert.equal(stat.n,3);finite(stat.min);finite(stat.max);finite(stat.median);assert(stat.min<=stat.median&&stat.median<=stat.max);}for(const p of r.points){assert(p.complete);assert.equal(p.valid_repetitions,3);assert.equal(p.errors,0);}}
// Meaningful scoring checks: normalization, missing denominators, invalid schemas,
// literal parsing, duplicate/unknown IDs, paired equality and unrounded decisions.
assert.equal(canonical('a *B,-'),'AB');assert.equal(quantile([0,10],.25),2.5);
const literal="{'MERCHANT_NAME': 'SYNTHETIC SHOP', 'CITY': None}";assert.equal(parseOutput(literal).status,'schema_invalid');assert.equal(parseOutput('prefix ```json\n{"MERCHANT_NAME":"SYNTHETIC SHOP"}\n```').values.MERCHANT_NAME,'SYNTHETIC SHOP');assert.equal(parseOutput('<think>ignored</think>{"MERCHANT_NAME":"SYNTHETIC SHOP"}').values.MERCHANT_NAME,'SYNTHETIC SHOP');assert.equal(parseOutput('not JSON').values,null);assert.throws(()=>parseOutput('{"MERCHANT_NAME":12}'));
const gold=rows('fixtures/gold.jsonl'),a=rows('fixtures/model-a.jsonl'),b=rows('fixtures/model-b.jsonl');
assert(gold.every(r=>r.id.startsWith('synthetic-')));const scored=score(gold,a);assert.equal(scored.records,6);assert.equal(scored.overall.tp,9);assert.equal(scored.overall.gold,10);assert.equal(scored.overall.pred,9);assert.equal(scored.overall.f1_pct,1800/19);assert.equal(scored.overall.raw_exact_pct,400/6);assert.equal(scored.canonical_exact_pct,500/6);
assert.equal(score(gold,[]).missing_or_null,6);assert.equal(score(gold,[]).overall.recall_pct,0);assert.equal(score(gold,[]).overall.raw_exact_pct,0);assert.throws(()=>score(gold,[a[0],a[0]]));assert.throws(()=>score(gold,[{id:'outside',output:'{}'}]));
const same=compare(gold,a,a,{resamples:2000,familySize:9});assert.equal(same.delta_f1_pp,0);assert.deepEqual(same.ci95_f1_pp,[0,0]);assert.equal(same.noninferiority.lower_bound_pp,0);assert(same.noninferiority.passed);const example=compare(gold,a,b,{resamples:50000,familySize:9});assert.equal(example.metrics_a.f1_pct,1800/19);assert.equal(example.metrics_b.f1_pct,1600/18);assert.equal(example.bootstrap.seed,42);
const random=seed42Generator(10);assert.deepEqual(Array.from({length:10},random),[0,7,6,4,4,8,0,6,2,0]);
// Public aggregate objects must not grow private provenance or row-level fields.
const forbiddenKeys=new Set(['source_files','source_file','sha256','input_sha256','gold_rows','predictions','output','target','descriptor','input','record_id','email','hostname']);
function audit(value){if(Array.isArray(value)){value.forEach(audit);return;}if(value&&typeof value==='object'){for(const[k,v]of Object.entries(value)){assert(!forbiddenKeys.has(k),'Disallowed aggregate field: '+k);audit(v);}}else if(typeof value==='string'){assert(!/\/(?:Users|home|Workspace)\//i.test(value),'Absolute private path');assert(!/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i.test(value),'Email');assert(!/\b[0-9a-f]{64}\b/i.test(value),'Private digest');}}
for(const value of[q,s,h,c])audit(value);
console.log('PASS: 41 quality cells; 4 checkpoint sweeps; 4 hosted settings; 33 serving rows / 165 concurrency points; paired-scoring synthetic checks; aggregate privacy schema.');
