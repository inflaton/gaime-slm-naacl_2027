import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {FIELDS,canonical,parseOutput,score,compare,quantile,seed42Generator} from '../lib/evaluation.mjs';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const rows=p=>fs.readFileSync(path.join(root,p),'utf8').trim().split('\n').map(JSON.parse);
const q=read('aggregates/quality.json'),s=read('aggregates/serving.json'),h=read('aggregates/hosted.json'),c=read('aggregates/checkpoints.json'),config=read('configs/evaluation.json');
const paired=read('aggregates/paired-comparisons.json');
const finite=x=>assert(Number.isFinite(x),'Expected finite number');
const bounds=(x,lo,hi)=>{finite(x);assert(x>=lo&&x<=hi);};
assert.equal(q.cells.length,48);assert.equal(q.production.test_records,1213);assert.equal(q.production.f1_pct,96.95);
const unique=new Set();for(const r of q.cells){const id=[r.model,r.selection,r.precision].join('|');assert(!unique.has(id));unique.add(id);assert.equal(r.test_records,1213);bounds(r.f1_pct,0,100);bounds(r.raw_exact_pct,0,100);r.ci95_f1_pp.forEach(finite);assert(r.ci95_f1_pp[0]<=r.ci95_f1_pp[1]);for(const field of FIELDS){bounds(r.per_field_f1_pct[field],0,100);finite(r.per_field_delta_pp[field]);}
 if(r.adjusted_lower_bound_pp!==null){finite(r.adjusted_lower_bound_pp);assert(r.adjusted_lower_bound_pp<=r.ci95_f1_pp[0]+0.01);assert.equal(r.meets_adjusted_margin,r.adjusted_lower_bound_pp>-config.bootstrap.margin_f1_pp);}else assert.equal(r.meets_adjusted_margin,null);
}
const primary=q.cells.filter(r=>r.selection==='validation_f1');assert.equal(primary.length,12);assert.equal(primary.filter(r=>r.meets_adjusted_margin).length,4);assert(primary.filter(r=>r.meets_adjusted_margin).every(r=>r.model.startsWith('gemma4-12b')));
assert.equal(q.cells.filter(r=>r.selection==='validation_loss').length,12);
assert.equal(q.cells.filter(r=>r.selection==='retained_test_selected').length,18);
assert.equal(q.cells.filter(r=>r.selection==='proxy').length,6);
for(const model of ['gemma3-4b-jo','gemma3-4b-ft','llama31-8b-proxy-jo','llama31-8b-proxy-ft'])assert.deepEqual(q.cells.filter(r=>r.model===model).map(r=>r.precision),['bf16','fp8','W4A16']);
assert.equal(h.settings.length,4);for(const r of h.settings){assert.equal(r.test_records,1213);bounds(r.f1_pct,0,100);assert(r.adjusted_lower_bound_pp<-1);assert(r.adjusted_lower_bound_pp<r.ci95_f1_pp[0]);}
assert.equal(c.schema_version,2);
assert.equal(c.runs.length,4);for(const r of c.runs){assert.equal(r.curve.length,12);assert.equal(new Set(r.curve.map(p=>p.step)).size,12);const loss=Math.min(...r.curve.map(p=>p.validation_loss)),vf=Math.max(...r.curve.map(p=>p.validation_f1_pct)),tf=Math.max(...r.curve.map(p=>p.test_f1_pct));assert.equal(r.curve.find(p=>p.step===r.selected_by_validation_loss).validation_loss,loss);assert.equal(r.curve.find(p=>p.step===r.selected_by_validation_f1).validation_f1_pct,vf);assert.equal(r.curve.find(p=>p.step===r.test_oracle_step).test_f1_pct,tf);assert(r.validation_f1_test_gain_pp>0);finite(r.training_runtime_seconds);const p=q.cells.find(x=>x.model===r.model&&x.selection==='validation_f1'&&x.precision==='bf16');assert.equal(p.f1_pct,r.serving_evaluation_f1_pct);}
assert.equal(s.blocks.length,42);assert.equal(s.blocks.filter(r=>r.role==='candidate').length,30);assert.equal(s.blocks.filter(r=>r.role==='proxy').length,12);
assert.equal(new Set(s.blocks.map(r=>[r.block,r.model,r.precision].join('|'))).size,s.blocks.length);
for(const r of s.blocks){assert(r.complete);assert.equal(r.repetitions,3);assert.deepEqual(r.points.map(p=>p.concurrency),[1,8,16,32,64]);assert.deepEqual(r.points.map(p=>p.requests),[16,24,48,96,192]);assert.equal(s.blocks.filter(p=>p.role==='proxy'&&p.block===r.block&&p.prompt===r.prompt&&p.precision==='bf16').length,1);assert(q.cells.some(p=>p.model===r.model&&p.selection===r.selection&&p.precision===r.precision));for(const stat of [r.loaded_weights_gib,...r.points.flatMap(p=>[p.latency_p50_seconds,p.latency_p95_seconds,p.rows_per_hour])]){assert.equal(stat.n,3);finite(stat.min);finite(stat.max);finite(stat.median);assert(stat.min<=stat.median&&stat.median<=stat.max);}for(const p of r.points){assert(p.complete);assert.equal(p.valid_repetitions,3);assert.equal(p.errors,0);}}
// Paired quality estimates come from retained record-level comparisons, not
// subtraction of rounded published F1. Serving ratios use explicit block IDs.
assert.equal(paired.comparisons.length,12);assert.equal(paired.bootstrap.resamples,50000);assert.equal(paired.bootstrap.seed,42);
assert.equal(new Set(paired.comparisons.map(r=>[r.model,r.precision].join('|'))).size,12);
const point=(r,concurrency,metric)=>r.points.find(p=>p.concurrency===concurrency)[metric].median;
const close=(a,b)=>assert(Math.abs(a-b)<1e-10,`Inconsistent aggregate ratio: ${a} versus ${b}`);
for(const r of paired.comparisons){
 assert.equal(r.test_records,1213);assert.equal(r.selection,'validation_f1');r.ci95_f1_pp.forEach(finite);finite(r.delta_f1_pp);assert(r.ci95_f1_pp[0]<=r.delta_f1_pp&&r.delta_f1_pp<=r.ci95_f1_pp[1]);
 const qa=q.cells.find(p=>p.model===r.model&&p.selection===r.selection&&p.precision===r.precision),qb=q.cells.find(p=>p.model===r.reference_model&&p.selection==='proxy'&&p.precision===r.precision);assert(qa&&qb);assert.equal(qa.prompt,qb.prompt);assert.equal(qa.prompt,r.prompt);assert(Math.abs(qa.f1_pct-qb.f1_pct-r.delta_f1_pp)<0.011);
 const a=s.blocks.find(p=>p.block===r.model_benchmark_block&&p.model===r.model&&p.precision===r.precision),b=s.blocks.find(p=>p.block===r.reference_benchmark_block&&p.model===r.reference_model&&p.precision===r.precision);assert(a&&b);
 close(r.weights_model_over_reference,a.loaded_weights_gib.median/b.loaded_weights_gib.median);close(r.latency_speedup_reference_over_model,point(b,1,'latency_p50_seconds')/point(a,1,'latency_p50_seconds'));close(r.throughput_model_over_reference,point(a,64,'rows_per_hour')/point(b,64,'rows_per_hour'));
}
assert.equal(paired.comparisons.find(r=>r.model==='gemma4-12b-jo'&&r.precision==='W4A16').delta_f1_pp,1.03);
assert.deepEqual(paired.comparisons.find(r=>r.model==='gemma4-e4b-jo'&&r.precision==='W4A16').ci95_f1_pp,[-1.51,0.1]);
assert.equal(paired.block_consistency.length,4);
for(const r of paired.block_consistency){
 const a=s.blocks.find(p=>p.block===r.model_benchmark_block&&p.role==='proxy'&&p.precision==='bf16'),b=s.blocks.find(p=>p.block===r.reference_benchmark_block&&p.model===a.model&&p.precision==='bf16');assert(a&&b);
 close(r.weights_reference_block_over_model_block,b.loaded_weights_gib.median/a.loaded_weights_gib.median);close(r.latency_reference_block_over_model_block,point(b,1,'latency_p50_seconds')/point(a,1,'latency_p50_seconds'));close(r.throughput_reference_block_over_model_block,point(b,64,'rows_per_hour')/point(a,64,'rows_per_hour'));
}
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
for(const value of[q,s,h,c,paired])audit(value);
console.log('PASS: 48 quality cells; 4 checkpoint sweeps; 4 hosted settings; 42 serving rows / 210 concurrency points; 12 paired contrasts; paired-scoring synthetic checks; aggregate privacy schema.');
