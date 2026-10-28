import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath} from 'node:url';
import {FIELDS,canonical,parseOutput,score,compare,quantile,seed42Generator} from '../lib/evaluation.mjs';
const artifactRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');

// Scan public JSON recursively so new reports and configuration files cannot
// silently bypass the release check. Template column names are metadata, not rows.
const privateKeys=new Set(['source_files','source_file','sha256','input_sha256','record_id','record_ids','descriptor','descriptors','email','hostname']);
const aggregateRowKeys=new Set(['gold_rows','predictions','output','target','input']);
export function auditPublicValue(value,{kind='aggregate',location='public JSON'}={}){
 if(Array.isArray(value)){value.forEach((entry,index)=>auditPublicValue(entry,{kind,location:`${location}[${index}]`}));return;}
 if(value&&typeof value==='object'){
  for(const[key,entry]of Object.entries(value)){
   assert(!privateKeys.has(key.toLowerCase())&&!(kind==='aggregate'&&aggregateRowKeys.has(key.toLowerCase())),`Disallowed public field at ${location}.${key}`);
   auditPublicValue(entry,{kind,location:`${location}.${key}`});
  }
 }else if(typeof value==='string'){
  assert(!/\/(?:Users|home|Workspace|private|tmp|root|mnt)\//i.test(value),`Absolute private path at ${location}`);
  assert(!/(?:^|[\s"'(])[A-Z]:[\\/]|\\\\[^\\\s]+\\/i.test(value),`Absolute private path at ${location}`);
  assert(!/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i.test(value),`Email at ${location}`);
  assert(!/\b[0-9a-f]{64}\b/i.test(value),`Private digest at ${location}`);
 }
}
export function auditPublicFiles(root=artifactRoot){
 const audited=[];
 function walk(directory,kind){
  for(const entry of fs.readdirSync(directory,{withFileTypes:true}).sort((a,b)=>a.name.localeCompare(b.name))){
   const file=path.join(directory,entry.name);
   assert(!entry.isSymbolicLink(),`Public JSON directory must not contain a symlink: ${path.relative(root,file)}`);
   if(entry.isDirectory())walk(file,kind);
   else if(entry.isFile()&&entry.name.endsWith('.json')){
    const relative=path.relative(root,file);auditPublicValue(JSON.parse(fs.readFileSync(file,'utf8')),{kind,location:relative});audited.push(relative);
   }
  }
 }
 walk(path.join(root,'aggregates'),'aggregate');walk(path.join(root,'configs'),'config');return audited;
}

export function validateArtifact(root=artifactRoot){
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const rows=p=>fs.readFileSync(path.join(root,p),'utf8').trim().split('\n').map(JSON.parse);
const q=read('aggregates/quality.json'),s=read('aggregates/serving.json'),h=read('aggregates/hosted.json'),c=read('aggregates/checkpoints.json'),config=read('configs/evaluation.json');
const paired=read('aggregates/paired-comparisons.json');
const stability=read('aggregates/selection-stability.json'),errors=read('aggregates/error-analysis.json');
const servingConfig=read('configs/serving.json'),trainingConfig=read('configs/training.json');
const finite=x=>assert(Number.isFinite(x),'Expected finite number');
const bounds=(x,lo,hi)=>{finite(x);assert(x>=lo&&x<=hi);};
const count=(x,maximum=Infinity)=>{assert(Number.isInteger(x),'Expected an integer count');bounds(x,0,maximum);};
const close=(a,b,tolerance=1e-10)=>{finite(a);finite(b);assert(Math.abs(a-b)<=tolerance,`Inconsistent aggregate value: ${a} versus ${b}`);};
const summary=(stat,n=3)=>{assert.equal(stat.n,n);for(const key of ['min','median','max']){finite(stat[key]);assert(stat[key]>=0);}assert(stat.min<=stat.median&&stat.median<=stat.max);};
const summarize=values=>{const sorted=[...values].sort((a,b)=>a-b);return {min:sorted[0],median:quantile(sorted,.5),max:sorted.at(-1),n:sorted.length};};
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
assert.equal(c.runs.length,4);
for(const r of c.runs){
 assert.equal(r.curve.length,12);assert.equal(new Set(r.curve.map(p=>p.step)).size,12);
 for(const p of r.curve){
  bounds(p.validation_f1_pct,0,100);
  assert(Math.abs(p.validation_f1_pct*100-Math.round(p.validation_f1_pct*100))<1e-8,'Validation F1 must retain the original four-decimal fractional precision');
  assert(Number.isInteger(p.validation_strict_record_exact_count),'Missing strict-match tie-break count');
  bounds(p.validation_strict_record_exact_count,0,config.sample_counts.validation);
 }
 // These percentages already encode the original scorer's rounding. Keep full
 // precision in the general-purpose scorer; rank the retained selection values.
 const valPick=[...r.curve].sort((a,b)=>b.validation_f1_pct-a.validation_f1_pct||b.validation_strict_record_exact_count-a.validation_strict_record_exact_count||a.step-b.step)[0];
 const lossPick=[...r.curve].sort((a,b)=>a.validation_loss-b.validation_loss||a.step-b.step)[0];
 assert.equal(r.selected_by_validation_f1,valPick.step,'Inconsistent rounded-F1/tie-break selection');
 assert.equal(r.selected_by_validation_loss,lossPick.step);
 const tf=Math.max(...r.curve.map(p=>p.test_f1_pct));assert.equal(r.curve.find(p=>p.step===r.test_oracle_step).test_f1_pct,tf);
 assert(r.validation_f1_test_gain_pp>0);finite(r.training_runtime_seconds);
 const p=q.cells.find(x=>x.model===r.model&&x.selection==='validation_f1'&&x.precision==='bf16');assert.equal(p.f1_pct,r.serving_evaluation_f1_pct);
 const last=r.curve.reduce((a,b)=>a.step>b.step?a:b),final=r.final_checkpoint;
 assert.equal(last.step,trainingConfig.final_step);assert.equal(final.step,last.step);
 for(const [key,value]of Object.entries(last))assert.equal(final[key],value,`Final checkpoint differs from the retained sweep: ${r.model}.${key}`);
 close(final.delta_vs_production_pp,last.test_f1_pct-q.production.f1_pct);
 close(final.final_minus_loss_selected_f1_pp,last.test_f1_pct-lossPick.test_f1_pct);
 close(final.validation_selected_minus_final_f1_pp,valPick.test_f1_pct-last.test_f1_pct);
 close(final.oracle_minus_final_f1_pp,tf-last.test_f1_pct);
 assert.deepEqual(Object.keys(r.selection_summary),['validation_f1','validation_loss','test_oracle']);
 for(const [rule,step]of [['validation_f1',r.selected_by_validation_f1],['validation_loss',r.selected_by_validation_loss],['test_oracle',r.test_oracle_step]]){
  const retained=r.selection_summary[rule],checkpoint=r.curve.find(p=>p.step===step);
  assert.equal(retained.step,step,'Selection summary must identify its retained checkpoint');
  assert.equal(retained.validation_f1_pct,checkpoint.validation_f1_pct);assert.equal(retained.test_f1_pct,checkpoint.test_f1_pct);
  // These are separately retained display contrasts; subtraction of rounded F1
  // may differ by 0.01 pp, unlike the final-checkpoint arithmetic above.
  close(retained.delta_vs_production_pp,checkpoint.test_f1_pct-q.production.f1_pct,.011);
  close(retained.oracle_gap_pp,tf-checkpoint.test_f1_pct,.011);
 }
}
assert.equal(stability.schema_version,1);assert.equal(stability.runs.length,c.runs.length);
assert.equal(new Set(stability.runs.map(r=>r.model)).size,c.runs.length);
assert.equal(stability.validation_records,config.sample_counts.validation);
assert.equal(stability.resamples,config.selection.validation_resampling.bootstrap.resamples);
assert.equal(stability.resamples,config.selection.validation_resampling.random_halves.resamples);
assert.equal(stability.seed,config.selection.validation_resampling.seed);
for(const r of stability.runs){
 const checkpoints=c.runs.find(p=>p.model===r.model);assert(checkpoints);
 assert.deepEqual(r.available_steps,checkpoints.curve.map(p=>p.step));
 assert.deepEqual(r.test_f1_pct_by_step,Object.fromEntries(checkpoints.curve.map(p=>[p.step,p.test_f1_pct])));
 assert.equal(r.validation_loss_step,checkpoints.selected_by_validation_loss);
 assert.equal(r.final_step,checkpoints.final_checkpoint.step);
 assert.equal(r.full_validation_pick_step,checkpoints.selected_by_validation_f1);
 assert.equal(r.reported_validation_pick_step,checkpoints.selected_by_validation_f1);
 for(const scheme of ['bootstrap','random_half']){
  const result=r[scheme],entries=Object.entries(result.selected_step_counts);let total=0;
  for(const [step,n]of entries){assert(r.available_steps.includes(Number(step)));count(n,stability.resamples);total+=n;}
  assert.equal(total,stability.resamples,'Selection frequencies must cover every resample');
  const fraction=predicate=>entries.reduce((sum,[step,n])=>sum+(predicate(Number(step))?n:0),0)/total;
  close(result.same_as_full_pick_fraction,fraction(step=>step===r.full_validation_pick_step));
  close(result.beats_validation_loss_fraction,fraction(step=>r.test_f1_pct_by_step[step]>r.test_f1_pct_by_step[r.validation_loss_step]));
  close(result.matches_or_beats_final_fraction,fraction(step=>r.test_f1_pct_by_step[step]>=r.test_f1_pct_by_step[r.final_step]));
  close(result.beats_final_fraction,fraction(step=>r.test_f1_pct_by_step[step]>r.test_f1_pct_by_step[r.final_step]));
  const selected=entries.flatMap(([step,n])=>Array(n).fill(r.test_f1_pct_by_step[step])).sort((a,b)=>a-b);
  const mean=selected.reduce((sum,value)=>sum+value,0)/total;
  close(result.test_f1_pct.mean,mean,.0005+1e-10);
  for(const [key,p]of [['p05',.05],['p50',.5],['p95',.95]])close(result.test_f1_pct[key],quantile(selected,p),.0005+1e-10);
  close(result.mean_oracle_regret_pp,Math.max(...Object.values(r.test_f1_pct_by_step))-mean,.0005+1e-10);
 }
}
assert.equal(s.blocks.length,42);assert.equal(s.blocks.filter(r=>r.role==='candidate').length,30);assert.equal(s.blocks.filter(r=>r.role==='proxy').length,12);
assert.equal(new Set(s.blocks.map(r=>[r.block,r.model,r.precision].join('|'))).size,s.blocks.length);
for(const r of s.blocks){assert(r.complete);assert.equal(r.repetitions,3);assert.deepEqual(r.points.map(p=>p.concurrency),[1,8,16,32,64]);assert.deepEqual(r.points.map(p=>p.requests),[16,24,48,96,192]);assert.equal(s.blocks.filter(p=>p.role==='proxy'&&p.block===r.block&&p.prompt===r.prompt&&p.precision==='bf16').length,1);assert(q.cells.some(p=>p.model===r.model&&p.selection===r.selection&&p.precision===r.precision));for(const stat of [r.loaded_weights_gib,...r.points.flatMap(p=>[p.latency_p50_seconds,p.latency_p95_seconds,p.rows_per_hour])]){assert.equal(stat.n,3);finite(stat.min);finite(stat.max);finite(stat.median);assert(stat.min<=stat.median&&stat.median<=stat.max);}for(const p of r.points){assert(p.complete);assert.equal(p.valid_repetitions,3);assert.equal(p.errors,0);}}
for(const r of s.blocks){
 summary(r.kv_cache_capacity_tokens);for(const key of ['min','median','max'])count(r.kv_cache_capacity_tokens[key]);
 for(const p of r.points)for(const key of ['output_tokens_per_second','mean_output_tokens','mean_gpu_power_watts','gpu_energy_wh_per_request'])summary(p[key]);
 const workload=r.full_workload,resourceConfig=servingConfig.benchmark.resource_estimation;
 assert.equal(workload.concurrency,resourceConfig.concurrency);assert.equal(workload.requests,resourceConfig.requests);
 assert.equal(workload.repetitions,r.repetitions);assert.equal(workload.repeat_measurements.length,r.repetitions);
 assert.deepEqual(workload.repeat_measurements.map(p=>p.repetition),[1,2,3]);
 for(const repeat of workload.repeat_measurements){
  assert.equal(repeat.requests,workload.requests);assert(repeat.wall_seconds>0);assert(repeat.mean_gpu_power_watts>0);
  close(repeat.device_hours_per_million,1e6*repeat.wall_seconds/(3600*repeat.requests));
  close(repeat.gpu_energy_kwh_per_million,repeat.device_hours_per_million*repeat.mean_gpu_power_watts/1000);
 }
 for(const key of ['wall_seconds','device_hours_per_million','gpu_energy_kwh_per_million']){
  summary(workload[key]);const expected=summarize(workload.repeat_measurements.map(p=>p[key]));
  for(const statistic of ['min','median','max','n'])close(workload[key][statistic],expected[statistic]);
 }
 const power=summarize(workload.repeat_measurements.map(p=>p.mean_gpu_power_watts));
 assert.deepEqual(power,r.points.find(p=>p.concurrency===workload.concurrency).mean_gpu_power_watts);
}
// Paired quality estimates come from retained record-level comparisons, not
// subtraction of rounded published F1. Serving ratios use explicit block IDs.
assert.equal(paired.comparisons.length,12);assert.equal(paired.bootstrap.resamples,50000);assert.equal(paired.bootstrap.seed,42);
assert.equal(new Set(paired.comparisons.map(r=>[r.model,r.precision].join('|'))).size,12);
const point=(r,concurrency,metric)=>r.points.find(p=>p.concurrency===concurrency)[metric].median;
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
assert.equal(errors.schema_version,1);assert.equal(errors.reference_model,'production');
assert.deepEqual(errors.fields.map(r=>r.field),config.error_analysis.fields);
assert.deepEqual(Object.keys(errors.taxonomy.definitions),errors.taxonomy.order);
for(const field of errors.fields){
 assert.equal(field.test_records,config.sample_counts.test);
 assert.equal(field.gold_nonempty,q.production.gold_nonempty_counts[field.field]);count(field.gold_in_descriptor,field.gold_nonempty);
 const outcomes=(r,expectedF1)=>{
  for(const key of ['correct','wrong','missed','spurious','both_empty'])count(r[key],field.test_records);
  assert.equal(r.correct+r.wrong+r.missed,field.gold_nonempty);
  assert.equal(r.correct+r.wrong+r.missed+r.spurious+r.both_empty,field.test_records);
  const predicted=r.correct+r.wrong+r.spurious;
  close(r.f1_pct,200*r.correct/(field.gold_nonempty+predicted),.005+1e-10);
  assert.equal(r.f1_pct,expectedF1);
 };
 outcomes(field.production_outcomes,q.production.per_field_f1_pct[field.field]);
 assert.equal(field.comparisons.length,6);
 assert.equal(new Set(field.comparisons.map(r=>`${r.model}|${r.precision}`)).size,6);
 for(const r of field.comparisons){
  const quality=q.cells.find(p=>p.model===r.model&&p.precision===r.precision&&p.selection===r.selection);assert(quality);
  assert.equal(r.prompt,quality.prompt);outcomes(r.outcomes,quality.per_field_f1_pct[field.field]);
  const paired=r.paired;
  for(const key of ['gold_nonempty','both_correct','production_only','model_only','both_incorrect','spurious_model','spurious_production'])count(paired[key],field.test_records);
  assert.equal(paired.gold_nonempty,field.gold_nonempty);
  assert.equal(paired.both_correct+paired.production_only+paired.model_only+paired.both_incorrect,field.gold_nonempty);
  assert.equal(paired.both_correct+paired.production_only,field.production_outcomes.correct);
  assert.equal(paired.both_correct+paired.model_only,r.outcomes.correct);
  assert.equal(paired.net_true_positives,paired.model_only-paired.production_only);
  assert.equal(paired.spurious_model,r.outcomes.spurious);assert.equal(paired.spurious_production,field.production_outcomes.spurious);bounds(paired.sign_test_p,0,1);
  for(const [key,total]of [['loss_type_counts',paired.production_only],['production_loss_type_counts',paired.model_only]]){
   assert.deepEqual(Object.keys(r[key]),errors.taxonomy.order);
   Object.values(r[key]).forEach(n=>count(n,total));assert.equal(Object.values(r[key]).reduce((sum,n)=>sum+n,0),total,'Error categories must partition their corresponding losses');
  }
  const presence=r.descriptor_presence_counts;assert.equal(presence.losses,paired.production_only);assert.equal(presence.wins,paired.model_only);
  count(presence.losses_gold_in_descriptor,presence.losses);count(presence.losses_prediction_in_descriptor,presence.losses);count(presence.wins_gold_in_descriptor,presence.wins);
  if(r.precision==='bf16')assert.equal(r.precision_overlap,null);
  else{
   const overlap=r.precision_overlap;assert.equal(overlap.reference_precision,'bf16');
   const reference=field.comparisons.find(p=>p.model===r.model&&p.precision===overlap.reference_precision);assert(reference);
   for(const key of ['shared_loss_records','identical_loss_outputs','new_loss_records','resolved_bf16_loss_records'])count(overlap[key],field.test_records);
   assert(overlap.identical_loss_outputs<=overlap.shared_loss_records);
   assert.equal(overlap.shared_loss_records+overlap.new_loss_records,paired.production_only);
   assert.equal(overlap.shared_loss_records+overlap.resolved_bf16_loss_records,reference.paired.production_only);
  }
 }
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
const publicFiles=auditPublicFiles(root);
console.log(`PASS: 48 quality cells; 4 checkpoint sweeps and final checkpoints; 8 selection-stability summaries; 18 field-error contrasts; 4 hosted settings; 42 serving workloads / 210 concurrency points; 12 paired contrasts; paired-scoring synthetic checks; privacy checks for ${publicFiles.length} public JSON files.`);
}

if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url))validateArtifact();
