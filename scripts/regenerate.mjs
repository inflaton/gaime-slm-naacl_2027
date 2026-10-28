import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const read=p=>JSON.parse(fs.readFileSync(path.join(root,p),'utf8'));
const q=read('aggregates/quality.json'),s=read('aggregates/serving.json');
const h=read('aggregates/hosted.json'),c=read('aggregates/checkpoints.json');
const paired=read('aggregates/paired-comparisons.json');
const stability=read('aggregates/selection-stability.json'),errors=read('aggregates/error-analysis.json');
const production=read('configs/production-context.json');
const out=path.join(root,'generated');
const write=(name,text)=>fs.writeFileSync(path.join(out,name),text);
const csv=rows=>rows.map(r=>r.map(v=>v==null?'':`"${String(v).replaceAll('"','""')}"`).join(',')).join('\n')+'\n';
// Match the paper's Python fixed-point formatting, including ties-to-even.
// Work from the exact binary float so multiplying by 10 does not create a false tie.
function f(value,d=2){
 if(value==null)return '—';
 const n=Number(value);
 if(!Number.isFinite(n))throw new Error(`Non-finite report value: ${value}`);
 const buffer=new ArrayBuffer(8),view=new DataView(buffer);
 view.setFloat64(0,Math.abs(n));
 const bits=view.getBigUint64(0),encodedExponent=Number((bits>>52n)&2047n);
 const significand=(bits&((1n<<52n)-1n))+(encodedExponent===0?0n:1n<<52n);
 const exponent=encodedExponent===0?-1074:encodedExponent-1023-52;
 let numerator=significand*10n**BigInt(d),denominator=1n;
 if(exponent>=0)numerator<<=BigInt(exponent);else denominator<<=BigInt(-exponent);
 let rounded=numerator/denominator;
 const twiceRemainder=2n*(numerator%denominator);
 if(twiceRemainder>denominator||(twiceRemainder===denominator&&rounded%2n===1n))rounded++;
 const digits=String(rounded).padStart(d+1,'0'),sign=n<0||Object.is(n,-0)?'-':'';
 return sign+(d?digits.slice(0,-d)+'.'+digits.slice(-d):digits);
}
const esc=x=>String(x).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
const txt=(x,y,text,opts='')=>`<text x="${x}" y="${y}" ${opts}>${esc(text)}</text>`;
const line=(x1,y1,x2,y2,opts='')=>`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" ${opts}/>`;
const svg=(w,he,body,title)=>`<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${he}" viewBox="0 0 ${w} ${he}" role="img"><title>${esc(title)}</title><style>text{font-family:Arial,sans-serif;font-size:12px;fill:#17202a}line,path{fill:none}text.title{font-size:20px;font-weight:bold}.label{font-size:11px}</style><rect width="100%" height="100%" fill="white"/>${body}</svg>\n`;
const color=p=>({bf16:'#2166ac',fp8:'#d6604d',W4A16:'#1b7837'}[p]??'#666');
const modelColor=model=>model.includes('12b')?'#1f5fa8':'#c8541d';
const precs=['bf16','fp8','W4A16'];
const gemmaModels=['gemma4-12b-jo','gemma4-12b-ft','gemma4-e4b-jo','gemma4-e4b-ft'];
const priorModels=['qwen35-4b-jo-think','qwen35-4b-ft-think','gemma3-4b-jo','gemma3-4b-ft'];
const proxyModels=['llama31-8b-proxy-jo','llama31-8b-proxy-ft'];
const labels={
 'gemma4-12b-jo':'12B JO','gemma4-12b-ft':'12B FT',
 'gemma4-e4b-jo':'E4B JO','gemma4-e4b-ft':'E4B FT',
 'qwen35-4b-jo-think':'Qwen 3.5 4B JO','qwen35-4b-ft-think':'Qwen 3.5 4B FT',
 'gemma3-4b-jo':'Gemma 3 4B JO','gemma3-4b-ft':'Gemma 3 4B FT',
 'llama31-8b-proxy-jo':'Llama 8B JO','llama31-8b-proxy-ft':'Llama 8B FT',
};
function unique(rows,description){
 if(rows.length!==1)throw new Error(`Expected one ${description}; found ${rows.length}`);
 return rows[0];
}
function quality(model,selection,precision){
 return unique(q.cells.filter(r=>r.model===model&&r.selection===selection&&r.precision===precision),`quality cell ${model}/${selection}/${precision}`);
}
// Every denominator is a bf16 proxy from the same prompt and benchmark block.
// The precision test is essential now that the proxy block also contains fp8/W4A16.
const serving=s.blocks.map(r=>{
 const proxy=unique(s.blocks.filter(p=>p.role==='proxy'&&p.precision==='bf16'&&p.block===r.block&&p.prompt===r.prompt),`bf16 proxy for ${r.block}/${r.prompt}`);
 const point=(row,n)=>unique(row.points.filter(p=>p.concurrency===n),`concurrency ${n} for ${row.block}/${row.model}/${row.precision}`);
 const one=point(r,1),max=point(r,64),po=point(proxy,1),pm=point(proxy,64);
 const score=quality(r.model,r.selection,r.precision);
 return{...r,f1_pct:score.f1_pct,memory:r.loaded_weights_gib.median,
  memory_ratio:r.loaded_weights_gib.median/proxy.loaded_weights_gib.median,
  latency:one.latency_p50_seconds.median,speedup:po.latency_p50_seconds.median/one.latency_p50_seconds.median,
  throughput:max.rows_per_hour.median,throughput_ratio:max.rows_per_hour.median/pm.rows_per_hour.median};
});
function servingCell(model,precision,block){
 return unique(serving.filter(r=>r.model===model&&r.precision===precision&&r.block===block),`serving cell ${block}/${model}/${precision}`);
}
const groups=[
 {title:'Gemma 4: validation-F1 checkpoints',rows:gemmaModels.flatMap(model=>precs.map(p=>servingCell(model,p,model)))},
 {title:'Prior 4B adapters: retained test-selected checkpoints',rows:priorModels.flatMap(model=>precs.map(p=>servingCell(model,p,'prior-4b')))},
 {title:'Llama 8B proxies: production architecture at three precisions',rows:proxyModels.flatMap(model=>precs.map(p=>servingCell(model,p,p==='bf16'?(model.endsWith('-jo')?'gemma4-12b-jo':'gemma4-12b-ft'):'llama-quant')))},
];
const mainGroups=[groups[0],groups[2]],priorGroup=groups[1];
const mainRows=mainGroups.flatMap(g=>g.rows),allDisplayed=groups.flatMap(g=>g.rows);
const rowKey=r=>`${r.block}/${r.model}/${r.precision}`;
const displayed=new Set(allDisplayed.map(rowKey));
const supplementary=serving.filter(r=>!displayed.has(rowKey(r)));
const cols=['f1_pct','memory','memory_ratio','latency','speedup','throughput','throughput_ratio'];
const digits=[2,1,2,2,2,0,2];
const rounded=(r,i)=>Number(f(r[cols[i]],digits[i]));
// Each manuscript table ranks all its displayed rows, including Llama proxies.
const bestFor=rows=>cols.map((_,i)=>([1,2,3].includes(i)?Math.min:Math.max)(...rows.map(r=>rounded(r,i))));
const mainBest=bestFor(mainRows),priorBest=bestFor(priorGroup.rows);
const display=(r,i,best=null)=>{
 const n=rounded(r,i);
 let value=i===5?n.toLocaleString('en-US'):f(n,digits[i]);
 if([2,4,6].includes(i))value+='×';
 return best&&n===best[i]?'**'+value+'**':value;
};
for(const r of c.runs)if(!Number.isFinite(r.serving_evaluation_f1_pct))throw new Error(`Missing serving-evaluation F1 for ${r.model}`);
fs.mkdirSync(out,{recursive:true});
write('quality.csv',csv([['model','selection','prompt','precision','F1_percent','raw_exact_percent','delta_F1_pp','CI95_low_pp','CI95_high_pp','adjusted_lower_bound_pp','meets_adjusted_margin'],...q.cells.map(r=>[r.label,r.selection,r.prompt,r.precision,r.f1_pct,r.raw_exact_pct,r.delta_f1_pp,...r.ci95_f1_pp,r.adjusted_lower_bound_pp,r.meets_adjusted_margin])]));
write('hosted.csv',csv([['model','prompt','demonstrations','F1_percent','raw_exact_percent','canonical_exact_percent','delta_F1_pp','CI95_low_pp','CI95_high_pp','supplementary_lower_bound_pp'],...h.settings.map(r=>[h.model,r.prompt,r.demonstrations,r.f1_pct,r.raw_exact_pct,r.canonical_exact_pct,r.delta_f1_pp,...r.ci95_f1_pp,r.adjusted_lower_bound_pp])]));
write('serving.csv',csv([['block','model','precision','selection','F1_percent','weights_GiB','weights_model_over_proxy','latency_p50_c1_seconds','latency_speedup_proxy_over_model','throughput_c64_rows_per_hour','throughput_model_over_proxy'],...serving.map(r=>[r.block,r.label,r.precision,r.selection,r.f1_pct,r.memory,r.memory_ratio,r.latency,r.speedup,r.throughput,r.throughput_ratio])]));
write('concurrency.csv',csv([['block','model','precision','concurrency','requests','repetitions','p50_latency_median_s','p50_latency_min_s','p50_latency_max_s','p95_latency_median_s','throughput_median_rows_h','throughput_min_rows_h','throughput_max_rows_h'],...s.blocks.flatMap(r=>r.points.map(p=>[r.block,r.label,r.precision,p.concurrency,p.requests,p.valid_repetitions,p.latency_p50_seconds.median,p.latency_p50_seconds.min,p.latency_p50_seconds.max,p.latency_p95_seconds.median,p.rows_per_hour.median,p.rows_per_hour.min,p.rows_per_hour.max]))]));
write('checkpoints.csv',csv([['model','step','validation_loss','validation_F1_percent','validation_strict_record_exact_count','test_F1_percent','selected_by_val_F1','selected_by_val_loss','final_checkpoint','test_oracle'],...c.runs.flatMap(r=>r.curve.map(p=>[r.label,p.step,p.validation_loss,p.validation_f1_pct,p.validation_strict_record_exact_count,p.test_f1_pct,p.step===r.selected_by_validation_f1,p.step===r.selected_by_validation_loss,p.step===r.final_checkpoint.step,p.step===r.test_oracle_step]))]));
write('per-field.csv',csv([['model','selection','precision','field','F1_percent','delta_vs_production_pp'],...q.cells.flatMap(r=>Object.entries(r.per_field_f1_pct).map(([field,value])=>[r.label,r.selection,r.precision,field,value,r.per_field_delta_pp[field]]))]));
write('paired-comparisons.csv',csv([['model','reference_model','selection','prompt','precision','paired_test_records','delta_F1_pp','unadjusted_CI95_low_pp','unadjusted_CI95_high_pp','model_benchmark_block','reference_benchmark_block','weights_model_over_reference','latency_speedup_reference_over_model','throughput_model_over_reference'],...paired.comparisons.map(r=>[r.label,r.reference_label,r.selection,r.prompt,r.precision,r.test_records,r.delta_f1_pp,...r.ci95_f1_pp,r.model_benchmark_block,r.reference_benchmark_block,r.weights_model_over_reference,r.latency_speedup_reference_over_model,r.throughput_model_over_reference])]));

const header='| Model / precision | F1 (%) ↑ | Weights (GiB) ↓ | Model/proxy ↓ | p50 c=1 (s) ↓ | Speed-up ↑ | c=64 rows/h ↑ | Model/proxy ↑ |\n|:--|--:|--:|--:|--:|--:|--:|--:|';
let md='# Aggregate results\n\nAll numbers below are retained study aggregates. Synthetic demonstration outputs are separate. Quality comparisons use 1,213 paired records. F1 and exact-match rates are percentages; differences are percentage points.\n\n## Main serving comparison\n\nThis is the manuscript table subset. Values are medians from three runs with server restarts, warm-up and matched requests; model order was rotated. Ratios use the same-prompt **bf16** proxy measured in the same block. Displayed bf16 Llama rows come from the 12B blocks; fp8/W4A16 Llama ratios use bf16 baselines from the separate `llama-quant` block. Bold marks column bests within each manuscript table, including proxies and ties at the displayed precision. Prior 4B adapters and Llama proxies are test-selected; the displayed Qwen adapters use the Think training template. Loaded weights exclude server and KV-cache memory. Throughput values, ratios and rankings in this table use concurrency 64; latency and speed-up use concurrency 1. E4B GPTQ uses basic full-model calibration with its per_layer_model_projection retained in bf16, whereas 12B and Llama use sequential calibration; quantization-loss comparisons reflect both model and pipeline.\n';
for(const group of [...mainGroups,priorGroup]){
 if(group===priorGroup)md+='\n## Prior 4B adapters (Appendix Table 5)\n\nThis separate table ranks only its twelve displayed prior-adapter configurations.\n';
 md+='\n### '+group.title+'\n\n'+header+'\n';
 for(const r of group.rows){
  const label=r.precision==='bf16'?`${labels[r.model]} bf16`:`&emsp;${r.precision}`;
  md+='| '+label+' | '+cols.map((_,i)=>display(r,i,group===priorGroup?priorBest:mainBest)).join(' | ')+' |\n';
 }
}
md+='\n### Supplementary configurations and block-matched proxies\n\nThese rows retain the Qwen Nothink variants and the remaining measured proxy references. They are excluded from the main table ranking.\n\n'+header+'\n';
for(const r of supplementary){
 const label=r.model.startsWith('qwen35-')&&!/nothink/i.test(r.label)?r.label+' (Nothink)':r.label;
 md+='| '+label+' '+r.precision+' — '+r.block+' | '+cols.map((_,i)=>display(r,i)).join(' | ')+' |\n';
}
md+='\n## Comparisons at matched precision\n\n### Paired quality comparisons\n\nEach Gemma 4 checkpoint is compared with the same-prompt Llama proxy at the same precision on 1,213 records. Differences and two-sided 95% intervals use 50,000 paired-record bootstrap resamples, seed 42 and linear percentiles. These intervals are exploratory and unadjusted; they condition on the retained checkpoints, including test-selected proxies, and do not extend the primary production-reference families.\n\n| Model | Reference | Precision | Delta F1 (pp) | Unadjusted paired 95% CI (pp) |\n|:--|:--|:--|--:|:--|\n';
for(const r of paired.comparisons)md+=`| ${r.label} | ${r.reference_label} | ${r.precision} | ${f(r.delta_f1_pp)} | [${r.ci95_f1_pp.map(n=>f(n)).join(', ')}] |\n`;
md+='\n### Resource ratios across benchmark blocks\n\nThese resource ratios compare medians from the Gemma 4 benchmark block with the same-prompt, same-precision Llama setting in the separate `llama-quant` block. They are descriptive comparisons across blocks, not paired within-block serving measurements. Repeated bf16 proxies provide a consistency check; the paired quality intervals above do not quantify uncertainty in these resource ratios. Latency speed-up uses concurrency 1; throughput ratios use concurrency 64.\n\n| Model | Reference | Precision | Weights model/reference ↓ | Latency speed-up c=1 reference/model ↑ | Throughput c=64 model/reference ↑ |\n|:--|:--|:--|--:|--:|--:|\n';
for(const r of paired.comparisons)md+=`| ${r.label} | ${r.reference_label} | ${r.precision} | ${f(r.weights_model_over_reference)}× | ${f(r.latency_speedup_reference_over_model)}× | ${f(r.throughput_model_over_reference)}× |\n`;
md+='\n## Checkpoint selection (Table 1)\n\n| Model | Rule | Step | Validation F1 (%) | Sweep test F1 (%) | Delta vs production (pp) | Gap to test oracle (pp) |\n|:--|:--|--:|--:|--:|--:|--:|\n';
for(const r of c.runs){
 for(const [rule,step,key] of [['Validation F1',r.selected_by_validation_f1,'validation_f1'],['Validation loss',r.selected_by_validation_loss,'validation_loss'],['Final checkpoint',r.final_checkpoint.step,null],['Test oracle (descriptive)',r.test_oracle_step,'test_oracle']]){
  const p=r.curve.find(p=>p.step===step);
  const delta=key?r.selection_summary[key].delta_vs_production_pp:r.final_checkpoint.delta_vs_production_pp;
  const gap=key?r.selection_summary[key].oracle_gap_pp:r.final_checkpoint.oracle_minus_final_f1_pp;
  md+='| '+[r.label,rule,step,f(p.validation_f1_pct),f(p.test_f1_pct),f(delta),f(gap)].join(' | ')+' |\n';
 }
}
md+='\nValidation-F1 selection maximizes the fractional score rounded to four decimal places (0.01 F1 percentage point), then validation strict record exact-match count, then prefers the earlier step. All 48 tie-break counts are retained in checkpoints.csv. Validation-loss ties prefer the earlier step. The final checkpoint is a no-selection baseline; the test oracle is descriptive. Retained selection contrasts need not equal subtraction of rounded F1 scores; final-checkpoint contrasts follow the manuscript’s rounded-score calculation. Validation-F1 selection was adopted after inspection of test sweeps. Sweep and serving-evaluation scores are distinct observations; headline results consistently use the complete serving evaluation.\n\n## Hosted reference\n\n| Setting | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Supplementary lower bound (pp) |\n|:--|--:|--:|--:|:--|--:|\n'+h.settings.map(r=>`| ${r.prompt}, ${r.demonstrations}-shot | ${f(r.f1_pct)} | ${f(r.raw_exact_pct)} | ${f(r.delta_f1_pp)} | [${r.ci95_f1_pp.map(x=>f(x)).join(', ')}] | ${f(r.adjusted_lower_bound_pp,3)} |`).join('\n')+'\n\nHosted bounds use the same alpha/9 cutoff outside the primary families. No setting clears −1 pp; this does not prove a deficit greater than one point.\n';
md+='\n## Primary quality comparisons (Table 2)\n\nProduction canonical micro-F1 is '+f(q.production.f1_pct)+'% and raw record exact match is '+f(q.production.raw_exact_pct)+'%. Each JO/FT family retains the divisor of nine planned cells. A one-sided lower bound strictly above −1 percentage point clears the planning criterion; this does not establish superiority or operational equivalence. Two-sided 95% intervals are exploratory. Hosted references above are supplementary; local Llama proxies remain descriptive.\n\n| Configuration | F1 (%) | Raw exact (%) | Delta F1 (pp) | 95% CI (pp) | Adjusted lower bound (pp) | Clears −1 pp |\n|:--|--:|--:|--:|:--|--:|:--|\n';
for(const model of gemmaModels)for(const precision of precs){
 const r=quality(model,'validation_f1',precision);
 md+='| '+[labels[model]+' '+precision,f(r.f1_pct),f(r.raw_exact_pct),f(r.delta_f1_pp),'['+r.ci95_f1_pp.map(n=>f(n)).join(', ')+']',f(r.adjusted_lower_bound_pp,3),r.meets_adjusted_margin?'Yes':'No'].join(' | ')+' |\n';
}
md+='\n## Field-level quality (Table 6)\n\nCanonical field F1 (%) on 1,213 records. Support counts nonempty human-labelled values. Gemma columns use validation-selected JO checkpoints. These field comparisons are descriptive, especially for COUNTRY with eight nonempty values.\n\n| Field | Support | Production | 12B bf16 | 12B W4A16 | E4B bf16 | E4B W4A16 |\n|:--|--:|--:|--:|--:|--:|--:|\n';
for(const [field,support] of Object.entries(q.production.gold_nonempty_counts)){
 const values=['gemma4-12b-jo','gemma4-e4b-jo'].flatMap(model=>['bf16','W4A16'].map(p=>f(quality(model,'validation_f1',p).per_field_f1_pct[field])));
 md+='| '+[field,support,f(q.production.per_field_f1_pct[field]),...values].join(' | ')+' |\n';
}
const selectionRows=stability.runs.flatMap(r=>['bootstrap','random_half'].map(scheme=>{
 const result=r[scheme];
 return{...result,model:r.model,label:r.label,scheme,expected_delta_vs_final_pp:result.test_f1_pct.mean-r.test_f1_pct_by_step[r.final_step]};
}));
write('selection-stability.csv',csv([['model','scheme','validation_records','resamples','same_as_full_pick_fraction','test_F1_mean_percent','test_F1_p05_percent','test_F1_p50_percent','test_F1_p95_percent','expected_delta_vs_final_pp','beats_validation_loss_fraction','matches_or_beats_final_fraction','beats_final_fraction','mean_oracle_regret_pp','selected_step_counts'],...selectionRows.map(r=>[r.label,r.scheme,stability.validation_records,stability.resamples,r.same_as_full_pick_fraction,r.test_f1_pct.mean,r.test_f1_pct.p05,r.test_f1_pct.p50,r.test_f1_pct.p95,r.expected_delta_vs_final_pp,r.beats_validation_loss_fraction,r.matches_or_beats_final_fraction,r.beats_final_fraction,r.mean_oracle_regret_pp,JSON.stringify(r.selected_step_counts)])]));
const errorRows=errors.fields.flatMap(field=>field.comparisons.map(r=>({...r,field:field.field,test_records:field.test_records})));
const errorTypes=errors.taxonomy.order;
write('error-analysis.csv',csv([['field','model','precision','test_records','gold_nonempty','both_correct','losses_to_production','wins_over_production','both_incorrect','spurious_model','spurious_production',...errorTypes.map(t=>'model_loss_'+t),...errorTypes.map(t=>'production_loss_'+t),'losses_gold_in_descriptor','losses_prediction_in_descriptor','shared_bf16_W4A16_losses','identical_loss_outputs','new_W4A16_losses','resolved_bf16_losses'],...errorRows.map(r=>[r.field,r.label,r.precision,r.test_records,r.paired.gold_nonempty,r.paired.both_correct,r.paired.production_only,r.paired.model_only,r.paired.both_incorrect,r.paired.spurious_model,r.paired.spurious_production,...errorTypes.map(t=>r.loss_type_counts[t]),...errorTypes.map(t=>r.production_loss_type_counts[t]),r.descriptor_presence_counts.losses_gold_in_descriptor,r.descriptor_presence_counts.losses_prediction_in_descriptor,r.precision_overlap?.shared_loss_records,r.precision_overlap?.identical_loss_outputs,r.precision_overlap?.new_loss_records,r.precision_overlap?.resolved_bf16_loss_records])]));
write('serving-resources.csv',csv([['block','model','precision','concurrency','requests','output_tokens_per_second_median','mean_output_tokens_median','mean_GPU_power_watts_median','GPU_energy_Wh_per_request_median','KV_cache_capacity_tokens_median','full_workload_wall_seconds_median','full_workload_device_hours_per_million_median','full_workload_GPU_kWh_per_million_median'],...s.blocks.flatMap(r=>r.points.map(p=>[r.block,r.label,r.precision,p.concurrency,p.requests,p.output_tokens_per_second.median,p.mean_output_tokens.median,p.mean_gpu_power_watts.median,p.gpu_energy_wh_per_request.median,r.kv_cache_capacity_tokens.median,...(p.concurrency===64?[r.full_workload.wall_seconds.median,r.full_workload.device_hours_per_million.median,r.full_workload.gpu_energy_kwh_per_million.median]:[null,null,null])]))]));
const decisionRows=[
 ['Llama architecture (proxy)','llama31-8b-proxy-jo','bf16','gemma4-12b-jo'],
 ['Highest F1 in this guide','gemma4-12b-jo','bf16','gemma4-12b-jo'],
 ['Within criterion, half proxy weight memory','gemma4-12b-jo','W4A16','gemma4-12b-jo'],
 ['Same proxy architecture, fp8','llama31-8b-proxy-jo','fp8','llama-quant'],
 ['Lowest memory in this guide','llama31-8b-proxy-jo','W4A16','llama-quant'],
 ['Fastest in this guide','gemma4-e4b-jo','W4A16','gemma4-e4b-jo'],
].map(([interpretation,model,precision,block])=>{
 const r=servingCell(model,precision,block),score=quality(model,r.selection,precision);
 return{...r,interpretation,delta_f1_pp:score.delta_f1_pp,hours:r.full_workload.device_hours_per_million.median,energy:r.full_workload.gpu_energy_kwh_per_million.median};
});
write('decision-guide.csv',csv([['interpretation','model','precision','block','F1_percent','delta_F1_pp','loaded_weights_GiB','latency_p50_c1_seconds','throughput_c64_rows_per_hour','device_hours_per_million','GPU_kWh_per_million'],...decisionRows.map(r=>[r.interpretation,r.label,r.precision,r.block,r.f1_pct,r.delta_f1_pp,r.memory,r.latency,r.throughput,r.hours,r.energy])]));
md+='\n## Decision guide (Table 4)\n\nThese labels apply only to the six configurations in this guide. Local Llama proxies are separately trained and test-selected; their precision variants have descriptive quality comparisons. Device-hours and GPU energy per million records use the full 192-request workload at concurrency 64, including ramp-down. For each repetition, H = 1,000,000 × wall seconds / (3,600 × 192), and GPU kWh/M = H × mean GPU watts / 1,000. Reported quantities are medians of these repetition-level estimates, not inverse headline throughput or products of medians. GPU telemetry excludes CPU and other system power. These short-workload extrapolations are not production costs or sustained capacity.\n\n| Priority in this guide | Configuration | F1 (%) ↑ | Delta F1 (pp) ↑ | Weights (GiB) ↓ | p50 c=1 (s) ↓ | c=64 rows/h ↑ | Device-hours/M ↓ | GPU kWh/M ↓ |\n|:--|:--|--:|--:|--:|--:|--:|--:|--:|\n';
const decisionCols=['f1_pct','delta_f1_pp','memory','latency','throughput','hours','energy'],decisionDigits=[2,2,1,2,0,1,1];
const decisionBest=decisionCols.map((key,i)=>([0,1,4].includes(i)?Math.max:Math.min)(...decisionRows.map(r=>Number(f(r[key],decisionDigits[i])))));
for(const r of decisionRows){
 const values=decisionCols.map((key,i)=>{
  const n=Number(f(r[key],decisionDigits[i]));let value=i===4?n.toLocaleString('en-US'):f(n,decisionDigits[i]);
  if(i===1&&n>0)value='+'+value;
  return n===decisionBest[i]?'**'+value+'**':value;
 });
 md+='| '+[r.interpretation,labels[r.model]+' '+r.precision,...values].join(' | ')+' |\n';
}
md+='\n## Validation-selection stability\n\nEach scheme re-applies checkpoint selection 2,000 times, then maps each selected step to its retained sweep test F1. Bootstrap samples contain 668 validation records drawn with replacement; random halves contain 334 without replacement. These summaries quantify sensitivity on fixed data and trajectories, not uncertainty on new data or repeated training runs. They differ from the paired-record quality intervals.\n\n| Model | Scheme | Same full-validation pick (%) | Beats loss pick (%) | Matches or beats final (%) | Expected test F1 (%) | Expected delta vs final (pp) |\n|:--|:--|--:|--:|--:|--:|--:|\n';
for(const r of selectionRows)md+='| '+[r.label,r.scheme,f(100*r.same_as_full_pick_fraction,1),f(100*r.beats_validation_loss_fraction,1),f(100*r.matches_or_beats_final_fraction,1),f(r.test_f1_pct.mean,3),f(r.expected_delta_vs_final_pp,3)].join(' | ')+' |\n';
md+='\n## Paired field errors\n\nA loss means production is correct and the candidate is wrong on a nonempty gold field; a win reverses those roles. The first matching string relation supplies the loss type. Counts are descriptive, exclude shared errors from the loss taxonomy, and do not diagnose causes or establish label correctness. For W4A16, shared/new/resolved losses compare with the same model at bf16. All nine loss-type counts and production-loss counts are retained in error-analysis.csv.\n\n| Field | Model / precision | Losses | Wins | Shared with bf16 | Identical loss outputs | New W4A16 losses | Resolved bf16 losses |\n|:--|:--|--:|--:|--:|--:|--:|--:|\n';
for(const r of errorRows)md+='| '+[r.field,r.label+' '+r.precision,r.paired.production_only,r.paired.model_only,r.precision_overlap?.shared_loss_records??'—',r.precision_overlap?.identical_loss_outputs??'—',r.precision_overlap?.new_loss_records??'—',r.precision_overlap?.resolved_bf16_loss_records??'—'].join(' | ')+' |\n';
md+='\n## Production training context\n\nThe production team reports '+production.weight_update_method.method+' of '+production.model+' using '+production.prompt+' through '+production.platform+'. The team recalls approximately '+production.runtime_estimate.wall_clock_minutes.join('–')+' minutes of training wall-clock time; accessible billing records suggest approximately US$'+production.cost_estimate.amount+' at list price. These retrospective operator-reported estimates are not independently verified runtime, GPU-hours or total cost. GPU type/count and detailed training settings remain unavailable. They provide historical context, not a matched comparison with local LoRA or evidence of savings. See configs/production-context.json for qualifications.\n';
write('tables.md',md);

// Figure 1 preserves the manuscript's order, symbols and inferential distinctions.
const forest=[];
for(const [group,selection]of [['Primary: validation F1','validation_f1'],['Sensitivity: validation loss','validation_loss']]){
 forest.push({group});
 for(const model of gemmaModels)for(const p of precs)forest.push({...quality(model,selection,p),text:labels[model]+' '+p});
}
forest.push({group:'Hosted Astra: supplementary'});
for(const [prompt,demonstrations]of [['JO',0],['FT',0],['JO',10],['FT',10]]){
 const r=unique(h.settings.filter(r=>r.prompt===prompt&&r.demonstrations===demonstrations),`hosted setting ${prompt}/${demonstrations}`);
 forest.push({...r,text:`Astra ${prompt} ${demonstrations}-shot`,hosted:true});
}
forest.push({group:'Local Llama proxies: descriptive'});
for(const model of proxyModels)for(const p of precs){
 const r=quality(model,'proxy',p);
 if(r.adjusted_lower_bound_pp!=null||r.meets_adjusted_margin!=null)throw new Error(`Proxy must remain descriptive: ${model}/${p}`);
 forest.push({...r,text:labels[model]+' '+p,proxy:true});
}
const fw=1100,fh=forest.length*24+168,xlo=-5.5,xhi=1.5,x=v=>430+(v-xlo)/(xhi-xlo)*620;
let fb=txt(30,30,'Quality difference from production','class="title"')
 +txt(30,52,'Bars: exploratory 95% CIs. Ticks: one-sided lower bounds at alpha = 0.05/9.')
 +txt(30,70,'Filled circles: bound > −1. Astra: open squares. Proxies: open diamonds, no adjusted bound.');
for(let n=-5;n<=1;n++)fb+=line(x(n),100,x(n),fh-68,`stroke="${n===0?'#555':'#ddd'}"`)+txt(x(n),fh-45,n,'text-anchor="middle"');
fb+=line(x(-1),100,x(-1),fh-68,'stroke="#333" stroke-dasharray="5 4"')+txt(x(-1),94,'margin −1','text-anchor="middle"');
forest.forEach((r,i)=>{
 const y=112+i*24;
 if(r.group){fb+=txt(30,y,r.group,'font-weight="bold"');return;}
 const col=r.proxy?'#555555':r.hosted?'#79549b':modelColor(r.model);
 fb+=txt(40,y+4,r.text)+line(x(r.ci95_f1_pp[0]),y,x(r.ci95_f1_pp[1]),y,`stroke="${col}" stroke-width="2"`);
 if(!r.proxy&&r.adjusted_lower_bound_pp!=null)fb+=line(x(r.adjusted_lower_bound_pp),y-5,x(r.adjusted_lower_bound_pp),y+5,`stroke="${col}" stroke-width="2"`);
 const xx=x(r.delta_f1_pp),fill=!r.hosted&&!r.proxy&&r.meets_adjusted_margin===true?col:'white';
 if(r.proxy)fb+=`<polygon points="${xx},${y-5} ${xx+5},${y} ${xx},${y+5} ${xx-5},${y}" fill="white" stroke="${col}" stroke-width="1.5"/>`;
 else if(r.hosted)fb+=`<rect x="${xx-4}" y="${y-4}" width="8" height="8" fill="white" stroke="${col}" stroke-width="1.5"/>`;
 else fb+=`<circle cx="${xx}" cy="${y}" r="4" fill="${fill}" stroke="${col}" stroke-width="1.5"/>`;
});
fb+=txt(740,fh-18,'Delta canonical F1 (percentage points)','text-anchor="middle"');
write('quality-intervals.svg',svg(fw,fh,fb,'Quality differences from production: 24 Gemma 4 cells, four hosted settings and six Llama proxies'));

// Four concurrency panels; matched bf16 proxies stay within their measured block.
let sb=txt(30,32,'Throughput under concurrent load (rows/hour)','class="title"')+txt(30,55,'Three-repetition medians. Lines connect measured concurrency levels; no interpolation model is fitted.');
c.runs.forEach((model,i)=>{
 const ox=60+(i%2)*570,oy=110+Math.floor(i/2)*330,rows=s.blocks.filter(r=>r.block===model.model),cs=[1,8,16,32,64],max=80000;
 const xx=n=>ox+40+cs.indexOf(n)*106,yy=v=>oy+235-v/max*210;
 sb+=txt(ox+40,oy,model.label,'font-weight="bold"');
 for(let v=0;v<=max;v+=20000)sb+=line(ox+40,yy(v),ox+465,yy(v),'stroke="#ddd"')+txt(ox+33,yy(v)+4,v/1000+'k','text-anchor="end"');
 for(const cc of cs)sb+=txt(xx(cc),oy+253,cc,'text-anchor="middle"');
 rows.forEach((r,j)=>{
  const col=r.role==='proxy'?'#666':color(r.precision);
  sb+=`<path d="${r.points.map((p,k)=>(k?'L':'M')+xx(p.concurrency)+','+yy(p.rows_per_hour.median)).join(' ')}" stroke="${col}" stroke-width="2"/>`;
  r.points.forEach(p=>sb+=`<circle cx="${xx(p.concurrency)}" cy="${yy(p.rows_per_hour.median)}" r="3" fill="${col}"/>`);
  sb+=txt(ox+45+j*111,oy+276,r.role==='proxy'?'bf16 proxy':r.precision,`style="fill:${col}"`);
 });
 sb+=txt(ox+248,oy+299,'Concurrency','text-anchor="middle"');
});
write('concurrency-all-precisions.svg',svg(1180,770,sb,'Supplementary throughput by concurrency for all Gemma precisions and block-matched bf16 proxies'));

// Manuscript Figure 2: both Llama precisions use the quantization-control block.
const seriesStyles=[
 {label:'Gemma 4 12B W4A16',model:'gemma4-12b',precision:'W4A16',col:'#1f5fa8',dash:''},
 {label:'Gemma 4 E4B W4A16',model:'gemma4-e4b',precision:'W4A16',col:'#c8541d',dash:''},
 {label:'Llama 8B bf16',model:'llama31-8b-proxy',precision:'bf16',col:'#555555',dash:'7 4'},
 {label:'Llama 8B W4A16',model:'llama31-8b-proxy',precision:'W4A16',col:'#2b7a58',dash:'7 3 2 3'},
];
let mb=txt(30,30,'Latency and throughput versus concurrent requests','class="title"')
 +txt(30,53,'Points: medians; error bars: observed minimum–maximum across three runs, not confidence intervals.')
 +txt(30,73,'Gemma: candidate blocks. Both Llama precisions: quantization-control block. Prefix caching remains enabled.');
for(const [pi,prompt] of ['jo','ft'].entries())for(const [mi,metric] of ['latency_p50_seconds','rows_per_hour'].entries()){
 const ox=65+mi*570,oy=120+pi*310,max=mi?80000:7;
 const xx=n=>ox+40+(n-1)/63*440,yy=v=>oy+215-v/max*195;
 mb+=txt(ox+40,oy,`${prompt.toUpperCase()}: ${mi?'throughput (thousand rows/h)':'p50 latency (seconds)'}`,'font-weight="bold"');
 for(const v of mi?[0,20000,40000,60000,80000]:[0,2,4,6])mb+=line(ox+40,yy(v),ox+480,yy(v),'stroke="#ddd"')+txt(ox+30,yy(v)+4,mi?v/1000:v,'text-anchor="end"');
 for(const n of [1,8,16,32,64])mb+=txt(xx(n),oy+235,n,'text-anchor="middle"');
 for(const style of seriesStyles){
  const model=`${style.model}-${prompt}`,block=style.model.startsWith('llama')?'llama-quant':model;
  const row=servingCell(model,style.precision,block);
  mb+=`<path d="${row.points.map((p,k)=>(k?'L':'M')+xx(p.concurrency)+','+yy(p[metric].median)).join(' ')}" stroke="${style.col}" stroke-width="2" stroke-dasharray="${style.dash}"/>`;
  for(const p of row.points){
   const v=p[metric],px=xx(p.concurrency);
   if(v.min<0||v.max>max)throw new Error(`Concurrency plot range exceeded: ${model}/${metric}`);
   mb+=line(px,yy(v.min),px,yy(v.max),`stroke="${style.col}"`)+line(px-3,yy(v.min),px+3,yy(v.min),`stroke="${style.col}"`)+line(px-3,yy(v.max),px+3,yy(v.max),`stroke="${style.col}"`);
   mb+=`<circle cx="${px}" cy="${yy(v.median)}" r="3" fill="${style.col}"/>`;
  }
 }
 mb+=txt(ox+260,oy+260,'Concurrent requests','text-anchor="middle"');
}
seriesStyles.forEach((style,i)=>{
 const x=115+(i%2)*530,y=735+Math.floor(i/2)*24;
 mb+=line(x,y-4,x+36,y-4,`stroke="${style.col}" stroke-width="2" stroke-dasharray="${style.dash}"`)+txt(x+47,y,style.label);
});
write('concurrency.svg',svg(1180,785,mb,'Manuscript Figure 2: JO and FT latency and throughput with observed min–max ranges'));

// Each checkpoint panel includes F1 and loss, both selection rules and the test oracle.
function star(x,y,radius=6){
 return Array.from({length:10},(_,i)=>{const a=-Math.PI/2+i*Math.PI/5,r=i%2?radius*0.43:radius;return `${x+r*Math.cos(a)},${y+r*Math.sin(a)}`;}).join(' ');
}
let cb=txt(30,32,'Checkpoint sweeps','class="title"')
 +txt(30,55,'Upper panels: solid = test F1; dashed = validation F1; horizontal black = production. Lower panels: validation loss.')
 +txt(30,75,'Vertical lines: dashed black = validation-F1 pick; gray = validation-loss pick. Stars: test oracle (descriptive).');
c.runs.forEach((r,i)=>{
 const ox=60+(i%2)*570,oy=120+Math.floor(i/2)*420;
 const xx=step=>ox+40+(step-150)/(2350-150)*425;
 const yf=v=>oy+205-(v-87)/11*190,yl=v=>oy+325-(v*100-0.65)/1.3*75;
 const col=modelColor(r.model);
 if(r.curve.some(p=>p.test_f1_pct<87||p.test_f1_pct>98||p.validation_f1_pct<87||p.validation_f1_pct>98||p.validation_loss*100<0.65||p.validation_loss*100>1.95))throw new Error(`Checkpoint plot range exceeded: ${r.model}`);
 cb+=txt(ox+40,oy-9,r.label,'font-weight="bold"')+txt(ox+40,oy+8,'F1 (%)','class="label"')+txt(ox+40,oy+237,'Validation loss (×10⁻²)','class="label"');
 for(let v=88;v<=98;v+=2)cb+=line(ox+40,yf(v),ox+465,yf(v),'stroke="#ddd"')+txt(ox+30,yf(v)+4,v,'text-anchor="end"');
 for(const v of [0.8,1.2,1.6])cb+=line(ox+40,yl(v/100),ox+465,yl(v/100),'stroke="#ddd"')+txt(ox+30,yl(v/100)+4,f(v,1),'text-anchor="end"');
 for(const [step,style]of [[r.selected_by_validation_loss,'stroke="#999"'],[r.selected_by_validation_f1,'stroke="#222" stroke-dasharray="5 4"']]){
  cb+=line(xx(step),oy+15,xx(step),oy+205,style)+line(xx(step),oy+250,xx(step),oy+325,style);
 }
 cb+=line(ox+40,yf(q.production.f1_pct),ox+465,yf(q.production.f1_pct),'stroke="#222" stroke-width="1"');
 for(const [key,dashed]of [['test_f1_pct',false],['validation_f1_pct',true]])cb+=`<path d="${r.curve.map((p,k)=>(k?'L':'M')+xx(p.step)+','+yf(p[key])).join(' ')}" stroke="${col}" stroke-width="2" ${dashed?'stroke-dasharray="6 4"':''}/>`;
 r.curve.forEach(p=>cb+=`<circle cx="${xx(p.step)}" cy="${yf(p.test_f1_pct)}" r="2.3" fill="${col}"/>`);
 const oracle=unique(r.curve.filter(p=>p.step===r.test_oracle_step),`test oracle for ${r.model}`);
 cb+=`<polygon points="${star(xx(oracle.step),yf(oracle.test_f1_pct))}" fill="white" stroke="${col}" stroke-width="1.5"/>`;
 cb+=`<path d="${r.curve.map((p,k)=>(k?'L':'M')+xx(p.step)+','+yl(p.validation_loss)).join(' ')}" stroke="${col}" stroke-width="2"/>`;
 r.curve.forEach(p=>cb+=`<circle cx="${xx(p.step)}" cy="${yl(p.validation_loss)}" r="2.3" fill="${col}"/>`);
 const lossPick=unique(r.curve.filter(p=>p.step===r.selected_by_validation_loss),`validation-loss pick for ${r.model}`);
 cb+=`<circle cx="${xx(lossPick.step)}" cy="${yl(lossPick.validation_loss)}" r="3.5" fill="#222"/>`;
 for(const step of [400,1200,2000])cb+=txt(xx(step),oy+349,step,'text-anchor="middle"');
 cb+=txt(ox+248,oy+374,'Training step','text-anchor="middle"');
});
write('checkpoint-sweeps.svg',svg(1180,940,cb,'Validation and test F1, validation loss, selection rules and test oracles across checkpoint sweeps'));
console.log('Regenerated 11 CSV files, manuscript-aligned tables with supplementary rows, and 4 SVG figures from published aggregate data.');
