import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import {validateArtifact} from '../scripts/validate.mjs';

const artifactRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
function alteredArtifact(t,file,change){
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'aggregate-validation-'));
 t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
 for(const directory of ['aggregates','configs','fixtures'])fs.cpSync(path.join(artifactRoot,directory),path.join(root,directory),{recursive:true});
 const destination=path.join(root,'aggregates',file),value=JSON.parse(fs.readFileSync(destination,'utf8'));
 change(value);fs.writeFileSync(destination,JSON.stringify(value));return root;
}

test('final checkpoint summaries must remain tied to the saved sweep',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{value.runs[0].final_checkpoint.test_f1_pct+=.01;});
 assert.throws(()=>validateArtifact(root),/Final checkpoint differs/);
});

test('selection frequencies must account for all validation resamples',t=>{
 const root=alteredArtifact(t,'selection-stability.json',value=>{value.runs[0].bootstrap.selected_step_counts['1400']++;});
 assert.throws(()=>validateArtifact(root),/Selection frequencies must cover every resample/);
});

test('retained selection summaries must identify the selected checkpoint',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{value.runs[0].selection_summary.validation_f1.step=200;});
 assert.throws(()=>validateArtifact(root),/Selection summary must identify its retained checkpoint/);
});

test('the second training seed retains the complete original checkpoint schedule',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{value.replicates[0].curve.pop();});
 assert.throws(()=>validateArtifact(root),/Replicate must retain the same twelve-checkpoint schedule/);
});

test('the replicate uses strict record exact match to resolve rounded validation F1 ties',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{
  const run=value.replicates[0],selected=run.curve.find(point=>point.step===run.selected_by_validation_f1);
  run.curve[0].validation_f1_pct=selected.validation_f1_pct;
  run.curve[0].validation_strict_record_exact_count=selected.validation_strict_record_exact_count+1;
 });
 assert.throws(()=>validateArtifact(root),/Inconsistent rounded-F1\/tie-break selection/);
});

test('replicate comparisons cannot acquire confirmatory noninferiority claims',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{value.replicates[0].production_comparisons.validation_f1.confirmatory_noninferiority=true;});
 assert.throws(()=>validateArtifact(root),/Replicate comparisons must remain exploratory/);
});

test('replicate record resampling keeps its bootstrap seed distinct from its training seed',t=>{
 const root=alteredArtifact(t,'checkpoints.json',value=>{value.replicates[0].production_comparisons.validation_f1.bootstrap.seed=43;});
 assert.throws(()=>validateArtifact(root));
});

test('a second training seed cannot silently replace a primary quality cell',t=>{
 const root=alteredArtifact(t,'quality.json',value=>{value.cells.find(row=>row.selection==='validation_f1').seed=43;});
 assert.throws(()=>validateArtifact(root),/Primary quality families must not include a second-seed replicate/);
});

test('resource extrapolation uses the full workload duration per repetition',t=>{
 const root=alteredArtifact(t,'serving.json',value=>{value.blocks[0].full_workload.repeat_measurements[0].device_hours_per_million*=1.05;});
 assert.throws(()=>validateArtifact(root),/Inconsistent aggregate value/);
});

test('field-error taxonomy counts must partition production-only and model-only losses',t=>{
 const root=alteredArtifact(t,'error-analysis.json',value=>{value.fields[0].comparisons[0].loss_type_counts.missed++;});
 assert.throws(()=>validateArtifact(root),/Error categories must partition/);
});
