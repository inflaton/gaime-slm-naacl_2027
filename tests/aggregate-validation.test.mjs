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

test('resource extrapolation uses the full workload duration per repetition',t=>{
 const root=alteredArtifact(t,'serving.json',value=>{value.blocks[0].full_workload.repeat_measurements[0].device_hours_per_million*=1.05;});
 assert.throws(()=>validateArtifact(root),/Inconsistent aggregate value/);
});

test('field-error taxonomy counts must partition production-only and model-only losses',t=>{
 const root=alteredArtifact(t,'error-analysis.json',value=>{value.fields[0].comparisons[0].loss_type_counts.missed++;});
 assert.throws(()=>validateArtifact(root),/Error categories must partition/);
});
