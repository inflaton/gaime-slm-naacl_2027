import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import {auditPublicFiles,auditPublicValue} from '../scripts/validate.mjs';

test('configuration templates retain legitimate column and output metadata',()=>{
 assert.doesNotThrow(()=>auditPublicValue({
  columns:{prompt:'instruction',query:'input',response:'output'},
  input:'private/train.jsonl',output:'private/prepared',
  model_url:'https://example.org/model',runtime_estimate:{wall_clock_minutes:[30,40]}
 },{kind:'config'}));
});

for(const [name,value]of [
 ['private provenance key',{nested:{source_file:'private.csv'}}],
 ['case-insensitive record identifier',{nested:{Record_ID:'synthetic-test-leak'}}],
 ['Unix private path',{nested:['/Users/example/private.jsonl']}],
 ['Windows private path',{nested:['C:\\Users\\example\\private.jsonl']}],
 ['UNC private path',{nested:['\\\\server\\share\\private.jsonl']}],
 ['email',{nested:['synthetic-person@example.test']}],
 ['source digest',{nested:['a'.repeat(64)]}]
])test(`public values reject ${name} in aggregates and configurations`,()=>{
 for(const kind of ['aggregate','config'])assert.throws(()=>auditPublicValue(value,{kind}));
});

test('aggregate exports cannot embed prediction rows',()=>{
 assert.throws(()=>auditPublicValue({measurements:[{output:'SYNTHETIC ROW'}]}));
});

test('new nested aggregate and configuration files are discovered automatically',t=>{
 const root=fs.mkdtempSync(path.join(os.tmpdir(),'public-json-audit-'));
 t.after(()=>fs.rmSync(root,{recursive:true,force:true}));
 for(const directory of ['aggregates/new','configs/new'])fs.mkdirSync(path.join(root,directory),{recursive:true});
 fs.writeFileSync(path.join(root,'aggregates/new/report.json'),JSON.stringify({count:3}));
 fs.writeFileSync(path.join(root,'configs/new/template.json'),JSON.stringify({output:'private/prepared'}));
 assert.deepEqual(auditPublicFiles(root),['aggregates/new/report.json','configs/new/template.json']);
 fs.writeFileSync(path.join(root,'configs/new/context.json'),JSON.stringify({contact:'synthetic-person@example.test'}));
 assert.throws(()=>auditPublicFiles(root),/configs\/new\/context\.json\.contact/);
});
