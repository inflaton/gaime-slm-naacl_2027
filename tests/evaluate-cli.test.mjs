import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';

const root = fileURLToPath(new URL('../',import.meta.url));
const cli = path.join(root,'scripts/evaluate.mjs');
const scoreArgs = ['score','--gold','fixtures/gold.jsonl','--pred','fixtures/model-a.jsonl'];
const compareArgs = ['compare','--gold','fixtures/gold.jsonl','--a','fixtures/model-a.jsonl','--b','fixtures/model-b.jsonl'];
const run = args => spawnSync(process.execPath,[cli,...args],{cwd:root,encoding:'utf8'});
const success = args => {
  const result = run(args);
  assert.equal(result.status,0,result.stderr);
  assert.equal(result.stderr,'');
  return JSON.parse(result.stdout);
};

test('score preserves fixture counts and percentages',() => {
  const result = success(scoreArgs);
  assert.equal(result.records,6);
  assert.equal(result.overall.tp,9);
  assert.equal(result.overall.f1_pct,1800 / 19);
});

test('omitting family size explicitly leaves comparison unadjusted',() => {
  const result = success(compareArgs);
  assert.equal(result.bootstrap.resamples,50000);
  assert.equal(result.bootstrap.seed,42);
  assert.equal(result.metrics_a.f1_pct,1800 / 19);
  assert.equal(result.metrics_b.f1_pct,1600 / 18);
  assert.equal(result.noninferiority.enabled,false);
  assert.equal(result.noninferiority.family_divisor,null);
  assert.equal(result.noninferiority.lower_bound_pp,null);
  assert.equal(result.noninferiority.passed,null);
});

test('explicit family size enables adjustment and respects resample count',() => {
  const result = success([...compareArgs,'--family-size','9','--resamples','200']);
  assert.equal(result.bootstrap.resamples,200);
  assert.equal(result.noninferiority.enabled,true);
  assert.equal(result.noninferiority.family_divisor,9);
  assert.equal(result.noninferiority.alpha,0.05);
  assert(Number.isFinite(result.noninferiority.lower_bound_pp));
  assert.equal(typeof result.noninferiority.passed,'boolean');
});

test('--out writes the same result without stdout in both modes',() => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'evaluation-cli-'));
  try {
    for (const args of [scoreArgs,[...compareArgs,'--family-size','9','--resamples','200']]) {
      const expected = success(args);
      const out = path.join(dir,'result with spaces.json');
      const result = run([...args,'--out',out]);
      assert.equal(result.status,0,result.stderr);
      assert.equal(result.stderr,'');
      assert.equal(result.stdout,'');
      assert.deepEqual(JSON.parse(fs.readFileSync(out,'utf8')),expected);
    }
  } finally {
    fs.rmSync(dir,{recursive:true,force:true});
  }
});

const invalidCases = [
  ['missing mode',[],/Expected mode/],
  ['unknown mode',['unknown'],/Expected mode/],
  ['unknown flag',[...scoreArgs,'--mystery','value'],/Unknown option/],
  ['unexpected positional value',[...scoreArgs,'extra'],/Unknown option or argument/],
  ['duplicate gold',[...scoreArgs,'--gold','other.jsonl'],/Repeated option: --gold/],
  ['duplicate family size',[...compareArgs,'--family-size','9','--family-size','1'],/Repeated option: --family-size/],
  ['duplicate resamples',[...compareArgs,'--resamples','10','--resamples','20'],/Repeated option: --resamples/],
  ['duplicate output',[...scoreArgs,'--out','first','--out','second'],/Repeated option: --out/],
  ['family flag without a value',[...compareArgs,'--family-size'],/Missing value for --family-size/],
  ['resample flag without a value',[...compareArgs,'--resamples'],/Missing value for --resamples/],
  ['output flag without a value',[...scoreArgs,'--out'],/Missing value for --out/],
  ['flag used as family size',[...compareArgs,'--family-size','--out','result.json'],/Missing value for --family-size/],
  ['flag used as input path',['score','--gold','--pred','file'],/Missing value for --gold/],
  ['blank path',['score','--gold','  ','--pred','file'],/Missing value for --gold/],
  ['empty family size',[...compareArgs,'--family-size',''],/Missing value for --family-size/],
  ['missing gold',['score','--pred','does-not-exist.jsonl'],/Missing required option: --gold/],
  ['missing prediction',['score','--gold','does-not-exist.jsonl'],/Missing required option: --pred/],
  ['missing comparison A',['compare','--gold','does-not-exist.jsonl','--b','does-not-exist.jsonl'],/Missing required option: --a/],
  ['missing comparison B',['compare','--gold','does-not-exist.jsonl','--a','does-not-exist.jsonl'],/Missing required option: --b/],
  ['prediction in compare mode',[...compareArgs,'--pred','file'],/--pred is not supported in compare mode/],
  ...['--a','--b','--resamples','--family-size'].map(flag => [
    `${flag} in score mode`,[...scoreArgs,flag,'1'],new RegExp(`${flag} is not supported in score mode`),
  ]),
  ...['--resamples','--family-size'].flatMap(flag => ['0','-1','1.5','NaN','Infinity','9x','1e2','9007199254740992'].map(value => [
    `${flag} invalid number ${value}`,[...compareArgs,flag,value],new RegExp(`${flag} must be a positive, safe decimal integer`),
  ])),
];
for (const [name,args,message] of invalidCases) {
  test(`rejects ${name}`,() => {
    const result = run(args);
    assert.equal(result.status,1);
    assert.equal(result.stdout,'');
    assert.match(result.stderr,message);
    assert.doesNotMatch(result.stderr,/ENOENT/);
  });
}

test('invalid arguments are rejected before input reads or output writes',() => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(),'evaluation-cli-'));
  try {
    const existing = path.join(dir,'existing.json');
    const absent = path.join(dir,'absent.json');
    fs.writeFileSync(existing,'retain this content');
    for (const out of [existing,absent]) {
      const result = run(['compare','--out',out,'--gold','does-not-exist.jsonl','--a','a','--b','b','--family-size']);
      assert.equal(result.status,1);
      assert.match(result.stderr,/Missing value for --family-size/);
      assert.doesNotMatch(result.stderr,/ENOENT/);
    }
    assert.equal(fs.readFileSync(existing,'utf8'),'retain this content');
    assert.equal(fs.existsSync(absent),false);
  } finally {
    fs.rmSync(dir,{recursive:true,force:true});
  }
});
