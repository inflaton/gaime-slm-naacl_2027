import fs from 'node:fs';
import {score,compare} from '../lib/evaluation.mjs';
const [mode,...args]=process.argv.slice(2);const option=name=>{const i=args.indexOf(name);return i<0?undefined:args[i+1];};
const rows=p=>{if(!p)throw Error('Missing input path');return fs.readFileSync(p,'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);};
if(!['score','compare'].includes(mode))throw Error('Usage: node scripts/evaluate.mjs score --gold FILE --pred FILE | compare --gold FILE --a FILE --b FILE [--resamples N] [--family-size N]');
const gold=rows(option('--gold'));
const result=mode==='score'?score(gold,rows(option('--pred'))):compare(gold,rows(option('--a')),rows(option('--b')),{resamples:Number(option('--resamples')??50000),familySize:option('--family-size')===undefined?null:Number(option('--family-size'))});
const output=JSON.stringify(result,null,2)+'\n';if(option('--out'))fs.writeFileSync(option('--out'),output);else process.stdout.write(output);
