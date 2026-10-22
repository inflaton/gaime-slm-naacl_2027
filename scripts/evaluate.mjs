import fs from 'node:fs';
import {score,compare} from '../lib/evaluation.mjs';

const usage = `Usage:
  node scripts/evaluate.mjs score --gold FILE --pred FILE [--out FILE]
  node scripts/evaluate.mjs compare --gold FILE --a FILE --b FILE [--resamples N] [--family-size N] [--out FILE]

N must be a positive, safe decimal integer. Compare defaults to 50000 resamples;
omitting --family-size produces an unadjusted descriptive interval.`;
const modes = {
  score: {required: ['--gold', '--pred'], optional: ['--out']},
  compare: {required: ['--gold', '--a', '--b'], optional: ['--resamples', '--family-size', '--out']},
};
const knownOptions = new Set(Object.values(modes).flatMap(({required,optional}) => [...required,...optional]));

function parseArguments(argv) {
  const [mode,...args] = argv;
  if (!Object.hasOwn(modes,mode)) throw Error(`Expected mode "score" or "compare".\n${usage}`);
  const {required,optional} = modes[mode];
  const allowed = new Set([...required,...optional]);
  const options = new Map();
  for (let i = 0; i < args.length; i += 2) {
    const name = args[i];
    if (!knownOptions.has(name)) throw Error(`Unknown option or argument: ${name}`);
    if (!allowed.has(name)) throw Error(`${name} is not supported in ${mode} mode`);
    if (options.has(name)) throw Error(`Repeated option: ${name}`);
    const value = args[i + 1];
    if (value === undefined || !value.trim() || value.startsWith('--')) throw Error(`Missing value for ${name}`);
    options.set(name,value);
  }
  for (const name of required) {
    if (!options.has(name)) throw Error(`Missing required option: ${name}`);
  }
  for (const name of ['--resamples','--family-size']) {
    if (!options.has(name)) continue;
    const value = options.get(name);
    if (!/^\d+$/.test(value) || !Number.isSafeInteger(Number(value)) || Number(value) < 1) {
      throw Error(`${name} must be a positive, safe decimal integer`);
    }
    options.set(name,Number(value));
  }
  return {mode,options};
}

const rows = file => fs.readFileSync(file,'utf8').trim().split(/\r?\n/).filter(Boolean).map(JSON.parse);

try {
  // Validate the entire invocation before reading inputs or creating output files.
  const {mode,options} = parseArguments(process.argv.slice(2));
  const gold = rows(options.get('--gold'));
  const result = mode === 'score'
    ? score(gold,rows(options.get('--pred')))
    : compare(gold,rows(options.get('--a')),rows(options.get('--b')),{
      resamples: options.get('--resamples') ?? 50000,
      familySize: options.get('--family-size') ?? null,
    });
  const output = JSON.stringify(result,null,2) + '\n';
  if (options.has('--out')) fs.writeFileSync(options.get('--out'),output);
  else process.stdout.write(output);
} catch (error) {
  process.stderr.write(`Error: ${error.message}\n`);
  process.exitCode = 1;
}
