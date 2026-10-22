import assert from 'node:assert/strict';
import test from 'node:test';
import { FIELDS, parseOutput, score } from '../lib/evaluation.mjs';

const target = value => Object.fromEntries(FIELDS.map(field => [field, field === 'MERCHANT_NAME' ? value : '']));
const literalOutput = (rawValue, quote) => `{${FIELDS.map(field => `${quote}${field}${quote}: ${quote}${field === 'MERCHANT_NAME' ? rawValue : ''}${quote}`).join(', ')}}`;
const scored = (output, value) => score([{ id: 'synthetic', target: target(value) }], [{ id: 'synthetic', output }]);

function rejectsWithoutCredit(output, value) {
  assert.deepEqual(parseOutput(output), { values: null, status: 'parse_failed' });
  const result = scored(output, value);
  assert.equal(result.parse_failed, 1);
  assert.equal(result.overall.f1_pct, 0);
  assert.equal(result.overall.raw_exact_pct, 0);
}

function acceptsWithFullCredit(output, value) {
  assert.deepEqual(parseOutput(output), { values: target(value), status: 'schema_valid' });
  const result = scored(output, value);
  assert.equal(result.parse_failed, 0);
  assert.equal(result.overall.f1_pct, 100);
  assert.equal(result.overall.raw_exact_pct, 100);
}

for (const quote of ["'", '"']) {
  for (const [name, lineBreak] of [['LF', '\n'], ['CR', '\r'], ['CRLF', '\r\n']]) {
    test(`${quote}-quoted raw ${name} cannot earn credit`, () => {
      const value = `Maple${lineBreak}Store`;
      rejectsWithoutCredit(literalOutput(value, quote), value);
    });

    test(`${quote}-quoted escaped ${name} line continuation remains valid`, () => {
      acceptsWithFullCredit(literalOutput(`Maple\\${lineBreak}Store`, quote), 'MapleStore');
    });
  }

  for (const [name, escape, value] of [['newline', '\\n', '\n'], ['carriage return', '\\r', '\r'], ['CRLF', '\\r\\n', '\r\n']]) {
    test(`${quote}-quoted ${name} escape remains valid`, () => {
      acceptsWithFullCredit(literalOutput(`Maple${escape}Store`, quote), `Maple${value}Store`);
    });
  }

  test(`${quote}-quoted literal NUL cannot earn credit`, () => {
    const value = 'Maple\0Store';
    rejectsWithoutCredit(literalOutput(value, quote), value);
  });

  test(`${quote}-quoted escaped NUL remains valid`, () => {
    acceptsWithFullCredit(literalOutput('Maple\\x00Store', quote), 'Maple\0Store');
  });

  test(`${quote}-quoted escaped quote and backslash remain valid`, () => {
    acceptsWithFullCredit(literalOutput(`Maple\\${quote}\\\\Store`, quote), `Maple${quote}\\Store`);
  });
}

test('ordinary JSON and the null-output contract remain valid', () => {
  acceptsWithFullCredit(JSON.stringify(target('Maple Store')), 'Maple Store');
  assert.deepEqual(parseOutput(null), { values: null, status: 'parse_failed' });
  const result = scored(null, 'Maple Store');
  assert.equal(result.missing_or_null, 1);
  assert.equal(result.parse_failed, 0);
  assert.equal(result.overall.f1_pct, 0);
});

test('null field values are normalized and counted as schema invalid', () => {
  const fields = target('Maple Store');
  fields.CITY = null;
  assert.deepEqual(parseOutput(JSON.stringify(fields)), { values: target('Maple Store'), status: 'schema_invalid' });
});
