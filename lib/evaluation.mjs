// Portable canonical scoring and paired record bootstrap; no network or private dependencies.
export const FIELDS = ['INTERMEDIARY','MERCHANT_NAME','DYNAMIC_DESCRIPTOR','STREET','ZIP','CITY','STATE','COUNTRY','PHONE','WEBSITE'];
export const canonical = value => String(value).toUpperCase().replace(/[* ,\-]/g, '');
const object = x => x !== null && typeof x === 'object' && !Array.isArray(x);
const normalize = obj => Object.fromEntries(FIELDS.map(k => [k, obj?.[k] == null ? '' : String(obj[k])]));

// Safe string/null dictionary subset of Python literal syntax used by legacy outputs.
// This never evaluates code. Nested objects, lists and numeric Python literals are rejected.
function literal(text) {
  let i = 0;
  const ws = () => { while (/\s/.test(text[i] ?? '') && i < text.length) i++; };
  const string = () => {
    ws(); const quote = text[i++]; if (!['"', "'"].includes(quote)) throw Error('Expected string');
    let result = '';
    while (i < text.length) {
      let ch = text[i++]; if (ch === quote) return result;
      if (ch !== '\\') { result += ch; continue; }
      ch = text[i++];
      const simple = { n:'\n',r:'\r',t:'\t',b:'\b',f:'\f',v:'\v',a:'\x07',"'":"'",'"':'"','\\':'\\' };
      if (Object.hasOwn(simple,ch)) { result += simple[ch]; continue; }
      if (ch === '\n') continue;
      const len = {x:2,u:4,U:8}[ch];
      if (len) {const hex = text.slice(i,i+len);if (!new RegExp(`^[a-fA-F0-9]{${len}}$`).test(hex)) throw Error('Invalid escape');result += String.fromCodePoint(parseInt(hex,16));i+=len;continue;}
      if (/[0-7]/.test(ch)) {let oct=ch;while(oct.length<3&&/[0-7]/.test(text[i]??'')) oct+=text[i++];result+=String.fromCodePoint(parseInt(oct,8));continue;}
      result += '\\'+ch;
    }
    throw Error('Unterminated string');
  };
  ws(); if(text[i++]!=='{') throw Error('Expected dictionary'); const result=Object.create(null); ws();
  while(text[i]!=='}') {
    const key=string();ws();if(text[i++]!==':')throw Error('Expected colon');ws();
    if(text.slice(i,i+4)==='None'){result[key]=null;i+=4;}else result[key]=string();
    ws();if(text[i]===','){i++;ws();}else if(text[i]!=='}')throw Error('Expected comma');
  }
  i++;ws();if(i!==text.length)throw Error('Trailing content');return result;
}
export function parseOutput(text) {
  if (text == null) return {values:null,status:'parse_failed'};
  if (typeof text !== 'string') throw Error('Prediction output must be a string or null');
  let s=text.trim();if(s.includes('</think>'))s=s.slice(s.lastIndexOf('</think>')+8).trim();
  const candidates=[s,/```(?:json)?\s*(\{[\s\S]*?\})\s*```/.exec(s)?.[1],s.includes('{')&&s.includes('}')?s.slice(s.indexOf('{'),s.lastIndexOf('}')+1):null];
  let value=null;for(const candidate of candidates){if(!candidate)continue;try{const x=JSON.parse(candidate);if(object(x)){value=x;break;}}catch{}}
  if(value===null){try{value=literal(s);}catch{}}
  if(value===null)return {values:null,status:'parse_failed'};
  // Reject unsupported non-string containers instead of silently changing legacy stringification.
  if(Object.values(value).some(x=>x!==null&&typeof x!=='string'))throw Error('Use string/null field values; non-string legacy coercion requires the Python reference scorer');
  const valid=Object.keys(value).length===FIELDS.length&&FIELDS.every(k=>Object.hasOwn(value,k)&&typeof value[k]==='string');
  return {values:normalize(value),status:valid?'schema_valid':'schema_invalid'};
}
export function align(gold,predictions) {
  if(!Array.isArray(gold)||!gold.length)throw Error('Gold must contain records');
  const ids=new Set(),pred=new Map();
  for(const row of gold){if(typeof row.id!=='string'||ids.has(row.id))throw Error('Invalid or duplicate gold ID');ids.add(row.id);}
  for(const row of predictions){if(!ids.has(row.id))throw Error('Prediction outside reserved gold set');if(pred.has(row.id))throw Error('Duplicate prediction ID');if(row.output!=null&&typeof row.output!=='string')throw Error('Output must be a string or null');pred.set(row.id,row);}
  return gold.map(row=>{
    const g=typeof row.target==='string'?JSON.parse(row.target):row.target;
    if(!object(g)||FIELDS.some(k=>!Object.hasOwn(g,k)||typeof g[k]!=='string')||Object.keys(g).length!==FIELDS.length)throw Error('Gold target must contain exactly ten string fields');
    const missing=!pred.has(row.id)||pred.get(row.id).output==null,parsed=parseOutput(pred.get(row.id)?.output),p=parsed.values??normalize({});
    const fields=FIELDS.map(k=>({tp:+!!(g[k]&&p[k]&&canonical(g[k])===canonical(p[k])),gold:+!!g[k],pred:+!!p[k]}));
    return {fields,tp:fields.reduce((s,x)=>s+x.tp,0),gold:fields.reduce((s,x)=>s+x.gold,0),pred:fields.reduce((s,x)=>s+x.pred,0),raw_exact:+!!(parsed.values&&FIELDS.every(k=>g[k]===p[k])),canonical_exact:+!!(parsed.values&&FIELDS.every(k=>canonical(g[k])===canonical(p[k]))),status:parsed.status,missing,truncated:!!pred.get(row.id)?.truncated};
  });
}
export function metrics(records) {
  const tp=records.reduce((s,x)=>s+x.tp,0),gold=records.reduce((s,x)=>s+x.gold,0),pred=records.reduce((s,x)=>s+x.pred,0);
  return {tp,gold,pred,precision_pct:pred?100*tp/pred:0,recall_pct:gold?100*tp/gold:0,f1_pct:gold+pred?200*tp/(gold+pred):0,raw_exact_pct:100*records.reduce((s,x)=>s+x.raw_exact,0)/records.length};
}
export function score(gold,predictions) {
  const r=align(gold,predictions);
  return {records:r.length,overall:metrics(r),per_field:Object.fromEntries(FIELDS.map((k,i)=>{const {raw_exact_pct,...m}=metrics(r.map(x=>({...x,...x.fields[i]})));return [k,m];})),canonical_exact_pct:100*r.reduce((s,x)=>s+x.canonical_exact,0)/r.length,schema_valid:r.filter(x=>x.status==='schema_valid').length,schema_invalid:r.filter(x=>x.status==='schema_invalid').length,missing_or_null:r.filter(x=>x.missing).length,parse_failed:r.filter(x=>!x.missing&&x.status==='parse_failed').length,truncated:r.filter(x=>x.truncated).length};
}
export function quantile(sorted,q){if(!sorted.length||q<0||q>1)throw Error('Invalid percentile');const p=(sorted.length-1)*q,i=Math.floor(p);return sorted[i]+(sorted[Math.min(i+1,sorted.length-1)]-sorted[i])*(p-i);}
// Fixed seed 42 PCG64 XSL-RR state, as used by NumPy default_rng(42).
// Each uint64 provides low then high uint32; Lemire bounded integers avoid modulo bias.
export function seed42Generator(n){
  if(!Number.isInteger(n)||n<1||n>0x1fffff)throw Error('Invalid record count');
  let state=274674114334540486603088602300644985544n,buffer=null;
  const increment=332724090758049132448979897138935081983n,mask128=(1n<<128n)-1n,mask64=(1n<<64n)-1n,multiplier=47026247687942121848144207491837523525n;
  function next32(){if(buffer!==null){const v=buffer;buffer=null;return v;}state=(state*multiplier+increment)&mask128;const z=((state>>64n)^state)&mask64,rot=state>>122n,raw=((z>>rot)|(z<<((64n-rot)&63n)))&mask64;buffer=Number(raw>>32n);return Number(raw&0xffffffffn);}
  const two32=4294967296,threshold=two32%n;
  return ()=>{if(n===1)return 0;let m=next32()*n;while(m%two32<threshold)m=next32()*n;return Math.floor(m/two32);};
}
export function compare(gold,a,b,{resamples=50000,alpha=0.05,familySize=null,margin=1}={}) {
  if(!Number.isInteger(resamples)||resamples<1)throw Error('Invalid resamples');if(!(alpha>0&&alpha<1)||!(margin>0&&Number.isFinite(margin)))throw Error('Invalid alpha or margin');if(familySize!==null&&(!Number.isInteger(familySize)||familySize<1))throw Error('Invalid family size');
  const ar=align(gold,a),br=align(gold,b),am=metrics(ar),bm=metrics(br),sample=seed42Generator(gold.length),d=new Float64Array(resamples);
  for(let r=0;r<resamples;r++){let at=0,bt=0,ag=0,ap=0,bp=0;for(let j=0;j<gold.length;j++){const k=sample();at+=ar[k].tp;bt+=br[k].tp;ag+=ar[k].gold;ap+=ar[k].pred;bp+=br[k].pred;}d[r]=(ag+ap?200*at/(ag+ap):0)-(ag+bp?200*bt/(ag+bp):0);}
  d.sort();const lower=familySize===null?null:quantile(d,alpha/familySize);
  return {records:gold.length,metric:'canonical micro-F1',difference:'A minus B, percentage points',metrics_a:am,metrics_b:bm,delta_f1_pp:am.f1_pct-bm.f1_pct,ci95_f1_pp:[quantile(d,.025),quantile(d,.975)],bootstrap:{resamples,seed:42,unit:'record',percentile_method:'linear'},noninferiority:{enabled:familySize!==null,family_divisor:familySize,alpha,margin_pp:margin,lower_bound_pp:lower,passed:lower===null?null:lower>-margin},scope:'The 95% interval is exploratory. A family-adjusted cutoff does not correct adaptive study choices.'};
}
