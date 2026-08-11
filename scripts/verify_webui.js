const fs = require('fs');
const html = fs.readFileSync('EmindS1/webui/index.html', 'utf8');
const m = html.match(/<script>([\s\S]*?)<\/script>/);
try { new Function(m[1]); console.log('JS_SYNTAX_OK'); } catch (e) { console.error('JS_SYNTAX_FAIL:', e.message); process.exit(1); }
const checks = [
  ['fetch /chat', html.includes("fetch('/chat'")],
  ['stream:true', html.includes('stream:true')],
  ['temperature', html.includes('temperature:+')],
  ['top_p', html.includes('top_p:+')],
  ['top_k', html.includes('top_k:+')],
  ['rep_penalty', html.includes('repetition_penalty:+')],
  ['max_new_tokens', html.includes('max_new_tokens:+')],
  ['d.done meta', html.includes('d.done')],
  ['DONE sentinel', html.includes("'[DONE]'")],
  ['AbortController', html.includes('AbortController')],
  ['topK DOM', html.includes('id="topK"')],
  ['repP DOM', html.includes('id="repP"')],
];
let ok = true;
checks.forEach(([n, p]) => { console.log((p ? 'PASS' : 'FAIL') + '  ' + n); if (!p) ok = false; });
console.log(ok ? 'ALL PASS' : 'HAS FAILURES');
process.exit(ok ? 0 : 1);
