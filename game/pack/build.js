const fs=require('fs'),path=require('path');
const SRC=process.argv[2], OUT=process.argv[3];
const PAGES=['Main','Encyclopedia','Character','StyleGuide','Workshop'];
const MAINONLY=process.env.MAINONLY==='1';
const pages=MAINONLY?['Main']:PAGES;
const nameOf=(n)=>n==='Main'?'Agon':n;
// every component any page imports
const comps=new Set(); const read=(n)=>fs.readFileSync(path.join(SRC,n+'.dc.html'),'utf8');
const need=(n)=>{ if(comps.has(n))return; comps.add(n); const s=read(n); for(const m of s.matchAll(/<dc-import[^>]*\bname="([^"]+)"/g)) need(m[1]); };
pages.forEach(need);
const art=fs.readFileSync(path.join(SRC,'art.js'),'utf8');
const rt=fs.readFileSync(path.join(__dirname,'dc-runtime.js'),'utf8');
const J=(x)=>JSON.stringify(x).replace(/</g,'\\u003c').split(String.fromCharCode(8232)).join('\\u2028').split(String.fromCharCode(8233)).join('\\u2029');
const parse=(n)=>{ const s=read(n); const script=/<script type="text\/x-dc"[^>]*>([\s\S]*?)<\/script>/.exec(s)[1]; const dp=/data-props='([^']*)'/.exec(s); let props={}; if(dp){ try{ const o=JSON.parse(dp[1]); for(const k in o) if(k[0]!=='$'&&o[k]&&o[k].default!==undefined) props[k]=o[k].default; }catch(e){} }
  const tplAll=/<x-dc>([\s\S]*?)<\/x-dc>/.exec(s)[1]; const helmet=(/<helmet>([\s\S]*?)<\/helmet>/.exec(tplAll)||[])[1]||''; const tpl=tplAll.replace(/<helmet>[\s\S]*?<\/helmet>/,'');
  return {script,props,tpl,helmet}; };
const links=(t)=>t.replace(/href="([A-Za-z]+)\.dc\.html"/g,(m,n)=>pages.includes(n)?'href="'+nameOf(n)+'.html"':m);
for(const pg of pages){
  const defs=[...comps].map((n)=>{const p=parse(n); return {n,...p,tpl:links(p.tpl)};});
  const helm=defs.filter(d=>d.n===pg||pg===d.n).map(d=>d.helmet).join('\n')+defs.filter(d=>d.n!==pg).map(d=>d.helmet).join('\n');
  const html='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+(pg==='Main'?'Agon':'Agon - '+pg)+'</title>'+helm+'<style>html,body{margin:0;background:#1c110b}button,input,select,textarea,a,label,.dock{pointer-events:auto}#agon-root{overflow:clip !important}#app{display:flex;justify-content:center;align-items:flex-start;min-width:fit-content}</style></head><body><div id="app"></div>'
   +'<script>'+art.replace(/<\/script/gi,'<\\/script')+'</script><script>'+rt+'</script>'
   +'<script>(function(){var D='+J(defs.map(d=>({n:d.n,s:d.script,t:d.tpl,p:d.props})))+';try{D.forEach(function(d){DC.define(d.n,d.s,d.t,d.p);});window.__agon=DC.mount('+J(pg)+',document.getElementById("app"));}catch(e){document.body.innerHTML="<pre style=\\"color:#f88;padding:20px;white-space:pre-wrap\\">Startup error: "+(e&&e.stack||e)+"</pre>";throw e;}})();</script></body></html>';
  fs.writeFileSync(path.join(OUT,nameOf(pg)+'.html'),html); console.log(nameOf(pg)+'.html',(html.length/1024|0)+' KB');
}
