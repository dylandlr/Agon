/* Minimal standalone runtime for Agon's Design Component pages: templates ({{x}}, sc-for, sc-if, dc-import), state, events, DOM patching. */
(function () {
  'use strict';
  const REG = {}, CACHE = new Map(); let uid = 0;
  class DCLogic {
    constructor(props) { this.props = props || {}; this.state = {}; this.__host = null; }
    setState(o) { if (typeof o === 'function') o = o(this.state); if (o) Object.assign(this.state, o); if (this.__host) this.__host.schedule(); }
    forceUpdate() { if (this.__host) this.__host.schedule(); }
  }
  function look(scope, path) {
    path = path.trim();
    if (/^-?[\d.]+$/.test(path)) return +path;
    if (path === 'true') return true; if (path === 'false') return false;
    const parts = path.split('.'); let v;
    for (let i = scope.length - 1; i >= 0; i--) { if (scope[i] != null && parts[0] in scope[i]) { v = scope[i]; break; } }
    if (v === undefined) return undefined;
    let cur = v; for (const p of parts) { cur = cur == null ? undefined : cur[p]; }
    return cur;
  }
  const WHOLE = /^\{\{([^{}]*)\}\}$/, ANY = /\{\{(.*?)\}\}/g;
  const interp = (s, scope) => s.replace(ANY, (m, p) => { const v = look(scope, p); return v === undefined || v === null ? '' : v; });
  const dec = (s) => s.split('&quot;').join('"');
  function define(name, script, tpl, props) {
    const C = new Function('DCLogic', 'window', script + ';return Component;')(DCLogic, window);
    const t = document.createElement('template'); t.innerHTML = tpl;
    REG[name] = { C, nodes: t.content, defaults: props || {} };
  }
  // ---- template -> virtual nodes
  function build(nodes, scope, out, cx) {
    for (const n of nodes.childNodes ? nodes.childNodes : nodes) {
      if (n.nodeType === 3) { out.push({ k: 't', s: interp(n.data, scope) }); continue; }
      if (n.nodeType !== 1) continue;
      const tag = n.localName;
      if (tag === 'helmet') continue;
      if (tag === 'sc-for') {
        const m = WHOLE.exec(n.getAttribute('list') || ''), list = (m && look(scope, m[1])) || [], as = n.getAttribute('as');
        for (const it of list) build(n, scope.concat([{ [as]: it }]), out, cx);
        continue;
      }
      if (tag === 'sc-if') { const m = WHOLE.exec(n.getAttribute('value') || ''); if (m && look(scope, m[1])) build(n, scope, out, cx); continue; }
      if (tag === 'dc-import') {
        const C = REG[n.getAttribute('name')]; if (!C) continue; const props = Object.assign({}, C.defaults), pn = C.lc || (C.lc = Object.keys(C.defaults).reduce((o, k) => (o[k.toLowerCase()] = k, o), {}));
        for (const a of n.attributes) { if (a.name === 'name' || a.name === 'hint-size') continue; const m = WHOLE.exec(a.value); props[pn[a.name] || a.name] = m ? look(scope, m[1]) : dec(a.value); }
        let key = null; try { let ok = true; const js = JSON.stringify(props, (k, v) => { if (typeof v === 'function') ok = false; return v; }); if (ok) key = n.getAttribute('name') + '|' + js; } catch (e) { }
        let hit = key !== null ? CACHE.get(key) : null;
        if (!hit) { const inst = new C.C(props), vals = inst.renderVals(); hit = []; build(C.nodes, [vals], hit, { id: 'u' + (uid++) }); if (key !== null) { if (CACHE.size > 3000) CACHE.clear(); CACHE.set(key, hit); } }
        for (const x of hit) out.push(x);
        continue;
      }
      const v = { k: 'e', tag, ns: n.namespaceURI, a: {}, e: {}, c: [] };
      for (const a of n.attributes) {
        const nm = a.name; if (nm.startsWith('hint-')) continue;
        const m = WHOLE.exec(a.value); let val;
        if (/^on[a-z]+$/i.test(nm)) { val = m ? look(scope, m[1]) : undefined; if (typeof val === 'function') v.e[nm.slice(2).toLowerCase()] = val; continue; }
        val = m ? look(scope, m[1]) : interp(a.value, scope);
        if (val === undefined || val === null || val === false) { if (!(m && val === false && nm.startsWith('aria-'))) continue; }
        val = String(val);
        if (cx && cx.id) { if (nm === 'id' && val === 'frame') val = cx.id; else if (val.indexOf('#frame') >= 0) val = val.split('url(#frame)').join('url(#' + cx.id + ')'); }
        v.a[nm] = val;
      }
      build(n, scope, v.c, cx); out.push(v);
    }
  }

  // ---- compiled template: each template node is analysed once (attribute kinds, paths) so a render only evaluates values
  const PCACHE = new Map();
  function parsePath(path) {
    path = path.trim(); let p = PCACHE.get(path); if (p) return p;
    if (/^-?[\d.]+$/.test(path)) p = { lit: true, v: +path }; else if (path === 'true') p = { lit: true, v: true }; else if (path === 'false') p = { lit: true, v: false };
    else { const parts = path.split('.'); p = { lit: false, parts, first: parts[0] }; }
    PCACHE.set(path, p); return p;
  }
  function lookP(scope, p) {
    if (p.lit) return p.v;
    let v; const f = p.first;
    for (let i = scope.length - 1; i >= 0; i--) { const sc = scope[i]; if (sc != null && f in sc) { v = sc; break; } }
    if (v === undefined) return undefined;
    let cur = v; const parts = p.parts; for (let j = 0; j < parts.length; j++) { cur = cur == null ? undefined : cur[parts[j]]; }
    return cur;
  }
  // interpolated string -> array of literal / path pieces
  function parseInterp(s) {
    const out = []; let last = 0; ANY.lastIndex = 0; let m;
    while ((m = ANY.exec(s))) { if (m.index > last) out.push(s.slice(last, m.index)); out.push(parsePath(m[1])); last = m.index + m[0].length; }
    if (last < s.length) out.push(s.slice(last));
    return out;
  }
  function runInterp(pieces, scope) {
    let r = '';
    for (let i = 0; i < pieces.length; i++) { const x = pieces[i]; if (typeof x === 'string') r += x; else { const v = lookP(scope, x); r += v === undefined || v === null ? '' : v; } }
    return r;
  }
  function compile(n) {
    let c = n.__dc; if (c) return c;
    if (n.nodeType === 3) { const d = n.data; c = d.indexOf('{{') < 0 ? { t: 'text', s: d } : { t: 'itext', p: parseInterp(d) }; n.__dc = c; return c; }
    if (n.nodeType !== 1) { c = { t: 'skip' }; n.__dc = c; return c; }
    const tag = n.localName;
    if (tag === 'helmet') c = { t: 'skip' };
    else if (tag === 'sc-for') { const m = WHOLE.exec(n.getAttribute('list') || ''); c = { t: 'for', lp: m ? parsePath(m[1]) : null, as: n.getAttribute('as'), kids: null, node: n }; }
    else if (tag === 'sc-if') { const m = WHOLE.exec(n.getAttribute('value') || ''); c = { t: 'if', vp: m ? parsePath(m[1]) : null, kids: null, node: n }; }
    else if (tag === 'dc-import') c = { t: 'import', node: n };
    else {
      const attrs = [];
      for (const a of n.attributes) {
        const nm = a.name; if (nm.startsWith('hint-')) continue;
        const m = WHOLE.exec(a.value);
        if (/^on[a-z]+$/i.test(nm)) { attrs.push({ k: 'ev', nm: nm.slice(2).toLowerCase(), p: m ? parsePath(m[1]) : null }); continue; }
        if (m) attrs.push({ k: 'whole', nm, p: parsePath(m[1]), aria: nm.startsWith('aria-') });
        else if (a.value.indexOf('{{') < 0) attrs.push({ k: 'static', nm, v: a.value, frame: nm === 'id' ? a.value === 'frame' : false, hasFrame: a.value.indexOf('#frame') >= 0 });
        else attrs.push({ k: 'interp', nm, p: parseInterp(a.value), nmId: nm === 'id' });
      }
      c = { t: 'el', tag, ns: n.namespaceURI, attrs, kids: null, node: n };
    }
    n.__dc = c; return c;
  }
  const kidsOf = (c) => { if (c.kids) return c.kids; const list = [], cn = c.node.childNodes; for (let i = 0; i < cn.length; i++) list.push(compile(cn[i])); return (c.kids = list); };
  function buildList(list, scope, out, cx) {
    for (let i = 0; i < list.length; i++) {
      const c = list[i];
      switch (c.t) {
        case 'skip': break;
        case 'text': out.push({ k: 't', s: c.s }); break;
        case 'itext': out.push({ k: 't', s: runInterp(c.p, scope) }); break;
        case 'for': {
          const l = (c.lp && lookP(scope, c.lp)) || [], kd = kidsOf(c);
          for (const it of l) buildList(kd, scope.concat([{ [c.as]: it }]), out, cx);
          break;
        }
        case 'if': if (c.vp && lookP(scope, c.vp)) buildList(kidsOf(c), scope, out, cx); break;
        case 'import': build({ childNodes: [c.node] }, scope, out, cx); break;
        default: {
          const v = { k: 'e', tag: c.tag, ns: c.ns, a: {}, e: {}, c: [] }, at = c.attrs;
          for (let j = 0; j < at.length; j++) {
            const a = at[j]; let val;
            if (a.k === 'ev') { val = a.p ? lookP(scope, a.p) : undefined; if (typeof val === 'function') v.e[a.nm] = val; continue; }
            if (a.k === 'static') { val = a.v; if (cx && cx.id) { if (a.frame) val = cx.id; else if (a.hasFrame) val = val.split('url(#frame)').join('url(#' + cx.id + ')'); } v.a[a.nm] = val; continue; }
            if (a.k === 'whole') { val = lookP(scope, a.p); if (val === undefined || val === null || val === false) { if (!(val === false && a.aria)) continue; } }
            else val = runInterp(a.p, scope);
            val = String(val);
            if (cx && cx.id) { if (a.nm === 'id' && val === 'frame') val = cx.id; else if (val.indexOf('#frame') >= 0) val = val.split('url(#frame)').join('url(#' + cx.id + ')'); }
            v.a[a.nm] = val;
          }
          buildList(kidsOf(c), scope, v.c, cx); out.push(v);
        }
      }
    }
  }
  function buildFast(nodes, scope, out, cx) {
    const cn = nodes.childNodes ? nodes.childNodes : nodes, list = [];
    for (let i = 0; i < cn.length; i++) list.push(compile(cn[i]));
    buildList(list, scope, out, cx);
  }
  // ---- patch the real DOM
  const same = (el, v) => v.k === 't' ? el.nodeType === 3 : el.nodeType === 1 && el.localName === v.tag && el.namespaceURI === v.ns;
  const FRESH = [];
  // elements that were just (re)built keep the shared clock phase of their looping animations, so a rebuilt walk cycle does not restart from frame 0
  function syncFresh() {
    if (!FRESH.length) return; const list = FRESH.splice(0); if (document.body.classList.contains('agon-paused')) return;
    for (const el of list) { if (!el.isConnected) continue; try { for (const an of el.getAnimations()) if (typeof CSSAnimation !== 'undefined' && an instanceof CSSAnimation && an.playState === 'running') an.startTime = 0; } catch (e) { } }
  }
  function make(v) {
    if (v.k === 't') return document.createTextNode(v.s);
    const el = document.createElementNS(v.ns, v.tag); el.__a = {}; el.__e = {}; apply(el, v); if (v.a.style && v.a.style.indexOf('infinite') >= 0) FRESH.push(el); kids(el, v.c); return el;
  }
  function apply(el, v) {
    const old = el.__a;
    for (const k in v.a) { const val = v.a[k]; if (old[k] !== val) { try { el.setAttribute(k, val); } catch (e) { } old[k] = val; if (k === 'value' && 'value' in el && el.value !== val) el.value = val; if (k === 'checked') el.checked = true; } }
    for (const k in old) if (!(k in v.a)) { el.removeAttribute(k); delete old[k]; if (k === 'checked') el.checked = false; }
    const oe = el.__e;
    for (const k in v.e) { if (oe[k] !== v.e[k]) { el['on' + k] = v.e[k]; oe[k] = v.e[k]; } }
    for (const k in oe) if (!(k in v.e)) { el['on' + k] = null; delete oe[k]; }
  }
  function kids(parent, list) {
    let i = 0; const cn = parent.childNodes;
    for (; i < list.length; i++) {
      const v = list[i], ex = cn[i];
      if (!ex) { parent.appendChild(make(v)); continue; }
      if (!same(ex, v)) { parent.replaceChild(make(v), ex); continue; }
      if (v.k === 't') { if (ex.data !== v.s) ex.data = v.s; } else { apply(ex, v); kids(ex, v.c); }
    }
    while (parent.childNodes.length > list.length) parent.removeChild(parent.lastChild);
  }
  function mount(name, container, props) {
    const C = REG[name], inst = new C.C(Object.assign({}, C.defaults, props || {}));
    const host = { raf: 0, schedule() { if (!this.raf) this.raf = requestAnimationFrame(() => { this.raf = 0; this.render(); }); }, render() { const vals = inst.renderVals(), out = []; buildFast(C.nodes, [vals], out, null); kids(container, out); syncFresh(); } };
    inst.__host = host; host.render(); if (inst.componentDidMount) inst.componentDidMount(); host.schedule();
    return inst;
  }
  window.DCLogic = DCLogic; window.DC = { define, mount, REG, __build: build, __buildFast: buildFast, __kids: kids };
})();
