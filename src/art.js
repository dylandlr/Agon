/* Agon art kit: one source for the palettes, expressions and motion that the map pawns,
   the portraits, the animals and the world tiles all share. Mirrors the color tokens in the
   Agon design system (outline, soil, grass, skin-1..6, hair-1..5, cloth-1..5, ...). */
(function () {
  var INK = '#1c110b';
  function rgb(hex) { var n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; }
  function hex(c) { return '#' + c.map(function (v) { return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0'); }).join(''); }
  /* one cel step darker (f < 1) or lighter (f > 1) */
  function shade(h, f) { return hex(rgb(h).map(function (v) { return v * f; })); }
  /* mix toward white by t, for highlight bands on dark colors */
  function tint(h, t) { return hex(rgb(h).map(function (v) { return v + (255 - v) * t; })); }
  /* mix two colors */
  function mix(a, b, t) { var x = rgb(a), y = rgb(b); return hex([0, 1, 2].map(function (i) { return x[i] + (y[i] - x[i]) * t; })); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  /* gene multiplier to [-1, 1] on a log2 scale */
  function lg(v) { return clamp(Math.log2(Math.max(0.25, v || 1)), -1, 1); }
  function pick(list, a) { return list[Math.min(list.length - 1, Math.floor(clamp(a, 0, 0.999) * list.length))]; }
  var SKIN = ['#f6d7bd', '#f1c9a5', '#e3b68c', '#d9a679', '#c68e63', '#b57e55', '#9c6b48', '#8c5a3b', '#6b4430', '#4a2e22'];
  var HAIRC = [['#1c110b', 5], ['#2e1c12', 4], ['#4a2e22', 4], ['#6a4328', 3], ['#8a5a2b', 3], ['#a4562a', 2], ['#b8452a', 1.2], ['#c9a45a', 2], ['#e0c987', 1.4], ['#ece3c4', 0.6], ['#9a9a94', 1.2], ['#e6e6e0', 0.8], ['#2f7f86', 0.35], ['#7a3f86', 0.35], ['#b84a78', 0.3]];
  var HAIR = HAIRC.map(function (h) { return h[0]; });
  var PANTS = ['#5a4a36', '#3e4a3a', '#4a3e56', '#6a5a44', '#2e3a4e', '#524030', '#6a4a3a', '#44524a'], BOOTS = ['#2a1d15', '#4a3322', '#3a3a30', '#5a3a24', '#2a2a36'];
  var CLOTH = ['#2f5f93', '#2f5d45', '#8a4a2a', '#9a8a24', '#4e544b', '#7a2f3a', '#5a3f7a', '#2f7f86', '#a4562a', '#6f7f3a', '#3c4a6e', '#8c7a5a', '#b8452a', '#c9a45a', '#4a6a4a', '#74808c', '#6a3a28', '#a8a090'];
  var TRIM = ['#e8dcc0', '#1c110b', '#c9a45a', '#b8452a', '#2f5f93', '#ece3c4', '#6f7f3a', '#7a2f3a'];
  var HAIRNAMES = ['Cropped', 'Long', 'Topknot', 'Bald', 'Mohawk', 'Bun', 'Ponytail', 'Curly', 'Bob', 'Spiky', 'Braid', 'Buzz'];
  var OUTFITNAMES = ['Plain tunic', 'Striped', 'Sash', 'Vest', 'Collar', 'Hem trim', 'Patched', 'Cloak', 'Dotted'];
  var BEARDNAMES = ['None', 'Stubble', 'Full beard', 'Moustache', 'Goatee', 'Long beard', 'Chinstrap', 'Van Dyke'];
  var ACCNAMES = ['None', 'Headband', 'Beanie'];
  /* deterministic hash of the three appearance channels, so one agent is always the same person */
  function hsh(a0, a1, a2, n) { var x = Math.sin(a0 * 127.1 + a1 * 311.7 + a2 * 74.7 + n * 19.19) * 43758.5453; return x - Math.floor(x); }
  function wpick(list, u) { var t = 0, i; for (i = 0; i < list.length; i++) t += list[i][1]; u *= t; for (i = 0; i < list.length; i++) { u -= list[i][1]; if (u < 0) return list[i][0]; } return list[list.length - 1][0]; }

  /* ---------- appearance genes. A genotype is 32 numbers: 16 loci of two alleles each (a locus is the pair at 2k, 2k+1).
     k0-2 skin (additive, three loci), k3 hair pigment (dark is dominant), k4 red hair (recessive), k5 hair texture (incomplete dominance),
     k6 baldness (dominant in men, recessive in women), k7-14 the eight face measures (additive), k15 beard density (men).
     A child takes one allele from each parent's pair at every locus, with rare mutation. ---------- */
  var G_BIN = { 4: 1, 6: 1 }, FACE_KEYS = ['az', 'po', 'es', 'bw', 'bpo', 'bl', 'nose', 'mw'];
  var FACE_RNG = { az: [21, 28], po: [89, 95], es: [0.85, 1.35], bw: [0.9, 1.9], bpo: [67, 75], bl: [1.3, 2.2], nose: [0.75, 1.55], mw: [0.8, 1.4] };
  function r2(v) { return Math.round(v * 100) / 100; }
  function g01(v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
  function nz(r) { return r() + r() + r() - 1.5; }
  function founderGe(r) {
    var g = [], i, anc = r(), eu = g01(0.35 + 0.55 * anc + (r() - 0.5) * 0.7), tx = r(), fq = 0.2 * (1 - anc * 0.8);
    for (i = 0; i < 6; i++) g.push(g01(anc + nz(r) * 0.22));
    g.push(g01(eu + (r() - 0.5) * 0.2), g01(eu + (r() - 0.5) * 0.2));
    g.push(r() < fq ? 1 : 0, r() < fq ? 1 : 0);
    g.push(g01(tx + (r() - 0.5) * 0.3), g01(tx + (r() - 0.5) * 0.3));
    g.push(r() < 0.2 ? 1 : 0, r() < 0.2 ? 1 : 0);
    for (i = 0; i < 8; i++) { var fc = r(); g.push(g01(fc + (r() - 0.5) * 0.25), g01(fc + (r() - 0.5) * 0.25)); }
    var bd = r(); g.push(g01(bd + (r() - 0.5) * 0.3), g01(bd + (r() - 0.5) * 0.3));
    return g.map(r2);
  }
  function breedGe(m, d, r) {
    var c = [], k, a, b;
    for (k = 0; k < 16; k++) {
      a = m[2 * k + (r() < 0.5 ? 0 : 1)]; b = d[2 * k + (r() < 0.5 ? 0 : 1)];
      if (r() < 0.012) a = G_BIN[k] ? 1 - a : g01(a + nz(r) * 0.3);
      if (r() < 0.012) b = G_BIN[k] ? 1 - b : g01(b + nz(r) * 0.3);
      if (r() < 0.5) c.push(r2(a), r2(b)); else c.push(r2(b), r2(a));
    }
    return c;
  }
  function geStr(g) { return g && g.length === 32 ? g.join(',') : ''; }
  function parseGe(s) { if (!s) return null; if (Array.isArray(s)) return s.length === 32 ? s : null; var a = String(s).split(',').map(Number); return a.length === 32 && a.every(function (v) { return v === v; }) ? a : null; }
  var HAIR_STOPS = [[0, '#e8dfc0'], [0.18, '#d9bd7a'], [0.34, '#b98a48'], [0.5, '#8a5a2b'], [0.66, '#5a3a24'], [0.82, '#2e1c12'], [1, '#1c110b']];
  function hairFromEu(eu, red) {
    if (red) return mix('#c9743a', '#8f3a1e', eu);
    for (var i = 1; i < HAIR_STOPS.length; i++) if (eu <= HAIR_STOPS[i][0]) return mix(HAIR_STOPS[i - 1][1], HAIR_STOPS[i][1], (eu - HAIR_STOPS[i - 1][0]) / (HAIR_STOPS[i][0] - HAIR_STOPS[i - 1][0]));
    return HAIR_STOPS[HAIR_STOPS.length - 1][1];
  }
  function phenotype(ge, sex) {
    var i, sk = 0, fc = {}, M = function (a) { return (ge[a] + ge[a + 1]) / 2; };
    for (i = 0; i < 6; i++) sk += ge[i]; sk /= 6;
    var eu = 0.65 * Math.max(ge[6], ge[7]) + 0.35 * Math.min(ge[6], ge[7]), red = ge[8] > 0.5 && ge[9] > 0.5, curl = M(10);
    var bn = (ge[12] > 0.5 ? 1 : 0) + (ge[13] > 0.5 ? 1 : 0), male = sex === 'M';
    FACE_KEYS.forEach(function (k, j) { var q = FACE_RNG[k]; fc[k] = q[0] + (q[1] - q[0]) * M(14 + 2 * j); });
    return { skin: sk, eu: eu, red: red, redCarrier: !red && (ge[8] > 0.5 || ge[9] > 0.5), hair: hairFromEu(eu, red), curl: curl, texture: curl < 0.38 ? 'straight' : curl < 0.66 ? 'wavy' : 'curly',
      balding: male ? bn >= 1 : bn >= 2, baldCarrier: bn === 1 && !male, face: fc, beardD: M(30) };
  }
  var STYLE_SETS = { straight: [0, 1, 2, 4, 5, 6, 8, 9, 10, 11], wavy: [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], curly: [7, 7, 7, 0, 2, 5, 10, 11, 4, 9] };
  var SKINN = ['very light', 'light', 'fair', 'olive', 'tan', 'brown', 'dark brown', 'deep brown'];
  var HAIRN2 = function (p) { if (p.red) return 'ginger'; var e = p.eu; return e < 0.12 ? 'platinum' : e < 0.26 ? 'blond' : e < 0.42 ? 'dark blond' : e < 0.58 ? 'brown' : e < 0.74 ? 'dark brown' : e < 0.9 ? 'very dark brown' : 'black'; };
  function geRows(ge, sex) {
    if (!ge) return []; var p = phenotype(ge, sex), c = function (v) { return (v > 0.5 ? 'D' : 'r'); };
    return [{ k: 'Skin tone', v: SKINN[Math.min(SKINN.length - 1, Math.floor(p.skin * SKINN.length))] + ' (' + p.skin.toFixed(2) + ')' },
      { k: 'Hair colour', v: HAIRN2(p) + (p.redCarrier ? ', carries red' : '') },
      { k: 'Hair texture', v: p.texture + ' (' + p.curl.toFixed(2) + ')' },
      { k: 'Balding', v: p.balding ? 'prone' : p.baldCarrier ? 'carrier' : 'none' }].concat(sex === 'M' ? [{ k: 'Beard growth', v: p.beardD < 0.3 ? 'sparse' : p.beardD < 0.6 ? 'average' : 'heavy' }] : []);
  }
  /* animal coat genes: 10 numbers, five loci. k0 tone, k1 warm/cool, k2 marking contrast (dominant), k3 melanism (dominant), k4 albinism (recessive). */
  function founderCoat(r) {
    var t = g01(0.5 + nz(r) * 0.3), h = g01(0.5 + nz(r) * 0.3), p = g01(0.5 + nz(r) * 0.3), g = [];
    g.push(t + nz(r) * 0.08, t + nz(r) * 0.08, h + nz(r) * 0.08, h + nz(r) * 0.08, p + nz(r) * 0.08, p + nz(r) * 0.08);
    g.push(r() < 0.03 ? 1 : 0, r() < 0.03 ? 1 : 0, r() < 0.04 ? 1 : 0, r() < 0.04 ? 1 : 0);
    return g.map(function (v, i) { return i < 6 ? r2(g01(v)) : v; });
  }
  function breedCoat(m, d, r) {
    if (!m || !d) return m || d || founderCoat(r); var c = [], k, a, b;
    for (k = 0; k < 5; k++) {
      a = m[2 * k + (r() < 0.5 ? 0 : 1)]; b = d[2 * k + (r() < 0.5 ? 0 : 1)];
      if (r() < 0.01) a = k >= 3 ? 1 - a : g01(a + nz(r) * 0.3); if (r() < 0.01) b = k >= 3 ? 1 - b : g01(b + nz(r) * 0.3);
      c.push(k >= 3 ? a : r2(a), k >= 3 ? b : r2(b));
    }
    return c;
  }
  function coatKey(c) { return c && c.length === 10 ? c.join(',') : ''; }
  function coatPheno(c) { var a = parseGe(0, 0) || null; c = typeof c === 'string' ? c.split(',').map(Number) : c; if (!c || c.length !== 10) return null; return { tone: (c[0] + c[1]) / 2, hue: (c[2] + c[3]) / 2, pat: Math.max(c[4], c[5]) * 0.6 + Math.min(c[4], c[5]) * 0.4, mel: c[6] > 0.5 || c[7] > 0.5, alb: c[8] > 0.5 && c[9] > 0.5 }; }
  function coatPal(P, c) {
    var f = coatPheno(c); if (!f) return P; var Q = {}, k, sh = function (col, m) { return col ? shade(col, m) : col; };
    for (k in P) Q[k] = P[k];
    if (f.alb) { Q.body = '#f1e6e0'; Q.belly = '#fbf3ee'; Q.mark = '#e4d2cc'; Q.ear = '#efc4c0'; Q.muzzle = '#f7e4de'; Q.eye = '#f0a09c'; return Q; }
    var kt = 1 + (f.tone - 0.5) * 0.5, wc = f.hue > 0.5 ? '#a4562a' : '#6a7482', wm = Math.abs(f.hue - 0.5) * 0.5, kp = 1 - (f.pat - 0.5) * 0.6;
    ['body', 'belly', 'ear', 'muzzle', 'horn', 'plate'].forEach(function (n) { if (P[n]) Q[n] = mix(sh(P[n], kt), wc, wm); });
    if (P.mark) Q.mark = mix(sh(P.mark, kt * kp), wc, wm * 0.6);
    if (f.mel) { Q.body = sh(Q.body, 0.36); Q.belly = sh(Q.belly || Q.body, 0.5); Q.mark = sh(Q.mark || Q.body, 0.5); if (Q.ear) Q.ear = sh(Q.ear, 0.45); if (Q.muzzle) Q.muzzle = sh(Q.muzzle, 0.5); }
    return Q;
  }
  /* appearance channels a0, a1, a2 in [0,1) to a look; with a genotype (ge) and sex the genes decide skin, hair colour and texture, face and beard */
  function look(a0, a1, a2, ge, sex, adult) {
    var h = function (n) { return hsh(a0, a1, a2, n); };
    var sf = clamp(a0, 0, 0.999) * (SKIN.length - 1), si = Math.floor(sf), skin = mix(SKIN[si], SKIN[Math.min(SKIN.length - 1, si + 1)], sf - si);
    var hair = h(23) < 0.02 ? HAIRC[12 + Math.floor(h(24) * 3)][0] : wpick(HAIRC.slice(0, 12), h(1)), style = Math.floor(h(2) * HAIRNAMES.length);
    var cloth = pick(CLOTH, a2), trim = TRIM[Math.floor(h(3) * TRIM.length)];
    if (mix(cloth, trim, 0) === trim) trim = TRIM[0];
    var outfit = Math.floor(h(4) * OUTFITNAMES.length);
    var beard = h(5) < 0.34 ? 1 + Math.floor(h(6) * 7) : 0, hirsute = h(22) < 0.03, acc = h(7) < 0.2 ? 1 : h(7) < 0.36 ? 2 : 0;
    if (acc === 2 && [2, 4, 5, 7, 9].indexOf(style) >= 0) acc = 1;
    var face = { az: 21 + h(11) * 7, po: 89 + h(12) * 6, es: 0.85 + h(13) * 0.5, bw: 0.9 + h(14) * 1.0, bpo: 67 + h(15) * 8, bl: 1.3 + h(16) * 0.9, nose: 0.75 + h(17) * 0.8, mw: 0.8 + h(18) * 0.6 };
    var G = parseGe(ge);
    if (G) {
      var P = phenotype(G, sex || 'M'), sf2 = clamp(P.skin, 0, 0.999) * (SKIN.length - 1), si2 = Math.floor(sf2);
      skin = mix(SKIN[si2], SKIN[Math.min(SKIN.length - 1, si2 + 1)], sf2 - si2); if (h(23) >= 0.02) hair = P.hair;
      var set = STYLE_SETS[P.texture]; style = set[Math.floor(h(2) * set.length)];
      if (P.balding && adult !== false) style = h(2) < 0.6 ? 3 : 11;
      if (acc === 2 && [2, 4, 5, 7, 9].indexOf(style) >= 0) acc = 1;
      beard = sex === 'M' && h(5) < clamp((P.beardD - 0.25) * 1.6, 0, 0.95) ? 1 + Math.floor(h(6) * 7) : 0; if (adult === false) beard = 0;
      face = { az: P.face.az, po: P.face.po, es: P.face.es, bw: P.face.bw, bpo: P.face.bpo, bl: P.face.bl, nose: P.face.nose, mw: P.face.mw };
    }
    return { skin: skin, hair: hair, cloth: cloth, trim: trim, hairStyle: style, outfit: outfit, beard: beard, hirsute: hirsute, acc: acc, face: face, pants: pick(PANTS, h(20)), boots: pick(BOOTS, h(21)) };
  }
  function circP(cx, cy, r) { return 'M' + (cx - r) + ' ' + cy + ' a' + r + ' ' + r + ' 0 1 0 ' + (2 * r) + ' 0 a' + r + ' ' + r + ' 0 1 0 ' + (-2 * r) + ' 0Z'; }
  function ellP(cx, cy, rx, ry) { return 'M' + (cx - rx) + ' ' + cy + ' a' + rx + ' ' + ry + ' 0 1 0 ' + (2 * rx) + ' 0 a' + rx + ' ' + ry + ' 0 1 0 ' + (-2 * rx) + ' 0Z'; }
  function lockP(x, y) { return 'M' + (x + 2.1) + ' ' + y + ' a2.1 2.1 0 0 1 2.1 2.1 V' + (y + 8.9) + ' a2.1 2.1 0 0 1 -4.2 0 V' + (y + 2.1) + ' a2.1 2.1 0 0 1 2.1 -2.1Z'; }
  /* soft shading layers shared by pawn and portrait: light from the top left, so the lower right of every form takes a shade.
     Shapes are in 48x48 pawn space. Same flat style, just with the volume a flat fill lacks. */
  function detailKit(L, c, bw, bh, view, feet) {
    var out = { neck: [], torso: [], skin: [], hair: [], feet: [] }, N = view === 'N', E = view === 'E', cy = 33;
    var Q = function (d, fill, op, stroke, sw) { return { d: d, fill: fill || 'none', stroke: stroke || 'none', sw: sw || 0, op: op == null ? 1 : op }; };
    if (!N) {
      out.skin.push(Q('M15.6 16.4 Q24 19.4 32.4 16.4 L31.8 18.2 Q24 21 16.2 18.2Z', c.skinShade, 0.3));
      if (view === 'S' || view === 'SE') {
        out.skin.push(Q('M31.4 15.6 A8.6 8.6 0 0 1 27.6 27.6 A10.6 10.6 0 0 0 31.4 15.6Z', c.skinShade, 0.4));
        out.skin.push(Q('M24.4 22 q-1 1.6 0.4 2.2', 'none', 0.75, c.skinShade, 0.9));
      }
    }
    var hl = tint(c.hair, 0.28);
    out.hair.push(Q('M17.6 15.2 Q21 12.4 26 12.8', 'none', 0.5, hl, 1.1));
    if (!N) { out.hair.push(Q('M21.4 13.6 q0.8 1.8 0.2 3.4', 'none', 0.5, c.hairShade, 0.8)); out.hair.push(Q('M27 13.2 q0.8 1.8 0.6 3.6', 'none', 0.5, c.hairShade, 0.8)); }
    (feet || [19, 29]).forEach(function (x) { out.feet.push(Q(ellP(x - 1, 40, 2, 0.8), '#ffffff', 0.12)); });
    return out;
  }
  /* hair and headwear as shape lists. back is drawn behind the body, front over the head. bx is the back-of-head x for the facing. */
  function hairKit(L, hair, hs, view, geo) {
    var back = [], front = [], cap = true, capFill = hair, st = L.hairStyle;
    var E = view === 'E', N = view === 'N', bx = E ? 21 : (view === 'S' ? 24 : 22.5);
    var P = function (d, f, sw) { return { d: d, fill: f || hair, stroke: '#1c110b', sw: sw || 1.7 }; };
    if (st === 1) { back.push(P(ellP(E ? 20.5 : 24, 22, E ? 8.6 : 11.4, 10.6), hs, 1.8)); if (geo) { front.push(P(lockP(geo.lockL, 18), hair, 1.6)); if (geo.lockROn) front.push(P(lockP(geo.lockR, 18), hair, 1.6)); } }
    else if (st === 2) back.push(P(circP(bx, 9, 4.2), hair, 1.8));
    else if (st === 3) cap = false;
    else if (st === 4) { cap = false; front.push(P('M' + (bx - 3.4) + ' 15.4 L' + (bx - 2) + ' 6.8 L' + bx + ' 9.6 L' + (bx + 2) + ' 5.8 L' + (bx + 3.4) + ' 15.4Z', hair, 1.7)); }
    else if (st === 5) back.push(P(circP(bx, 8.4, 4.8), hair, 1.8));
    else if (st === 6) { if (N) front.push(P(ellP(24, 30, 3.2, 8), hair, 1.7)); else back.push(P(ellP(E ? 14 : 35.6, 24, 3, 7.4), hair, 1.7)); }
    else if (st === 7) back.push(P(circP(E ? 21 : 24, 16.4, 12.4), hair, 1.9));
    else if (st === 8) back.push(P(ellP(E ? 21 : 24, 21.8, E ? 9 : 10.8, 8.8), hs, 1.8));
    else if (st === 9) back.push(P('M' + (bx - 9) + ' 14.6 L' + (bx - 7) + ' 7 L' + (bx - 4) + ' 12.2 L' + (bx - 1) + ' 5.4 L' + (bx + 2) + ' 12 L' + (bx + 5.4) + ' 6.2 L' + (bx + 7) + ' 12.6 L' + (bx + 9) + ' 14.6Z', hair, 1.7));
    else if (st === 10) { var bxx = N ? 24 : E ? 14.5 : 34; [24.6, 29.4, 34.2].forEach(function (y, i) { (N ? front : back).push(P(ellP(bxx, y + (N ? 4 : 0), 2.4 - i * 0.2, 2.8), i % 2 ? hs : hair, 1.5)); }); }
    else if (st === 11) capFill = hs;
    if (L.acc === 1) front.push({ d: 'M14.7 17.4 Q24 12.6 33.3 17.4', fill: 'none', stroke: L.trim, sw: 3.2 }, { d: 'M14.7 17.4 Q24 12.6 33.3 17.4', fill: 'none', stroke: '#1c110b', sw: 0.5 });
    if (L.acc === 2) front.push({ d: 'M14.6 19 A9.6 9.6 0 0 1 33.4 19 L32.6 14.8 Q24 10.8 15.4 14.8Z', fill: shade(L.cloth, 0.85), stroke: '#1c110b', sw: 1.8 }, { d: 'M15 16.6 Q24 13 33 16.6', fill: 'none', stroke: '#1c110b', sw: 0.9 });
    if (L.acc === 2) cap = false;
    return { cap: cap, capFill: capFill, back: back, front: front };
  }
  /* beard shapes live in the face group so they turn with it */
  function beardKit(L, hair, hs, view) {
    var b = L.beard, f = hs;
    if (!b) return [];
    if (view === 'E') {
      if (b === 1) return [{ d: 'M23.4 23 Q23.4 29.8 29 30.2 Q34 29.6 33.6 24.6 Q30 27.2 26.6 26.4Z', fill: f, stroke: 'none', sw: 0, op: 0.4 }];
      if (b === 2) return [{ d: 'M22.8 22 Q22.4 31.2 29.4 31.6 Q35 30.6 34 24.4 Q30.4 27.8 27 26.8 Q24.6 26.4 22.8 22Z', fill: hair, stroke: '#1c110b', sw: 1.4, op: 1 }];
      if (b === 3) return [{ d: 'M30.6 24.2 Q33.6 24.2 34.6 25.8 Q32 25.8 30.4 25.6Z', fill: hair, stroke: '#1c110b', sw: 1, op: 1 }];
      return [{ d: 'M30.4 27 Q32.4 31 33.6 27.4 Q32 28 30.4 27Z', fill: hair, stroke: '#1c110b', sw: 1.1, op: 1 }];
    }
    if (b === 1) return [{ d: 'M16.4 22.4 Q17.4 29.6 24 30 Q30.6 29.6 31.6 22.4 Q28 27 24 27 Q20 27 16.4 22.4Z', fill: f, stroke: 'none', sw: 0, op: 0.4 }];
    if (b === 2) return [{ d: 'M16 21.8 Q16.6 30.6 24 31.2 Q31.4 30.6 32 21.8 Q28.6 26.8 24 26.6 Q19.4 26.8 16 21.8Z', fill: hair, stroke: '#1c110b', sw: 1.4, op: 1 }];
    if (b === 3) return [{ d: 'M19.6 24.4 Q22 22.8 24 24.2 Q26 22.8 28.4 24.4 Q26 25.8 24 25 Q22 25.8 19.6 24.4Z', fill: hair, stroke: '#1c110b', sw: 1, op: 1 }];
    return [{ d: 'M21.6 27.2 Q24 32 26.4 27.2 Q24 28.2 21.6 27.2Z', fill: hair, stroke: '#1c110b', sw: 1.1, op: 1 }];
  }
  /* clothing pattern over the body ellipse (cx 24, cy 33, rx bw, ry bh) */
  function outfitKit(L, cloth, trim, bw, bh, view) {
    var o = L.outfit, back = [], front = [], cy = 33, NV = view === 'N' || view === 'NE';
    var chord = function (y) { var t = (y - cy) / bh; return bw * Math.sqrt(Math.max(0, 1 - t * t)); };
    var line = function (d, c, w, op) { front.push({ d: d, fill: 'none', stroke: c, sw: w, op: op == null ? 1 : op }); };
    if (NV) {
      /* seen from behind: front-only details (pockets, patches, buttons, chevron collar) are hidden; the sash runs the other way and a cape covers the back */
      if (o === 2) line('M' + (24 + bw * 0.72) + ' ' + (cy - bh * 0.62) + ' L' + (24 - bw * 0.62) + ' ' + (cy + bh * 0.58), trim, 3.2, 0.95);
      else if (o === 7) front.push({ d: ellP(24, cy - 1.2, bw * 0.98, bh * 0.94), fill: trim, stroke: 'none', sw: 0, op: 1 }, { d: 'M24 ' + (cy - bh * 0.8) + ' V' + (cy + bh * 0.7), fill: 'none', stroke: A_shade(trim, 0.7), sw: 0.9, op: 0.8 });
      else if (o === 4) line('M' + (24 - bw * 0.38) + ' ' + (cy - bh * 0.78) + ' Q24 ' + (cy - bh * 0.5) + ' ' + (24 + bw * 0.38) + ' ' + (cy - bh * 0.78), trim, 2.2, 1);
      else if (o === 1 || o === 5) { /* horizontal bands and hems wrap round the body */ }
      if (o === 1) [cy - bh * 0.62, cy - bh * 0.32, cy - bh * 0.02].forEach(function (y) { var c = chord(y) - 0.4; line('M' + (24 - c) + ' ' + y + ' H' + (24 + c), trim, 1.6, 0.9); });
      else if (o === 5) line('M' + (24 - bw * 0.78) + ' ' + (cy + bh * 0.5) + ' Q24 ' + (cy + bh * 1.05) + ' ' + (24 + bw * 0.78) + ' ' + (cy + bh * 0.5), trim, 2, 0.95);
      line('M' + (24 - bw * 0.3) + ' ' + (cy - bh * 0.84) + ' Q24 ' + (cy - bh * 0.6) + ' ' + (24 + bw * 0.3) + ' ' + (cy - bh * 0.84), A_shade(cloth, 0.7), 1.1, 0.8);
      return { back: back, front: front };
    }
    if (o === 1) [cy - bh * 0.62, cy - bh * 0.32, cy - bh * 0.02].forEach(function (y) { var c = chord(y) - 0.4; line('M' + (24 - c) + ' ' + y + ' H' + (24 + c), trim, 1.6, 0.9); });
    else if (o === 2) line('M' + (24 - bw * 0.72) + ' ' + (cy - bh * 0.62) + ' L' + (24 + bw * 0.62) + ' ' + (cy + bh * 0.58), trim, 3.2, 0.95);
    else if (o === 3) { front.push({ d: ellP(24, cy - bh * 0.18, bw * 0.62, bh * 0.5), fill: A_shade(cloth, 0.62), stroke: 'none', sw: 0, op: 0.92 }); line('M24 ' + (cy - bh * 0.68) + ' V' + (cy + bh * 0.28), '#1c110b', 0.9, 0.7); }
    else if (o === 4) line('M' + (24 - bw * 0.5) + ' ' + (cy - bh * 0.8) + ' L24 ' + (cy - bh * 0.3) + ' L' + (24 + bw * 0.5) + ' ' + (cy - bh * 0.8), trim, 2.2, 1);
    else if (o === 5) line('M' + (24 - bw * 0.78) + ' ' + (cy + bh * 0.5) + ' Q24 ' + (cy + bh * 1.05) + ' ' + (24 + bw * 0.78) + ' ' + (cy + bh * 0.5), trim, 2, 0.95);
    else if (o === 6) { front.push({ d: 'M' + (24 - bw * 0.5) + ' ' + (cy - bh * 0.1) + ' h' + (bw * 0.45) + ' v' + (bh * 0.3) + ' h' + (-bw * 0.45) + 'Z', fill: trim, stroke: '#1c110b', sw: 0.7, op: 0.9 }, { d: 'M' + (24 + bw * 0.1) + ' ' + (cy - bh * 0.55) + ' h' + (bw * 0.32) + ' v' + (bh * 0.28) + ' h' + (-bw * 0.32) + 'Z', fill: A_shade(cloth, 0.7), stroke: '#1c110b', sw: 0.7, op: 0.9 }); }
    else if (o === 7) back.push({ d: ellP(24, cy - 1.6, bw * 1.18, bh * 0.92), fill: trim, stroke: '#1c110b', sw: 1.8, op: 1 });
    else if (o === 8) [[-0.45, -0.4], [0.2, -0.5], [-0.15, 0.05], [0.5, 0.1], [-0.5, 0.5], [0.15, 0.55]].forEach(function (q) { front.push({ d: circP(24 + bw * q[0], cy + bh * q[1], 0.9), fill: trim, stroke: 'none', sw: 0, op: 0.9 }); });
    return { back: back, front: front };
  }
  function A_shade(h, f) { return shade(h, f); }

  /* ---------- expressions ----------
     eyes: dot | wide | half | happy | closed | pain | x     brow: inner-end tilt (+ angry, - worried), lift in px
     mouth: smile | grin | flat | frown | o | wave | pant | sleep     extras: blush, sweat, tear, zzz, tint */
  var EXPR = {
    neutral:  { label: 'Neutral',  eyes: 'dot',    tilt: 0,  lift: 0,  mouth: 'calm',  blush: 0.2 },
    happy:    { label: 'Happy',    eyes: 'happy',  tilt: -2, lift: 2,  mouth: 'grin',  blush: 0.6 },
    tired:    { label: 'Tired',    eyes: 'half',   tilt: -3, lift: -1, mouth: 'flat',  blush: 0.15 },
    hungry:   { label: 'Hungry',   eyes: 'dot',    tilt: -6, lift: 1,  mouth: 'wave',  blush: 0.2 },
    thirsty:  { label: 'Thirsty',  eyes: 'half',   tilt: -3, lift: 0,  mouth: 'pant',  blush: 0.3, sweat: true },
    scared:   { label: 'Scared',   eyes: 'wide',   tilt: -7, lift: 4,  mouth: 'o',     blush: 0,   sweat: true },
    angry:    { label: 'Angry',    eyes: 'dot',    tilt: 8,  lift: -2, mouth: 'frown', blush: 0.5 },
    hurt:     { label: 'In pain',  eyes: 'pain',   tilt: -5, lift: 2,  mouth: 'wave',  blush: 0.2, tear: true },
    sick:     { label: 'Sick',     eyes: 'half',   tilt: -4, lift: 0,  mouth: 'wave',  blush: 0,   tint: '#7fbf8e', sweat: true },
    cold:     { label: 'Cold',     eyes: 'closed', tilt: -3, lift: 0,  mouth: 'wave',  blush: 0,   tint: '#7cacdf' },
    sleep:    { label: 'Asleep',   eyes: 'closed', tilt: 0,  lift: 1,  mouth: 'sleep', blush: 0.3, zzz: true },
    dead:     { label: 'Dead',     eyes: 'x',      tilt: 0,  lift: 0,  mouth: 'flat',  blush: 0,   tint: '#aea69a' }
  };
  /* simulation state to the single expression shown. First match wins. */
  var EXPR_ORDER = ['dead', 'sleep', 'hurt', 'scared', 'angry', 'sick', 'cold', 'thirsty', 'hungry', 'tired', 'happy', 'neutral'];
  function expressionFor(a) {
    if (!a.alive && a.alive !== undefined) return 'dead';
    if (a.act === 'sleep') return 'sleep';
    if (a.bub === 'hurt' || a.inj > 0.45) return 'hurt';
    if (a.stress > 0.6 || a.san < 0.2) return 'scared';
    if (a.act === 'strike') return 'angry';
    if (a.infection > 0) return 'sick';
    if (a.temp < -0.7) return 'cold';
    if (a.hyd < 0.3) return 'thirsty';
    if (a.sat < 0.3) return 'hungry';
    if (a.ene < 0.3) return 'tired';
    if (a.sat > 0.7 && a.hyd > 0.7 && a.hea > 0.8 && a.stress < 0.15 && !(a.hap < 0.5)) return 'happy';
    return 'neutral';
  }

  /* ---------- motion ----------
     One tick is 600ms at 1x. Position glides linearly over the tick; body motion loops on its own clock. */
  var MOTION = {
    idle:   { css: 'ag-idle 2.6s ease-in-out infinite',  secs: 2.6,  text: 'Slow breath: 2% stretch on a 2.6s loop.' },
    move:   { css: 'ag-walk 0.3s ease-in-out infinite',  secs: 0.3,  text: 'No whole-body hop: the pawn itself steps, bobs and sways in time with its legs.' },
    run:    { css: 'ag-run 0.22s ease-in-out infinite',  secs: 0.22, text: 'Leans forward 4 degrees; the pawn handles the stride and bob.' },
    eat:    { css: 'ag-eat 0.6s ease-in-out infinite',   secs: 0.6,  text: 'Bend to the food: 3px down and a 6% squash, twice per bite.' },
    drink:  { css: 'ag-drink 0.9s ease-in-out infinite', secs: 0.9,  text: 'Slow deep dip: 4px down, lean 6 degrees toward the water.' },
    sleep:  { css: 'ag-sleep 3.2s ease-in-out infinite', secs: 3.2,  text: 'Slow breathing under the covers, eyes closed, Z rising.' },
    hurt:   { css: 'ag-hurt 0.4s ease-out 2',            secs: 0.4,  text: 'Two quick recoils: 3px side to side with a 4 degree twist.' },
    strike: { css: 'ag-strike 0.6s ease-in-out infinite', secs: 0.6, text: 'Wind up (up 5px, grow 8%), then lunge and snap back.' },
    shiver: { css: 'ag-shiver 0.12s linear infinite',    secs: 0.12, text: 'Fast 1px side-to-side jitter. Pairs with the cold face.' },
    dead2:  { css: 'ag-fall2 1.7s linear forwards',      secs: 1.7,  text: 'Ragdoll onto the back or face: stagger, drop, impact squash and bounce, ending flattened so the body reads as lying on the ground from above.' },
    dead:   { css: 'ag-fall 1.7s linear forwards',      secs: 1.7,  text: 'Ragdoll, four variants chosen per agonist (side left, side right, flat on back, flat on face): jolt back, fall with an impact squash, bounce twice and settle lying down. Head and arms flop on their own timing; the shadow stays on the ground.' },
    hop:    { css: 'ag-hop 0.45s ease-in-out infinite',  secs: 0.45, text: 'Small prey: squash, spring 8px up, land with a squash.' },
    lumber: { css: 'ag-lumber 1.1s ease-in-out infinite', secs: 1.1, text: 'Large prey: heavy 1.5 degree rocking, 1px rise.' },
    prowl:  { css: 'ag-prowl 0.7s ease-in-out infinite', secs: 0.7,  text: 'Predator: low body, slow stalking sway.' },
    slither: { css: 'ag-slither 0.8s ease-in-out infinite', secs: 0.8, text: 'Snake: side-to-side body wave with a slight skew.' },
    pounce: { css: 'ag-pounce 0.5s ease-in-out infinite', secs: 0.5, text: 'Attack: rear back and crouch, lunge 8px at the target with the nose dipping, snap at the bite, recoil and recover.' },
    work:    { css: 'ag-work 0.55s ease-in-out infinite',    secs: 0.55, text: 'Hammer stroke: lift 4px and lean back 5 degrees, then strike down 3px with a 4% squash.' },
    forage:  { css: 'ag-forage 1.2s ease-in-out infinite',   secs: 1.2,  text: 'Stoop to the bush: 5px down and a 10 degree bow, hold, rise.' },
    carry:   { css: 'ag-carry 0.42s ease-in-out infinite',   secs: 0.42, text: 'Loaded walk: the pawn steps shorter. Slower than a plain walk.' },
    wave:    { css: 'ag-wave 0.7s ease-in-out infinite',     secs: 0.7,  text: 'Coaxing or greeting: 6 degree sway left and right with a small lift.' },
    cheer:   { css: 'ag-cheer 0.6s ease-in-out infinite',    secs: 0.6,  text: 'Success jump: squash, spring 9px with a 6 degree twist, land.' },
    stumble: { css: 'ag-stumble 0.9s ease-in-out infinite',  secs: 0.9,  text: 'Unhinged: uneven lurches of up to 3px and 9 degrees. Pairs with sanity below 20%.' }
  };
  var ALIAS = { idle: 'idle', move: 'move', eat: 'eat', drink: 'drink', sleep: 'sleep', hurt: 'hurt', strike: 'strike', dead: 'dead', dead2: 'dead2', run: 'run', shiver: 'shiver', hop: 'hop', lumber: 'lumber', prowl: 'prowl', pounce: 'pounce', slither: 'slither', work: 'work', forage: 'forage', carry: 'carry', wave: 'wave', cheer: 'cheer', stumble: 'stumble' };
  // ground shadow animation paired with a motion: it stays on the ground and shrinks and fades while the body is in the air
  var SHM = { move: 'ag-sh-walk 0.3s ease-in-out infinite', run: 'ag-sh-run 0.22s ease-in-out infinite', hop: 'ag-sh-hop 0.45s ease-in-out infinite', cheer: 'ag-sh-cheer 0.6s ease-in-out infinite', pounce: 'ag-sh-pounce 0.5s ease-in-out infinite', strike: 'ag-sh-strike 0.5s ease-in-out infinite', carry: 'ag-sh-walk 0.42s ease-in-out infinite', stumble: 'ag-sh-walk 0.9s ease-in-out infinite' };
  /* one tick lasts TICK_MS at 1x (RimWorld-like pacing); every locomotion and work loop was authored for 600ms and is stretched to match */
  var TICK_MS = 1000;
  function stretch(css, ms, name) { var k = (ms || TICK_MS) / 600; if (name === 'dead' || name === 'dead2' || k === 1) return css; return css.replace(/(\d*\.?\d+)s\b/, function (m, v) { return (+v * k).toFixed(2) + 's'; }); }
  function shAnim(name, ms) { var c = SHM[ALIAS[name] || 'idle'] || 'none'; return c === 'none' ? c : stretch(c, ms, name); }
  function anim(name, ms) { var m = MOTION[ALIAS[name] || 'idle']; return m ? stretch(m.css, ms, name) : 'none'; }

  /* ---------- combat animation: one pose per action primitive (PRIM) and per combat state, in step with the combat system ----------
     body = whole-body transform steps (L(n) = n px toward the target, using --lx/--ly), atk = the striking arm (and the held weapon), sup = the other arm; angles are in degrees x --sg. */
  var L = function (n, extra) { return 'translate(calc(var(--lx,0) * ' + n + 'px), calc(var(--ly,0) * ' + n + 'px))' + (extra ? ' ' + extra : ''); };
  var CMX = {
    jab:     { body: [[0, 'none'], [22, L(-1.5)], [42, L(3.5)], [100, 'none']],                                   atk: [[0, 0], [20, -30], [40, -85], [62, -85], [100, 0]], sup: [[0, 0], [30, -14], [100, 0]] },
    swing:   { body: [[0, 'none'], [20, L(-1, 'rotate(-4deg)')], [55, L(3, 'rotate(5deg)')], [100, 'none']],          atk: [[0, 0], [20, -125], [45, -145], [60, 35], [100, 0]], sup: [[0, 0], [30, -20], [100, 0]] },
    slash:   { body: [[0, 'none'], [16, L(-1, 'rotate(-5deg)')], [40, L(2.5, 'rotate(6deg)')], [100, 'none']],        atk: [[0, 0], [15, -135], [40, 45], [60, 15], [100, 0]], sup: [[0, 0], [40, 14], [100, 0]] },
    thrust:  { body: [[0, 'none'], [26, L(-2)], [52, L(5)], [74, L(3)], [100, 'none']],                                atk: [[0, 0], [25, -50], [52, -100], [74, -100], [100, 0]], sup: [[0, 0], [40, -25], [100, 0]] },
    bash:    { body: [[0, 'none'], [36, 'translateY(-3px) scale(1.03,1.06)'], [62, L(4, 'scale(1.05,.9)')], [100, 'none']], atk: [[0, 0], [35, -165], [55, -172], [63, 32], [100, 0]], sup: [[0, 0], [40, -150], [62, 10], [100, 0]] },
    shove:   { body: [[0, 'none'], [28, L(-1.5)], [52, L(4.5)], [100, 'none']],                                         atk: [[0, 0], [30, -35], [52, -88], [76, -88], [100, 0]], sup: [[0, 0], [30, -35], [52, -88], [76, -88], [100, 0]] },
    grapple: { body: [[0, 'none'], [30, L(2, 'scale(1.02,.95)')], [100, L(1, 'scale(1.02,.95)')]],                      atk: [[0, 0], [30, -105], [60, -118], [100, -70]], sup: [[0, 0], [30, -105], [60, -118], [100, -70]] },
    shoot:   { body: [[0, 'none'], [20, L(.5)], [45, L(-1.8)], [100, 'none']],                                         atk: [[0, 0], [20, -96], [80, -96], [100, 0]], sup: [[0, 0], [20, -88], [80, -88], [100, 0]] },
    throw:   { body: [[0, 'none'], [28, L(-1.5, 'rotate(-4deg)')], [58, L(2.5, 'rotate(5deg)')], [100, 'none']],       atk: [[0, 0], [28, -155], [46, -172], [60, 12], [100, 0]], sup: [[0, 0], [40, -40], [100, 0]] },
    wind:    { body: [[0, 'none'], [40, L(-2, 'rotate(-5deg) scale(1.03,1.05)')], [100, L(-2, 'rotate(-5deg) scale(1.03,1.05)')]], atk: [[0, 0], [40, -158], [100, -158]], sup: [[0, 0], [40, -30], [100, -30]], hold: 1 },
    feint:   { body: [[0, 'none'], [30, L(2.5)], [60, L(-1.5)], [100, 'none']],                                          atk: [[0, 0], [30, -62], [58, -8], [100, 0]], sup: [[0, 0], [30, -30], [100, 0]] },
    guard:   { body: [[0, 'none'], [30, 'scale(1.02,.94)'], [100, 'scale(1.02,.94)']],                                  atk: [[0, 0], [30, -48], [100, -48]], sup: [[0, 0], [30, -48], [100, -48]], hold: 1 },
    taunt:   { body: [[0, 'none'], [25, 'translateY(-2px)'], [50, 'translateY(0)'], [75, 'translateY(-2px)'], [100, 'none']], atk: [[0, 0], [20, -130], [40, -100], [60, -132], [80, -100], [100, 0]], sup: [[0, 0], [20, -130], [45, -100], [70, -132], [100, 0]] },
    dash:    { run: 1, body: [[0, 'rotate(7deg)'], [100, 'rotate(7deg)']] },
    side:    { run: 1, body: [[0, 'none'], [40, 'rotate(calc(var(--ly,0) * 7deg + var(--lx,0) * -7deg)) translateY(2px) scale(1,.95)'], [100, 'none']] },
    back:    { run: 1, body: [[0, 'rotate(-6deg)'], [100, 'rotate(-6deg)']] },
    stun:    { loop: 0.55, body: [[0, 'rotate(-7deg) translateX(-1.5px)'], [25, 'rotate(5deg) translateX(1px) translateY(1px)'], [50, 'rotate(-4deg) translateX(1.5px)'], [75, 'rotate(7deg) translateX(-1px) translateY(1px)'], [100, 'rotate(-7deg) translateX(-1.5px)']], atk: [[0, 12], [50, -8], [100, 12]], sup: [[0, -8], [50, 12], [100, -8]] },
    held:    { loop: 0.3, body: [[0, 'translateX(-1.4px) rotate(-3deg)'], [50, 'translateX(1.4px) rotate(3deg)'], [100, 'translateX(-1.4px) rotate(-3deg)']], atk: [[0, -30], [50, 10], [100, -30]], sup: [[0, 10], [50, -30], [100, 10]] }
  };
  function cmFrames() {
    var o = '';
    Object.keys(CMX).forEach(function (k) {
      var m = CMX[k];
      ['body', 'atk', 'sup'].forEach(function (kind) {
        var f = m[kind]; if (!f) return;
        ['a', 'b'].forEach(function (par) {
          o += '@keyframes ag-c-' + k + '-' + kind + '-' + par + '{' + f.map(function (q) {
            return q[0] + '%{transform:' + (kind === 'body' ? q[1] : 'rotate(calc(var(--sg,1) * ' + q[1] + 'deg))' + (q[2] ? ' scaleY(' + q[2] + ')' : '')) + '}';
          }).join('') + '}';
        });
      });
    });
    return o;
  }
  /* animation value for one combat pose; the a/b parity makes the browser restart the one-shot every tick */
  function cmAnim(cm, kind, par, ms) {
    var m = CMX[cm]; if (!m || !m[kind]) return 'none';
    var d = m.loop ? m.loop : Math.max(0.15, (ms || TICK_MS) / 1000 * 0.9);
    return 'ag-c-' + cm + '-' + kind + '-' + (m.loop ? 'a' : par ? 'b' : 'a') + ' ' + d.toFixed(2) + 's ' + (m.loop ? 'linear infinite' : 'ease-out 1 ' + (m.hold ? 'forwards' : 'both'));
  }
  var CSS = '@keyframes ag-idle{0%,100%{transform:scale(1,1)}50%{transform:scale(1.02,0.98)}}' +
    '@keyframes ag-walk{0%,100%{transform:none}}' +
    '@keyframes ag-run{0%,100%{transform:rotate(4deg)}}' +
    '@keyframes ag-clunge{0%,100%{transform:translate(0,0)}14%{transform:translate(calc(var(--cgx,0)*-3px),calc(var(--cgy,0)*-2px)) rotate(calc(var(--cgx,1)*-4deg))}36%{transform:translate(calc(var(--cgx,0)*11px),calc(var(--cgy,0)*8px)) rotate(calc(var(--cgx,1)*6deg)) scale(1.05,0.95)}54%{transform:translate(calc(var(--cgx,0)*4px),calc(var(--cgy,0)*3px))}}' +
    '@keyframes ag-cbrace{0%,100%{transform:translate(0,0)}30%{transform:translate(calc(var(--cgx,0)*2px),calc(var(--cgy,0)*1.5px)) scale(0.98,1.03)}58%{transform:translate(calc(var(--cgx,0)*3.5px),calc(var(--cgy,0)*2.5px)) scale(1.02,0.97)}}' +
    '@keyframes ag-eat{0%,100%{transform:translateY(0) scale(1)}30%,60%{transform:translateY(3px) scale(1.04,0.94)}45%{transform:translateY(0) scale(1)}}' +
    '@keyframes ag-drink{0%,100%{transform:translateY(0) rotate(0)}40%,70%{transform:translateY(4px) rotate(6deg)}}' +
    '@keyframes ag-sleep{0%,100%{transform:scale(1,1)}50%{transform:scale(1.03,0.96)}}' +
    '@keyframes ag-hurt{0%,100%{transform:translateX(0) rotate(0);filter:none}12%{transform:translateX(-4px) translateY(-1px) rotate(-7deg);filter:brightness(1.9) saturate(2.2) hue-rotate(-25deg)}35%{transform:translateX(3px) rotate(4deg);filter:brightness(1.3) saturate(1.5)}65%{transform:translateX(-1.5px) rotate(-2deg);filter:none}}' +
    '@keyframes ag-armHit{0%,100%{transform:rotate(0)}12%{transform:rotate(calc(var(--sg,1) * -60deg))}30%{transform:rotate(calc(var(--sg,1) * -150deg))}42%{transform:rotate(calc(var(--sg,1) * -165deg))}52%{transform:rotate(calc(var(--sg,1) * 20deg)) scaleY(.86)}70%{transform:rotate(calc(var(--sg,1) * 8deg))}}' +
    '@keyframes ag-armBrace{0%,100%{transform:rotate(0)}30%{transform:rotate(calc(var(--sg,1) * -22deg))}44%{transform:rotate(calc(var(--sg,1) * 10deg))}}' +
    '@keyframes ag-strike{0%,100%{transform:translateY(0) rotate(0) scale(1)}30%{transform:translateY(-3px) rotate(-5deg) scale(1.03,1.06)}44%{transform:translateY(3px) rotate(7deg) scale(1.05,0.94)}60%{transform:translateY(0) rotate(2deg) scale(1)}}' +
    '@keyframes ag-shiver{0%,100%{transform:translateX(0)}25%{transform:translateX(-1px)}75%{transform:translateX(1px)}}' +
    '@keyframes ag-leg{0%{transform:rotate(-17deg)}50%{transform:rotate(17deg)}60%{transform:rotate(11deg) translateY(-.5px) scaleY(.96)}76%{transform:rotate(-6deg) translateY(-1.1px) scaleY(.92)}92%{transform:rotate(-15deg) translateY(-.2px) scaleY(.99)}100%{transform:rotate(-17deg)}}' +
    '@keyframes ag-qbob{0%,100%{translate:0 .5px}50%{translate:0 calc(var(--qb,1.6px) * -1)}}' +
    '@keyframes ag-qpitch{0%,100%{rotate:calc(var(--fx,1) * -.5deg)}50%{rotate:calc(var(--fx,1) * .5deg)}}' +
    '@keyframes ag-fall{0%{transform:translate(0,0) rotate(0);animation-timing-function:ease-out}8%{transform:translate(calc(var(--d,1) * 1px),-3px) rotate(calc(var(--d,1) * -5deg));animation-timing-function:ease-in}30%{transform:translate(calc(var(--d,1) * -5px),-1px) rotate(calc(var(--d,1) * 35deg));animation-timing-function:ease-in}50%{transform:translate(calc(var(--d,1) * -17px),-1px) rotate(calc(var(--d,1) * 97deg));animation-timing-function:ease-out}62%{transform:translate(calc(var(--d,1) * -19px),-9px) rotate(calc(var(--d,1) * 76deg));animation-timing-function:ease-in}76%{transform:translate(calc(var(--d,1) * -17px),-3px) rotate(calc(var(--d,1) * 94deg));animation-timing-function:ease-out}88%{transform:translate(calc(var(--d,1) * -18px),-6px) rotate(calc(var(--d,1) * 86deg));animation-timing-function:ease-in}100%{transform:translate(calc(var(--d,1) * -18px),-4px) rotate(calc(var(--d,1) * 90deg))}}' +
    '@keyframes ag-fall2{0%{transform:perspective(420px) translate(0,0) rotate(0) rotateX(0);animation-timing-function:ease-out}10%{transform:perspective(420px) translate(calc(var(--d,1) * 1px),-4px) rotate(calc(var(--d,1) * -3deg)) rotateX(calc(var(--t,1) * -4deg));animation-timing-function:ease-in}34%{transform:perspective(420px) translate(calc(var(--d,1) * 2px),0) rotate(calc(var(--d,1) * 5deg)) rotateX(calc(var(--t,1) * 12deg));animation-timing-function:ease-in}52%{transform:perspective(420px) translate(0,3px) rotate(calc(var(--d,1) * 8deg)) rotateX(calc(var(--t,1) * 53deg));animation-timing-function:ease-out}64%{transform:perspective(420px) translate(0,-6px) rotate(calc(var(--d,1) * 4deg)) rotateX(calc(var(--t,1) * 39deg));animation-timing-function:ease-in}78%{transform:perspective(420px) translate(0,2px) rotate(calc(var(--d,1) * 7deg)) rotateX(calc(var(--t,1) * 52deg));animation-timing-function:ease-out}90%{transform:perspective(420px) translate(0,-1px) rotate(calc(var(--d,1) * 5deg)) rotateX(calc(var(--t,1) * 50deg))}100%{transform:perspective(420px) translate(0,1px) rotate(calc(var(--d,1) * 6deg)) rotateX(calc(var(--t,1) * 50deg))}}' +
    '@keyframes ag-head{0%{transform:rotate(0)}8%{transform:rotate(calc(var(--d,1) * 9deg))}30%{transform:rotate(calc(var(--d,1) * -16deg))}50%{transform:rotate(calc(var(--d,1) * 20deg))}62%{transform:rotate(calc(var(--d,1) * -12deg))}76%{transform:rotate(calc(var(--d,1) * 11deg))}88%{transform:rotate(calc(var(--d,1) * -4deg))}100%{transform:rotate(calc(var(--d,1) * 7deg))}}' +
    '@keyframes ag-armL{0%{transform:rotate(0)}8%{transform:rotate(14deg)}30%{transform:rotate(-30deg)}50%{transform:rotate(38deg)}62%{transform:rotate(-26deg)}76%{transform:rotate(18deg)}88%{transform:rotate(-8deg)}100%{transform:rotate(10deg)}}' +
    '@keyframes ag-armR{0%{transform:rotate(0)}8%{transform:rotate(-14deg)}30%{transform:rotate(30deg)}50%{transform:rotate(-38deg)}62%{transform:rotate(26deg)}76%{transform:rotate(-18deg)}88%{transform:rotate(8deg)}100%{transform:rotate(-12deg)}}' +
    '@keyframes ag-sh-walk{0%,100%{transform:scale(1)}}' +
    '@keyframes ag-sh-run{0%,100%{transform:scale(1)}}' +
    '@keyframes ag-sh-hop{0%,100%{transform:scale(1.04,1)}35%{transform:scale(0.86);opacity:0.8}70%{transform:scale(1)}}' +
    '@keyframes ag-sh-cheer{0%,100%{transform:scale(1.05)}30%,60%{transform:scale(0.7);opacity:0.55}80%{transform:scale(1)}}' +
    '@keyframes ag-sh-pounce{0%,100%{transform:scale(1)}30%{transform:scale(1.08,1)}55%{transform:scale(0.72);opacity:0.6}}' +
    '@keyframes ag-sh-strike{0%,100%{transform:scale(1)}30%{transform:scale(0.88);opacity:0.8}50%{transform:scale(1.1,1)}}' +
    '@keyframes ag-hop{0%,100%{transform:translateY(0) scale(1.03,0.97)}35%{transform:translateY(-3px) scale(0.98,1.03)}70%{transform:translateY(0) scale(1)}}' +
    '@keyframes ag-lumber{0%,100%{transform:rotate(-1.5deg) translateY(0)}50%{transform:rotate(1.5deg) translateY(-1px)}}' +
    '@keyframes ag-prowl{0%,100%{transform:translateX(-1px) scaleY(1)}50%{transform:translateX(1px) scaleY(0.96)}}' +
    '@keyframes ag-pounce{0%,100%{translate:0 0;rotate:0deg;scale:1}20%{translate:calc(var(--fx,1) * -2px) 1px;rotate:calc(var(--fx,1) * -6deg);scale:1.06 .92}38%{translate:calc(var(--fx,1) * 8px) -3px;rotate:calc(var(--fx,1) * 4deg);scale:1.12 .94}50%{translate:calc(var(--fx,1) * 7px) -1px;rotate:calc(var(--fx,1) * 10deg);scale:1.08 .9}66%{translate:calc(var(--fx,1) * 8px) 0;rotate:calc(var(--fx,1) * -3deg);scale:1.1 .95}84%{translate:calc(var(--fx,1) * 2px) 0;rotate:0deg;scale:1}}' +
    '@keyframes ag-slither{0%,100%{transform:translateX(-2px) skewX(-5deg)}50%{transform:translateX(2px) skewX(5deg)}}' +
    '@keyframes ag-work{0%,100%{transform:translateY(0) rotate(0)}25%{transform:translateY(-4px) rotate(-5deg)}45%{transform:translateY(3px) rotate(6deg) scale(1.04,0.96)}60%{transform:translateY(0) rotate(0)}}' +
    '@keyframes ag-forage{0%,100%{transform:translateY(0) rotate(0)}35%,65%{transform:translateY(5px) rotate(10deg) scale(1.03,0.95)}}' +
    '@keyframes ag-carry{0%,100%{transform:none}}' +
    '@keyframes ag-wave{0%,100%{transform:rotate(0) translateY(0)}20%{transform:rotate(-6deg) translateY(-1px)}50%{transform:rotate(6deg) translateY(-1px)}80%{transform:rotate(-4deg)}}' +
    '@keyframes ag-cheer{0%,100%{transform:translateY(0) scale(1.06,0.94)}30%{transform:translateY(-9px) rotate(-6deg) scale(0.96,1.06)}60%{transform:translateY(-9px) rotate(6deg) scale(0.96,1.06)}80%{transform:translateY(0)}}' +
    '@keyframes ag-stumble{0%,100%{transform:translate(0,0) rotate(0)}15%{transform:translate(-3px,1px) rotate(-8deg)}35%{transform:translate(2px,-2px) rotate(6deg)}55%{transform:translate(-1px,2px) rotate(-3deg)}75%{transform:translate(3px,0) rotate(9deg)}}' +
    '@keyframes ag-fly{0%,100%{transform:translate(0,0)}25%{transform:translate(2.4px,-1.6px)}50%{transform:translate(0,-3px)}75%{transform:translate(-2.4px,-1.6px)}}' +
    '@keyframes ag-zzz{0%{opacity:0;transform:translate(0,4px) scale(0.7)}30%{opacity:1}100%{opacity:0;transform:translate(6px,-14px) scale(1.1)}}' +
    '@keyframes ag-drip{0%{opacity:0;transform:translateY(-2px)}30%{opacity:1}100%{opacity:0;transform:translateY(7px)}}' +
    '@keyframes ag-stepF{0%{transform:translateY(3.4px) scaleY(1)}50%{transform:translateY(-3.4px) scaleY(1)}75%{transform:translateY(-6px) scaleY(.76)}100%{transform:translateY(3.4px) scaleY(1)}}' +
    '@keyframes ag-stepS{0%{transform:rotate(var(--sw,18deg)) translateY(0)}50%{transform:rotate(calc(var(--sw,18deg) * -1)) translateY(0)}60%{transform:rotate(calc(var(--sw,18deg) * -.8)) translateY(-1.6px)}74%{transform:rotate(calc(var(--sw,18deg) * .1)) translateY(-2.4px)}90%{transform:rotate(calc(var(--sw,18deg) * .85)) translateY(-.8px)}100%{transform:rotate(var(--sw,18deg)) translateY(0)}}' +
    '@keyframes ag-armS{0%,100%{transform:rotate(var(--aw,24deg))}50%{transform:rotate(calc(var(--aw,24deg) * -.85))}}' +
    '@keyframes ag-bob{0%,100%{translate:0 .5px}50%{translate:0 calc(var(--bu,1.3px) * -1)}}' +
    '@keyframes ag-sway{0%,100%{rotate:calc(var(--sy,1.5deg) * -1)}50%{rotate:var(--sy,1.5deg)}}' +
'@keyframes ag-armF{0%,100%{transform:translateY(-3.2px) scaleY(.92) rotate(calc(7deg * var(--sg,1)))}50%{transform:translateY(3.2px) scaleY(1.08) rotate(calc(-7deg * var(--sg,1)))}}' +
    '@media (prefers-reduced-motion: reduce){*{animation:none !important;transition-duration:0.01ms !important}}';
  try { if (typeof document !== 'undefined' && !document.getElementById('agon-motion')) { var st = document.createElement('style'); st.id = 'agon-motion'; st.textContent = CSS + cmFrames(); document.head.appendChild(st); } } catch (e) {}

  /* ---------- animals ---------- */
  var ANIMALS = {
    small: { name: 'Small prey', body: '#cfc4b0', belly: '#e8dcc4', mark: '#a39a88', ear: '#e0a090' },
    large: { name: 'Large prey', body: '#7a6a52', belly: '#9a8a6a', mark: '#a39f93', plate: '#8a8a82' },
    pred:  { name: 'Predator',   body: '#8a3f2e', belly: '#b3684e', mark: '#4a2018', eye: '#d4c63c' },
    deer:   { name: 'Deer',          body: '#a8794e', belly: '#e4d2b0', mark: '#6b4a2c', ear: '#e0a090', horn: '#cdbd98' },
    goat:   { name: 'Mountain goat', body: '#e8e2d4', belly: '#fff8ea', mark: '#8a8478', horn: '#6b5a3c' },
    grouse: { name: 'Grouse',        body: '#8a6a48', belly: '#c9a878', mark: '#3a2a1c', comb: '#c2503c' },
    fox:    { name: 'Fox',           body: '#c8683a', belly: '#f1dcc0', mark: '#3a2018', eye: '#d4c63c' },
    bear:   { name: 'Bear',          body: '#5a4030', belly: '#7a5a40', mark: '#33241a', eye: '#d4c63c', muzzle: '#b08a62' },
    snake:  { name: 'Viper',         body: '#6a7a3a', belly: '#c8c27a', mark: '#2e3a1c', eye: '#d4c63c' }
  };
  /* climate variants: kind + '.' + variant. Same silhouette, different coat. Snow = tundra/alpine, sand = desert/badlands, ash = volcanic, marsh = wetland */
  var ANIMAL_VARS = {
    'bear.snow':   { name: 'Polar bear',    body: '#ece8dc', belly: '#fbf8ee', mark: '#b9b3a4', eye: '#2a2a2a', muzzle: '#d8d0bc' },
    'pred.snow':   { name: 'Arctic wolf',   body: '#e4e2da', belly: '#f6f4ec', mark: '#a8a69c', eye: '#8ec0d8' },
    'fox.snow':    { name: 'Arctic fox',    body: '#eef0ee', belly: '#ffffff', mark: '#9aa0a4', eye: '#2a2a2a' },
    'small.snow':  { name: 'Snow hare',     body: '#f2f2ee', belly: '#ffffff', mark: '#8e9294', ear: '#d8b0a8' },
    'deer.snow':   { name: 'Caribou',       body: '#9a8a78', belly: '#ece4d6', mark: '#5e5246', ear: '#cba89c', horn: '#d8ccb0' },
    'grouse.snow': { name: 'Ptarmigan',     body: '#eceae2', belly: '#ffffff', mark: '#9a9890', comb: '#c2503c' },
    'large.snow':  { name: 'Tundra boar',   body: '#8a7c68', belly: '#b0a48e', mark: '#b8b4a8', plate: '#c8c8c0' },
    'goat.snow':   { name: 'Snow goat',     body: '#f4f2ea', belly: '#ffffff', mark: '#a09a8e', horn: '#4a3c2a' },
    'pred.sand':   { name: 'Jackal',        body: '#c9a266', belly: '#e6d2a6', mark: '#6a4c26', eye: '#d4a83c' },
    'fox.sand':    { name: 'Fennec fox',    body: '#e4c692', belly: '#f8ecd2', mark: '#8a6a3a', eye: '#2a2a2a' },
    'small.sand':  { name: 'Jackrabbit',    body: '#c8ac84', belly: '#ecdcb8', mark: '#8a7050', ear: '#d8a088' },
    'deer.sand':   { name: 'Gazelle',       body: '#c6925e', belly: '#f0e2c4', mark: '#6a4424', ear: '#e0a090', horn: '#3a3026' },
    'grouse.sand': { name: 'Sandgrouse',    body: '#b89a68', belly: '#e0c890', mark: '#5a4426', comb: '#c28a3c' },
    'goat.sand':   { name: 'Ibex',          body: '#b89c78', belly: '#e8d8b8', mark: '#5a4a36', horn: '#4a3a24' },
    'large.sand':  { name: 'Warthog',       body: '#8a6e50', belly: '#b09470', mark: '#6a4a30', plate: '#9a8a70' },
    'bear.sand':   { name: 'Sun bear',      body: '#4a3626', belly: '#6a4c34', mark: '#2a1c12', eye: '#d4c63c', muzzle: '#c8a070' },
    'snake.sand':  { name: 'Sand viper',    body: '#cfb072', belly: '#f0e2b0', mark: '#8a5c2a', eye: '#d4a83c' },
    'pred.ash':    { name: 'Ash wolf',      body: '#6a6460', belly: '#8e8680', mark: '#2a2624', eye: '#e0802c' },
    'fox.ash':     { name: 'Cinder fox',    body: '#b0502a', belly: '#d8a070', mark: '#241410', eye: '#f0a030' },
    'small.ash':   { name: 'Ash hare',      body: '#9a948a', belly: '#c4bcae', mark: '#5a554e', ear: '#c08880' },
    'goat.ash':    { name: 'Basalt goat',   body: '#7a746c', belly: '#a8a094', mark: '#34302c', horn: '#2e2a26' },
    'snake.ash':   { name: 'Ember viper',   body: '#8a3a22', belly: '#d89a4a', mark: '#2a1208', eye: '#f0c830' },
    'snake.marsh': { name: 'Water moccasin',body: '#3e4a2a', belly: '#a8a868', mark: '#1a2410', eye: '#d4c63c' },
    'grouse.marsh':{ name: 'Marsh hen',     body: '#6a6a4a', belly: '#a8a47a', mark: '#2a2a1a', comb: '#c2503c' },
    'deer.marsh':  { name: 'Marsh deer',    body: '#8a6a44', belly: '#d4c4a0', mark: '#4a3420', ear: '#d09888', horn: '#b8a886' },
    'small.marsh': { name: 'Marsh rabbit',  body: '#8a7a62', belly: '#c8baa0', mark: '#5a4e3c', ear: '#c88a80' },
    'pred.marsh':  { name: 'Swamp wolf',    body: '#5e5444', belly: '#8a7e66', mark: '#2a241a', eye: '#c8c83c' },
    'large.marsh': { name: 'Mire boar',     body: '#4e4636', belly: '#766a52', mark: '#7a7a6a', plate: '#6a6a5a' },
    'small.vole': { x: 'vole', name: 'Vole', body: '#8a6e50', belly: '#c4ae8a', mark: '#5a4630', ear: '#c08878' },
    'small.lemming': { x: 'lemming', name: 'Lemming', body: '#b88a4e', belly: '#e4cf9c', mark: '#3a2a1a', ear: '#c89888' },
    'small.pika': { x: 'pika', name: 'Pika', body: '#8e8472', belly: '#bdb39c', mark: '#5a5246', ear: '#a88a7c' },
    'small.jerboa': { x: 'jerboa', name: 'Jerboa', body: '#d4b484', belly: '#f4e6c4', mark: '#8a6a40', ear: '#e0a890' },
    'grouse.quail': { x: 'quail', name: 'Quail', body: '#8c7a5a', belly: '#d8c49a', mark: '#3a2c1c', comb: '#2a2218' },
    'grouse.turkey': { x: 'turkey', name: 'Wild turkey', body: '#4a3626', belly: '#7a5a3a', mark: '#1e140c', comb: '#c43a32' },
    'grouse.duck': { x: 'duck', name: 'Marsh duck', body: '#5a7a6a', belly: '#e4e0cc', mark: '#2a4034', comb: '#d8a32a' },
    'deer.elk': { x: 'elk', name: 'Elk', body: '#8a6240', belly: '#d8c6a0', mark: '#3e2a1a', ear: '#c89484', horn: '#d8c8a0' },
    'deer.moose': { x: 'moose', name: 'Moose', body: '#3e2f24', belly: '#6a5646', mark: '#1a120c', ear: '#6a4a3c', horn: '#b8a67c' },
    'large.bison': { x: 'bison', name: 'Bison', body: '#4a3524', belly: '#6a4e36', mark: '#1e140c', plate: '#6e5a46' },
    'large.yak': { x: 'yak', name: 'Yak', body: '#2e2824', belly: '#4e443c', mark: '#14100c', plate: '#5a5048' },
    'goat.bighorn': { x: 'bighorn', name: 'Bighorn sheep', body: '#a89070', belly: '#e8dcc0', mark: '#6a5a44', horn: '#c8b890' },
    'large.capybara': { x: 'capybara', name: 'Capybara', body: '#8a6a4a', belly: '#b8946a', mark: '#5a4430', plate: '#9a7a58' },
    'large.peccary': { x: 'peccary', name: 'Peccary', body: '#5e5a52', belly: '#8a857a', mark: '#2a2824', plate: '#a8a094' },
    'fox.stoat': { x: 'stoat', name: 'Stoat', body: '#a67a4a', belly: '#f4eddc', mark: '#2a1c10', eye: '#2a2a2a' },
    'fox.badger': { x: 'badger', name: 'Badger', body: '#6e6e70', belly: '#c8c6c0', mark: '#1e1e20', eye: '#2a2a2a' },
    'grouse.hawk': { x: 'hawk', name: 'Hawk', body: '#7a5a3a', belly: '#e4d4b4', mark: '#3a2814', comb: '#5a4024', eye: '#f0d040', raptor: 'hawk' },
    'grouse.owl': { x: 'owl', name: 'Owl', body: '#a88c64', belly: '#ecdcc0', mark: '#5a4630', comb: '#a88c64', eye: '#f0c830', raptor: 'owl' },
    'grouse.vulture': { x: 'vulture', name: 'Vulture', body: '#2e2a28', belly: '#cfc8b8', mark: '#14100e', comb: '#c8806a', raptor: 'vulture' },
    'fox.lynx': { x: 'lynx', name: 'Lynx', body: '#a89478', belly: '#e8dcc4', mark: '#3a2c20', eye: '#d4c63c', tuft: true },
    'pred.wolf': { x: 'wolf', name: 'Gray wolf', body: '#8a867c', belly: '#dcd6c6', mark: '#3e3c38', eye: '#d4a83c' },
    'grouse.heron': { x: 'heron', name: 'Great blue heron', body: '#7a8ea4', belly: '#dfe4ea', mark: '#3a4658', eye: '#e8c840', bill: '#d6a83c', leg: '#7a6a48' },
    'fox.otter': { x: 'otter', name: 'River otter', body: '#6a4a32', belly: '#c4a888', mark: '#3e2a1c', eye: '#2a2a2a' },
    'bear.polarbear': { x: 'polarbear', name: 'Polar bear', body: '#f0ecdf', belly: '#fffdf4', mark: '#b9b3a4', eye: '#2a2a2a', muzzle: '#ddd4bc' },
    'large.muskox': { x: 'muskox', name: 'Muskox', body: '#3a2c22', belly: '#5a4636', mark: '#241a14', eye: '#2a2a2a', muzzle: '#8a7260' },
    'deer.camel': { x: 'camel', name: 'Dromedary camel', body: '#c4985a', belly: '#e0c08a', mark: '#8a6a38', eye: '#2a2a2a', muzzle: '#d8b078' },
    'fox.fennec': { x: 'fennec', name: 'Fennec fox', body: '#e6c894', belly: '#fbf0d8', mark: '#8a6a3a', eye: '#2a2a2a', ear: '#e8b4a0' },
    'pred.coyote': { x: 'coyote', name: 'Coyote', body: '#a8906a', belly: '#e0d0aa', mark: '#5a4630', eye: '#d4a83c' },
    'pred.hyena': { x: 'hyena', name: 'Hyena', body: '#a89264', belly: '#d0c094', mark: '#3e3020', eye: '#c8a02c', spot: '#5a4630' },
    'bear.wolverine': { x: 'wolverine', name: 'Wolverine', body: '#4a3828', belly: '#8a6a46', mark: '#241810', eye: '#d4c63c', muzzle: '#a88a60' },
    'pred.cougar': { x: 'cougar', name: 'Cougar', body: '#c4a070', belly: '#e8d8b8', mark: '#6a4c2a', eye: '#d4c63c', cat: true },
    'pred.snowleopard': { x: 'snowleopard', name: 'Snow leopard', body: '#cfd0c8', belly: '#f0f0ea', mark: '#4a4c52', eye: '#9ec4cc', cat: true, spot: '#4a4c54' },
    'deer.caribou': { x: 'caribou', name: 'Caribou', body: '#7a6a58', belly: '#e8e2d6', mark: '#4a3e32', ear: '#c0a898', horn: '#cdbf9e' },
    'small.hare': { x: 'hare', name: 'Arctic hare', body: '#dcd8d0', belly: '#f4f2ee', mark: '#3a3a38', ear: '#c8bcb4' },
    'grouse.ptarmigan': { x: 'ptarmigan', name: 'Ptarmigan', body: '#e4e0d8', belly: '#f6f4f0', mark: '#8a6a4a', comb: '#c8363a' },
    'deer.impala': { x: 'impala', name: 'Impala', body: '#b87a44', belly: '#f0e0c0', mark: '#4a2e18', ear: '#d89888', horn: '#2e241c' },
    'large.wildebeest': { x: 'wildebeest', wil: 1, name: 'Wildebeest', body: '#5a5048', belly: '#7a7064', mark: '#1e1a16', plate: '#8a8070' },
    'large.elephant': { x: 'elephant', ele: 1, name: 'Elephant', body: '#8e8a86', belly: '#a8a49e', mark: '#4a4642', plate: '#b4b0a8' },
    'grouse.ostrich': { x: 'ostrich', name: 'Ostrich', body: '#2a2622', belly: '#e8e2d0', mark: '#14100e', comb: '#d8a898' },
    'pred.lion': { x: 'lion', name: 'Lion', body: '#c9a063', belly: '#ead8b0', mark: '#6a4220', eye: '#d4a83c', cat: true },
    'large.tapir': { x: 'tapir', name: 'Tapir', body: '#4a4438', belly: '#6a6454', mark: '#1c1812', plate: '#6e6858' },
    'pred.jaguar': { x: 'jaguar', name: 'Jaguar', body: '#c9962c', belly: '#f0dcaa', mark: '#3a2410', eye: '#d4c63c', cat: true, spot: '#2a1a0c' },
    'snake.anaconda': { x: 'anaconda', name: 'Anaconda', body: '#4a5a30', belly: '#b8b878', mark: '#1c2410', eye: '#d4c63c' },
    'grouse.macaw': { x: 'macaw', name: 'Macaw', body: '#c02a2a', belly: '#e8c43a', mark: '#1a3a8a', comb: '#2a5ac8' },
    'small.paca': { x: 'paca', name: 'Paca', body: '#7a5a3e', belly: '#d8c4a0', mark: '#e8dcc0', ear: '#b88878' },
    'large.boar': { x: 'boar', name: 'Wild boar', body: '#4a3e32', belly: '#6a5a48', mark: '#1e1810', plate: '#7a6a58' },
    'small.beaver': { x: 'beaver', name: 'Beaver', body: '#6a4a30', belly: '#9a7a54', mark: '#3a2616', ear: '#8a6a50' },
    'large.tortoise': { x: 'tortoise', name: 'Giant tortoise', body: '#7a6a44', belly: '#a8a47a', mark: '#3e3624', plate: '#8c7c52' },
    'small.hyrax': { x: 'hyrax', name: 'Rock hyrax', body: '#8a7a62', belly: '#c8b898', mark: '#4a3e30', ear: '#a88878' },
    'snake.komodo': { x: 'komodo', name: 'Komodo dragon', body: '#6a5a46', belly: '#a89878', mark: '#3a3024', eye: '#d4a83c' },
    'pred.cheetah': { x: 'cheetah', name: 'Cheetah', body: '#d8b070', belly: '#f2e4c0', mark: '#3a2a18', eye: '#d4a83c', cat: true, spot: '#3a2a18' },
    'grouse.eagle': { x: 'eagle', name: 'Bald eagle', body: '#4a3a2a', belly: '#f4f0e6', mark: '#2a1e14', comb: '#d8a63a', eye: '#f0d040' },
    'fox.ocelot': { x: 'ocelot', name: 'Ocelot', body: '#c8a060', belly: '#f0e4c8', mark: '#3a2814', eye: '#d4c63c', spot: '#3a2814' },
    'snake.crocodile': { name: 'Crocodile', body: '#4a5a3a', belly: '#b4b07a', mark: '#1c2812', eye: '#d4c63c', croc: true }
  };
  var VAR_OF = { tundra: 'snow', alpine: 'snow', highlands: 'snow', desert: 'sand', badlands: 'sand', volcanic: 'ash', wetland: 'marsh' };
  function animalPal(kind, v) { return ANIMAL_VARS[kind + '.' + v] || ANIMALS[kind] || ANIMALS.small; }


  /* ---------- body builds and sizes ----------
     w widens the torso, h deepens it, s scales the whole figure about its feet. Children keep the full-size head. */
  var BUILDS = {
    lean:    { label: 'Lean',    w: 0.86, h: 0.96, s: 1.04 , hd: 0.98, lg: 0.6 },
    average: { label: 'Average', w: 1,    h: 1,    s: 1 , hd: 1, lg: 0 },
    stocky:  { label: 'Stocky',  w: 1.16, h: 1.04, s: 0.96 , hd: 1.02, lg: -0.5 },
    lanky:   { label: 'Lanky',   w: 0.8,  h: 1.2,  s: 1.1,  sh: 0.94, hp: 0.9 , hd: 0.93, lg: 1.6 },
    athletic:{ label: 'Athletic',w: 1.06, h: 1.08, s: 1.03, sh: 1.16, hp: 0.9 , hd: 0.97, lg: 0.4 },
    pear:    { label: 'Pear',    w: 1.12, h: 0.98, s: 0.97, sh: 0.84, hp: 1.18 , hd: 1, lg: -0.2 },
    barrel:  { label: 'Barrel',  w: 1.24, h: 1.14, s: 0.99, sh: 1.06, hp: 1.04 , hd: 1, lg: -0.4 },
    stout:   { label: 'Stout',   w: 1.2,  h: 0.84, s: 0.9,  sh: 1.04, hp: 1.04 , hd: 1.04, lg: -0.9 },
    heavy:   { label: 'Heavy',   w: 1.3,  h: 1.12, s: 1.05 , hd: 1.03, lg: -0.6 },
    child:   { label: 'Child',   w: 0.8,  h: 0.82, s: 0.78 , hd: 1.2, lg: -1.2 }
  };
  var BUILD_ORDER = ['lean', 'lanky', 'average', 'athletic', 'stocky', 'barrel', 'pear', 'stout', 'heavy', 'child'];
  /* body frame: width and height follow the genes for muscle, stamina, frame and fat store, so a trained agonist is broad through the chest and a soft one is round */
  function physique(ph) {
    ph = ph || {};
    var mu = lg(ph.muscle), st = lg(ph.stamina), fr = lg(ph.frame), ft = lg(ph.fat), fit = clamp(0.6 * mu + 0.4 * st, -1, 1);
    // phy = "sex,defBias,curveBias": two inherited appearance traits. Definition is how cut the muscle reads (muscle gene, low fat, the bias); curve is the hourglass shape (sex, fat, the bias).
    var pp = String(ph.phy || '').split(','), sx = pp[0] === 'F' ? 'F' : 'M', b3 = pp[1] === undefined || pp[1] === '' ? 0.5 : +pp[1], b4 = pp[2] === undefined || pp[2] === '' ? 0.5 : +pp[2];
    var def = clamp(0.55 * mu + (b3 - 0.5) * 1.2 - 0.4 * Math.max(0, ft) + 0.3, 0, 1.2);
    var curve = clamp((sx === 'F' ? 0.5 : -0.1) + (b4 - 0.5) * 1.3 + (sx === 'F' ? 0.25 : 0.08) * ft, -0.6, 1);
    return { fit: fit, fat: ft, def: def, curve: curve, sex: sx, w: clamp(1 + 0.12 * fit + 0.1 * fr + 0.16 * ft, 0.6, 1.38), h: clamp(1 + 0.05 * fit + 0.22 * fr - 0.04 * ft, 0.8, 1.32) };
  }
  function pawnDims(strength, build, scale, ph, ag) {
    ag = ag || {};
    var b = BUILDS[build] || BUILDS.average, px = physique(ph);
    return { def: px.def, curve: px.curve, sex: px.sex, hd: (b.hd || 1) * (ag.grow >= 0 && build === 'child' ? 1 - 0.14 * Math.min(1, ag.grow) : 1) * (1 - 0.06 * Math.min(1, Math.max(0, ag.old || 0))), lg: b.lg || 0, sh: b.sh || 1, hp: b.hp || 1, bw: Math.min(15, (7.6 + 1.6 * lg(strength)) * b.w * px.w), bh: 8.4 * b.h * px.h, scale: (scale || 1) * b.s, build: BUILDS[build] ? build : 'average', fit: px.fit, fat: px.fat };
  }

  /* torso: shoulders, waist and hips follow fitness and fat; muscle and rib lines are cel shading over the clothes, so a trained body reads broad and cut and a thin one reads narrow and ribby */
  function bodyKit(bw, bh, fit, fat, view, clothC, bt) {
    bt = bt || {};
    var cx = 24, yt = 33 - bh, yb = 33 + bh, f = function (v) { return Math.round(v * 100) / 100; };
    var def0 = clamp(bt.def || 0, 0, 1.2), cv = clamp(bt.curve || 0, -0.6, 1), up = clamp(Math.max(clamp(fit, 0, 1), def0 * 0.95), 0, 1), thin = clamp(-fit, 0, 1), fa = clamp(fat, 0, 1), lean = clamp(-fat, 0, 1);
    var sh = bw * (1 + 0.07 * fit - 0.04 * fat + 0.02 * def0) * (bt.sh || 1) * (1 - 0.04 * cv), wa = Math.max(bw * 0.66, bw * (1 - 0.1 * fit + 0.12 * fat) * (bt.hp || 1) * (1 + 0.1 * cv));
    wa = Math.min(wa, Math.max(sh * 1.15, bw * 0.9));
    var wn = Math.min(sh, wa) * (1 - 0.15 * cv + 0.05 * fat - 0.02 * def0), tz = 0.92 + 0.05 * def0;
    var d = 'M' + f(cx - sh * 0.5) + ' ' + f(yt) + ' C' + f(cx - sh * tz) + ' ' + f(yt) + ' ' + f(cx - sh) + ' ' + f(yt + bh * 0.3) + ' ' + f(cx - sh) + ' ' + f(yt + bh * 0.58)
      + ' C' + f(cx - sh) + ' ' + f(33 + bh * 0.02) + ' ' + f(cx - wn) + ' ' + f(33 + bh * 0.0) + ' ' + f(cx - wn) + ' ' + f(33 + bh * 0.22)
      + ' C' + f(cx - wn) + ' ' + f(33 + bh * 0.42) + ' ' + f(cx - wa) + ' ' + f(33 + bh * 0.4) + ' ' + f(cx - wa) + ' ' + f(33 + bh * 0.66)
      + ' C' + f(cx - wa) + ' ' + f(yb - 0.1) + ' ' + f(cx - wa * 0.66) + ' ' + f(yb) + ' ' + f(cx) + ' ' + f(yb)
      + ' C' + f(cx + wa * 0.66) + ' ' + f(yb) + ' ' + f(cx + wa) + ' ' + f(yb - 0.1) + ' ' + f(cx + wa) + ' ' + f(33 + bh * 0.66)
      + ' C' + f(cx + wa) + ' ' + f(33 + bh * 0.4) + ' ' + f(cx + wn) + ' ' + f(33 + bh * 0.42) + ' ' + f(cx + wn) + ' ' + f(33 + bh * 0.22)
      + ' C' + f(cx + wn) + ' ' + f(33 + bh * 0.0) + ' ' + f(cx + sh) + ' ' + f(33 + bh * 0.02) + ' ' + f(cx + sh) + ' ' + f(yt + bh * 0.58)
      + ' C' + f(cx + sh) + ' ' + f(yt + bh * 0.3) + ' ' + f(cx + sh * tz) + ' ' + f(yt) + ' ' + f(cx + sh * 0.5) + ' ' + f(yt) + 'Z';
    var dk = shade(clothC, 0.55), hi = shade(clothC, 1.3), L = [];
    function lk(p, col, w, op) { op *= 0.5; if (op > 0.05) L.push({ d: p, fill: 'none', stroke: mix(clothC, col, Math.min(0.5, op)), sw: w * 0.85, op: 1 }); }
    function ln() { /* muscle and rib cel lines are off: the build shows in the silhouette, not in scribbles */ }
    var s, y;
    if (view === 'S' || view === 'SE') {
      var k = view === 'SE' ? 0.9 : 1;
      for (s = -1; s <= 1; s += 2) {
        ln('M' + f(cx + s * sh * 0.05 * k) + ' ' + f(yt + bh * 0.66) + ' Q' + f(cx + s * sh * 0.42 * k) + ' ' + f(yt + bh * 0.92) + ' ' + f(cx + s * sh * 0.8 * k) + ' ' + f(yt + bh * 0.56), dk, 1.3, 0.18 + 0.8 * up);
        ln('M' + f(cx + s * (sh - 0.4)) + ' ' + f(yt + bh * 0.3) + ' Q' + f(cx + s * (sh - 2)) + ' ' + f(yt + bh * 0.72) + ' ' + f(cx + s * (sh - 0.7)) + ' ' + f(yt + bh * 1.0), dk, 1, 0.1 + 0.7 * up);
        if (view === 'S' || view === 'SE') {
          if (def0 > 0.25) { ln('M' + f(cx + s * (sh - 1.1)) + ' ' + f(yt + bh * 0.1) + ' Q' + f(cx + s * (sh + 0.3)) + ' ' + f(yt + bh * 0.42) + ' ' + f(cx + s * (sh - 1.5)) + ' ' + f(yt + bh * 0.72), dk, 1.3, 0.75 * clamp(def0, 0, 1)); ln('M' + f(cx + s * wa * 0.58) + ' ' + f(33 + bh * 0.02) + ' L' + f(cx + s * wn * 0.78) + ' ' + f(33 + bh * 0.5), dk, 1, 0.55 * clamp(def0, 0, 1)); ln('M' + f(cx + s * sh * 0.78) + ' ' + f(yt + bh * 0.8) + ' Q' + f(cx + s * wn * 0.9) + ' ' + f(33 + bh * 0.05) + ' ' + f(cx + s * wn * 0.7) + ' ' + f(33 + bh * 0.3), dk, 1, 0.5 * clamp(def0, 0, 1)); }
          if (cv > 0.3) { ln('M' + f(cx + s * sh * 0.14) + ' ' + f(yt + bh * 0.98) + ' Q' + f(cx + s * sh * 0.46) + ' ' + f(yt + bh * 1.22) + ' ' + f(cx + s * sh * 0.76) + ' ' + f(yt + bh * 0.98), dk, 1.2, 0.55 * cv); ln('M' + f(cx + s * wa * 0.9) + ' ' + f(33 + bh * 0.46) + ' Q' + f(cx + s * wa * 1.02) + ' ' + f(33 + bh * 0.7) + ' ' + f(cx + s * wa * 0.82) + ' ' + f(33 + bh * 0.96), hi, 1.1, 0.45 * cv); }
        }
        if (up > 0.6 && view === 'S') for (y = 0; y < 3; y++) ln('M' + f(cx - wa * 0.1 * s) + ' ' + f(33 + bh * (0.1 + 0.26 * y)) + ' Q' + f(cx + s * wa * 0.3) + ' ' + f(33 + bh * (0.16 + 0.26 * y)) + ' ' + f(cx + s * wa * 0.42) + ' ' + f(33 + bh * (0.1 + 0.26 * y)), dk, 1.1, 0.5 * clamp(up * 1.6 - 0.35, 0, 1));
        if (thin > 0.45) for (y = 0; y < 3; y++) ln('M' + f(cx + s * wa * 0.18) + ' ' + f(yt + bh * (0.95 + 0.26 * y)) + ' Q' + f(cx + s * wa * 0.5) + ' ' + f(yt + bh * (1.04 + 0.26 * y)) + ' ' + f(cx + s * wa * 0.82) + ' ' + f(yt + bh * (0.98 + 0.26 * y)), dk, 1.1, 0.55 * thin);
      }
      ln('M' + f(cx) + ' ' + f(yt + bh * 0.95) + ' L' + f(cx) + ' ' + f(33 + bh * 0.75), dk, 1.1, 0.4 * clamp(up * 1.6 - 0.35, 0, 1));
      lk('M' + f(cx - wa * 0.78) + ' ' + f(33 + bh * 0.18) + ' Q' + f(cx) + ' ' + f(33 + bh * (0.8 + 0.5 * fa)) + ' ' + f(cx + wa * 0.78) + ' ' + f(33 + bh * 0.18), dk, 1.2, 0.5 * fa);
      lk('M' + f(cx - sh * 0.88) + ' ' + f(yt + bh * 0.12) + ' Q' + f(cx - sh) + ' ' + f(yt + bh * 0.3) + ' ' + f(cx - sh) + ' ' + f(yt + bh * 0.58), hi, 1.3, 0.4);
    } else if (view === 'N' || view === 'NE') {
      for (s = -1; s <= 1; s += 2) {
        ln('M' + f(cx + s * sh * 0.86) + ' ' + f(yt + bh * 0.35) + ' Q' + f(cx + s * wa * 0.55) + ' ' + f(33 + bh * 0.1) + ' ' + f(cx + s * wa * 0.28) + ' ' + f(33 + bh * 0.85), dk, 1.3, 0.12 + 0.8 * up);
        ln('M' + f(cx + s * sh * 0.1) + ' ' + f(yt + bh * 0.4) + ' Q' + f(cx + s * sh * 0.5) + ' ' + f(yt + bh * 0.7) + ' ' + f(cx + s * sh * 0.72) + ' ' + f(yt + bh * 0.5), dk, 1.1, 0.1 + 0.4 * up + 0.2 * thin);
        if (up > 0.6 || thin > 0.45) for (y = 0; y < 3; y++) ln('M' + f(cx + s * wa * 0.15) + ' ' + f(yt + bh * (0.85 + 0.28 * y)) + ' Q' + f(cx + s * wa * 0.5) + ' ' + f(yt + bh * (0.92 + 0.28 * y)) + ' ' + f(cx + s * wa * 0.8) + ' ' + f(yt + bh * (0.88 + 0.28 * y)), dk, 1.1, 0.5 * thin);
      }
      ln('M' + f(cx) + ' ' + f(yt + bh * 0.2) + ' L' + f(cx) + ' ' + f(33 + bh * 0.8), dk, 1.1, 0.3 + 0.2 * up);
    } else {
      ln('M' + f(cx + bw * 0.12) + ' ' + f(yt + bh * 0.62) + ' Q' + f(cx + sh * 0.7) + ' ' + f(yt + bh * 0.95) + ' ' + f(cx + sh * 0.95) + ' ' + f(yt + bh * 0.72), dk, 1.3, 0.18 + 0.8 * up);
      ln('M' + f(cx - sh * 0.3) + ' ' + f(yt + bh * 0.3) + ' Q' + f(cx - sh * 0.8) + ' ' + f(yt + bh * 0.8) + ' ' + f(cx - sh * 0.3) + ' ' + f(yt + bh * 1.05), dk, 1, 0.1 + 0.7 * up);
      lk('M' + f(cx + wa * 0.1) + ' ' + f(33 + bh * 0.15) + ' Q' + f(cx + wa * (0.9 + 0.3 * fa)) + ' ' + f(33 + bh * 0.55) + ' ' + f(cx + wa * 0.2) + ' ' + f(33 + bh * 0.95), dk, 1.2, 0.45 * fa);
      if (thin > 0.45) for (y = 0; y < 3; y++) ln('M' + f(cx - wa * 0.2) + ' ' + f(yt + bh * (0.95 + 0.26 * y)) + ' Q' + f(cx + wa * 0.1) + ' ' + f(yt + bh * (1.02 + 0.26 * y)) + ' ' + f(cx + wa * 0.5) + ' ' + f(yt + bh * (0.98 + 0.26 * y)), dk, 1.1, 0.5 * thin);
    }
    var by0 = 33 + bh * 0.3, bwid = Math.max(2, wn + (wa - wn) * 0.3 - 0.15), belt = 'M' + f(cx - bwid) + ' ' + f(by0) + ' Q' + f(cx) + ' ' + f(by0 + 1.8) + ' ' + f(cx + bwid) + ' ' + f(by0);
    return { d: d, wa: wa, sh: sh, wn: wn, yt: yt, yb: yb, belt: belt, lines: L, lit: 'translate(-0.9 -0.8) translate(24 33) scale(' + f(0.95) + ' ' + f(0.94) + ') translate(-24 -33)' };
  }

  /* arms: each arm is a shoulder, elbow and hand placed by the facing, so which arm shows and whether it sits in front of or behind the torso follows the body turn.
     South and north show both arms at the sides, a three-quarter turn shows the near arm over the torso and the far arm behind it, and profile shows the near arm only. */
  /* ---------- joint rig ----------
     One set of joint positions per build and facing. The arms, the skeleton and the X-ray silhouette all read these,
     so a bone always sits inside the limb that is drawn over it.
     Torso-attached joints (shoulder, elbow, hand) are in body space, which the pawn shifts up by RIG_SHIFT; hips, knees and ankles are in world space. */
  var FEET = { S: [19, 29, 41, 41], N: [19, 29, 41, 41], E: [21, 28.6, 41.4, 40.6], SE: [19.6, 28.6, 41, 41], NE: [19.6, 28.6, 41, 41] };
  /* Leg stance from the hips. wa is the torso's half-width at the hips (it already carries build, fat, fitness and sex), so hip size decides where the legs sit:
     each leg is centred under its hip joint, a little inside the hip edge, and gets thicker with the hips; a gap of at least one unit always separates the legs.
     Returns { x: [left, right] centres, y: [left, right] foot rows, hw: leg half-width, rx: boot half-length, c: hip-joint offset from the centre line }. */
  function legStance(view, wa, ph) {
    var hw = clamp(0.26 * wa + 0.8, 1.6, 3.5), c = Math.max(hw + 0.5, 0.54 * wa), k = { S: 1, N: 1, SE: 0.9, NE: 0.9, E: 0 }[view];
    if (k == null) k = 1;
    if (view === 'E') { var hwE = hw * 0.92, cE = hwE + 0.6; return { x: [24.8 - cE, 24.8 + cE], y: [40.6, 41.6], hw: hwE, rx: Math.max(3.4, hwE + 1.5), c: cE }; }
    return { x: [24 - c * k, 24 + c * k], y: [41, 41], hw: hw, rx: hw + 1.1, c: c * k };
  }
  var NECK = 4.4;
  function bodyShift(dmx) { return 4.6 + ((dmx && dmx.lg) || 0); }
  function armJoints(view, bw, bh, fit, dmx) {
    dmx = dmx || {};
    /* arms hang straight down the outside of the torso, shoulder at the top corner, the way a block figure's do: the arm's inner edge sits flush with the torso side instead of lying across the body */
    var ys = 33 - bh * 0.72, yh = 33 + bh * 0.6, sh = bw * (1 + 0.07 * fit + 0.02 * clamp(dmx.def || 0, 0, 1.2)) * (dmx.sh || 1), wd = 3.7 * (1 + 0.2 * clamp(fit, -0.6, 1) + 0.12 * clamp(dmx.def || 0, 0, 1.2));
    var def = { S: [[-1, 'f'], [1, 'f']], N: [[-1, 'f'], [1, 'f']], SE: [[-1, 'f'], [1, 'b']], NE: [[1, 'f'], [-1, 'b']], E: [[0, 'f']] }[view] || [[-1, 'f'], [1, 'f']];
    var out = [];
    def.forEach(function (q) {
      var s = q[0], far = q[1] === 'b', sx, hx, ex;
      if (view === 'E') { sx = 24.4; hx = 25.8; ex = 25.2; }
      else if (view === 'S' || view === 'N') { var so = sh + wd * 0.5 - 0.45; sx = 24 + s * so; hx = 24 + s * (so + 0.15); ex = 24 + s * (so + 0.08); }
      else { var so2 = sh * (far ? 0.8 : 0.9) + wd * (far ? 0.2 : 0.5) - 0.45; sx = 24 + s * so2; hx = 24 + s * (so2 + 0.15); ex = 24 + s * (so2 + 0.08); }
      var ey = ys + (yh - ys) * 0.5, k = 0.44;
      out.push({ s: s, far: far, sh: [sx, ys], el: [ex, ey], ha: [hx, yh], cuff: [ex + (hx - ex) * k, ey + (yh - ey) * k], wd: wd });
    });
    return out;
  }
  function armKit(view, bw, bh, fit, clothC, skin, skinShade, dmx) {
    var f = function (v) { return Math.round(v * 100) / 100; };
    var back = [], front = [], fL = [], fR = [], hl = null, hr = null;
    armJoints(view, bw, bh, fit, dmx).forEach(function (J) {
      var s = J.s, far = J.far, wd = J.wd, sx = J.sh[0], ys = J.sh[1], ex = J.el[0], ey = J.el[1], hx = J.ha[0], yh = J.ha[1], cx2 = J.cuff[0], cy2 = J.cuff[1];
      var o = [], sl = far ? shade(clothC, 0.7) : shade(clothC, 0.93), sk = far ? skinShade : skin;
      var up = 'M' + f(sx) + ' ' + f(ys) + ' L' + f(ex) + ' ' + f(ey) + ' L' + f(cx2) + ' ' + f(cy2), lo = 'M' + f(ex) + ' ' + f(ey) + ' L' + f(hx) + ' ' + f(yh - 1.6);
      o.push({ d: up + ' L' + f(hx) + ' ' + f(yh - 1.6), fill: 'none', stroke: INK, sw: f(wd + 1.5), op: 1 });
      o.push({ d: lo, fill: 'none', stroke: sk, sw: f(wd - 0.9), op: 1 });
      o.push({ d: up, fill: 'none', stroke: sl, sw: f(wd), op: 1 });
      o.push({ d: 'M' + f(sx - s * 0.5) + ' ' + f(ys + 0.6) + ' L' + f(ex - s * 0.4) + ' ' + f(ey - 0.2), fill: 'none', stroke: shade(clothC, 1.25), sw: 0.8, op: far ? 0.2 : 0.45 });
      var td = view === 'E' ? 1 : -s;
      o.push({ d: 'M' + f(hx + td * 1.9 - 1.0) + ' ' + f(yh - 0.6) + ' a1.0 1.6 ' + (td * 18) + ' 1 0 2.0 0 a1.0 1.6 ' + (td * 18) + ' 1 0 -2.0 0Z', fill: sk, stroke: INK, sw: 1.0, op: 1 });
      o.push({ d: 'M' + f(hx - 2.1) + ' ' + f(yh) + ' a2.1 2.5 0 1 0 4.2 0 a2.1 2.5 0 1 0 -4.2 0Z', fill: sk, stroke: INK, sw: 1.0, op: 1 });
      o.push({ d: 'M' + f(hx + td * 0.5 - 0.7) + ' ' + f(yh - 0.4) + ' q0.7 -0.7 1.4 0', fill: 'none', stroke: skinShade, sw: 0.6, op: far ? 0 : 0.7 });
      (far ? back : front).push(o); if (!far) (s < 0 ? fL : fR).push(o);
      if (view === 'E') { hl = 23.2; hr = hx; } else if (s < 0) hl = hx; else hr = hx;
    });
    return { back: [].concat.apply([], back), front: [].concat.apply([], front), frontL: [].concat.apply([], fL), frontR: [].concat.apply([], fR), hl: hl, hr: hr };
  }
  /* flat cel colours only: any translucent layer is mixed into its base colour so overlapping detail never builds up darker seams */
  function solid(items, base) {
    return (items || []).filter(function (q) { return q.op == null || q.op > 0.02; }).map(function (q) {
      if (q.op == null || q.op >= 1) return q;
      var o = {}, k; for (k in q) o[k] = q[k];
      if (o.fill && o.fill !== 'none') o.fill = mix(base, o.fill, q.op);
      if (o.stroke && o.stroke !== 'none') o.stroke = mix(base, o.stroke, q.op);
      o.op = 1; return o;
    });
  }
  function scaleTf(scale) { return 'translate(24 42) scale(' + (+scale).toFixed(3) + ') translate(-24 -42)'; }

  /* ---------- decay ----------
     decay runs 0..1 over DECAY_TICKS after death. Skin pales, bloats, greens, browns, then flesh falls away in patches until only bones are left. */
  var DECAY_TICKS = 1000;
  /* perishables. Clock: 50 ticks a day. Shelf life is in days at 15 C; spoilage speeds up about 2.2x per +10 C and nearly stops below freezing (a cold store keeps raw meat roughly four days,
     frozen ground for weeks). The same rate runs a body from fresh to bare bone in DECAY_TICKS ticks at 15 C: about three weeks, a month and a half at 5 C, months under ice. */
  var DAY_TICKS = 50;
  var PERISH = { fruit: { days: 3, label: 'fruit' }, meat: { days: 2, label: 'raw meat', risk: 0.6 }, berry: { days: 5, label: 'berries' }, nightshade: { days: 5, label: 'berries' }, mushroom: { days: 3, label: 'mushrooms' }, toadstool: { days: 3, label: 'mushrooms' }, cactus: { days: 14, label: 'prickly pear', frz: 3 }, cattail: { days: 6, label: 'cattail shoots', frz: 4 }, lichen: { days: 40, label: 'lichen', risk: 0.25 }, herb: { days: 7, label: 'herbs', frz: 3 }, tuber: { days: 90, label: 'roots', frz: 8 }, nut: { days: 180, label: 'nuts', risk: 0.25 }, grain: { days: 240, label: 'grain', risk: 0.3 }, apple: { days: 20, label: 'apples', frz: 4 }, grape: { days: 5, label: 'grapes' }, banana: { days: 4, label: 'bananas', frz: 2 }, date: { days: 60, label: 'dates' }, olive: { days: 14, label: 'olives' }, allium: { days: 40, label: 'wild onions', frz: 6 }, carrot: { days: 21, label: 'wild carrots', frz: 5 }, gourd: { days: 45, label: 'gourds', frz: 3 } };
  /* shelf-life multipliers for preserved food, for when colonists learn drying, smoking, cooking or cold storage */
  var PRESERVE = { cooked: 2, dried: 8, smoked: 10, salted: 12, cellar: 2.5 };
  function spoilRate(T) { return T <= -2 ? 0.04 : Math.max(0.04, Math.min(3, Math.pow(2.2, (T - 15) / 10))); }
  /* a store is a list of lots [kind, units, spoilage 0..1]. Counts change elsewhere (gathering, eating, hauling), so each tick the lots are brought back in line with the count (new units form a fresh lot,
     missing units come off the oldest lot first), then every lot ages. Returns { lots, gone: {kind: n} } where gone are the units that went off this tick. */
  function lotsStep(lots, count, kind, T, rk) {
    var out = [], i, sum = 0, gone = {}, rate = spoilRate(T) * (rk === undefined ? 1 : rk);
    for (i = 0; i < lots.length; i++) if (lots[i][1] > 0) { out.push([lots[i][0], lots[i][1], lots[i][2]]); sum += lots[i][1]; }
    if (count > sum) { var d = count - sum, last = out[out.length - 1]; if (last && last[0] === kind && last[2] < 0.05) last[1] += d; else out.push([kind, d, 0]); }
    else if (count < sum) { var rm = sum - count; for (i = 0; i < out.length && rm > 0; i++) { var t = Math.min(rm, out[i][1]); out[i][1] -= t; rm -= t; } }
    var keep = [];
    for (i = 0; i < out.length; i++) { var l = out[i]; if (l[1] <= 0) continue; var pk = PERISH[l[0]] || PERISH.berry; l[2] += rate / (pk.days * DAY_TICKS) + (pk.frz && T < -1.5 ? 1 / (pk.frz * DAY_TICKS) : 0); if (l[2] >= 1) gone[l[0]] = (gone[l[0]] || 0) + l[1]; else keep.push(l); }
    return { lots: keep, gone: gone };
  }
  /* Cold is not one thing. It slows bacteria and mould for every material (spoilRate), and below freezing it nearly stops them, so rotten stays rotten and sound stays sound until it warms.
     But the material decides the rest: roots, herbs, cattail and prickly pear are wrecked by ice crystals (PERISH.frz = days to ruin when kept frozen), while meat, berries, nuts and grain are fine frozen.
     Plant and fungal poisons (nightshade alkaloids, amatoxins) are stable chemicals: neither age, cold nor drying removes them, so a toxic batch stays toxic. What does change with age is the chance that
     spoiled food makes the eater ill (spoilRisk): nothing under half spoiled, rising after. */
  function spoilRisk(kind, sp) { var pk = PERISH[kind] || PERISH.berry; return Math.min(0.6, Math.max(0, (sp - 0.5) / 0.5) * (pk.risk || 0.45)); }
  var DECAY_STAGES = [['Fresh', 0], ['Pallid', 0.08], ['Bloated', 0.28], ['Rotting', 0.5], ['Skeletal', 0.75], ['Skeleton', 1]];
  function decayOf(D) {
    D = clamp(+D || 0, 0, 1);
    var st = 0; for (var i = 0; i < DECAY_STAGES.length; i++) if (D >= DECAY_STAGES[i][1]) st = i;
    var pale = clamp(D / 0.3, 0, 1), green = clamp((D - 0.2) / 0.3, 0, 1), brown = clamp((D - 0.45) / 0.35, 0, 1);
    function tint(c) { c = mix(c, '#c9c3b4', pale * 0.5); c = mix(c, '#7f9a6a', green * 0.55); return mix(c, '#4a3626', brown * 0.7); }
    return {
      D: D, stage: st, label: DECAY_STAGES[st][0], tint: tint,
      bloat: clamp((D - 0.15) / 0.2, 0, 1) * (1 - clamp((D - 0.5) / 0.3, 0, 1)) * 0.16,
      fleshOp: 1 - clamp((D - 0.78) / 0.17, 0, 1), boneOp: clamp((D - 0.7) / 0.2, 0, 1),
      organOp: 1 - clamp((D - 0.25) / 0.45, 0, 1), flies: D > 0.2 && D < 0.88 ? (D > 0.45 ? 3 : 2) : 0, wet: 1 - clamp((D - 0.6) / 0.3, 0, 1)
    };
  }
  var PATCHES = [[20.5, 31.5, 3.6, 0.42], [28, 34.5, 3.2, 0.5], [24, 29.4, 2.8, 0.58], [17.6, 35.2, 2.6, 0.62], [30.8, 30.4, 2.6, 0.66], [21.6, 16.4, 3.4, 0.6], [28.2, 21.6, 2.8, 0.7], [24, 36.6, 2.8, 0.68], [26, 14.6, 2.4, 0.76]];
  function ragged(cx, cy, r, seed) {
    var pts = [], n = 9;
    for (var i = 0; i < n; i++) { var a = i / n * Math.PI * 2, j = 0.72 + 0.5 * (((Math.sin(seed * 12.9 + i * 78.2) * 43758.5) % 1 + 1) % 1); pts.push((cx + Math.cos(a) * r * j).toFixed(2) + ' ' + (cy + Math.sin(a) * r * j * 0.9).toFixed(2)); }
    return 'M' + pts.join('L') + 'Z';
  }
  function rotItems(D) {
    var out = [], brown = clamp((D - 0.45) / 0.35, 0, 1);
    var rot = mix('#6a3a2a', '#2e2418', brown);
    PATCHES.forEach(function (q, i) {
      var g = clamp((D - q[3]) / 0.22, 0, 1); if (g <= 0) return;
      var r = q[2] * g;
      out.push({ d: ragged(q[0], q[1], r, i + 1), fill: rot, stroke: '#1c110b', sw: 0.9 });
      if (g > 0.55) out.push({ d: ragged(q[0] + 0.2, q[1] + 0.2, r * 0.52, i + 11), fill: '#efe6d0', stroke: '#1c110b', sw: 0.6 });
    });
    return out;
  }
  /* pawn skeleton, top-down, same footprint as the X-ray view. Items tagged g:'organ' (the brain cap) belong to the soft tissue. */
  function skeleton(view, bw, bh, dm) {
    if (typeof view === 'number') { dm = { fit: 0, fat: 0, lg: 0, hd: 1 }; bh = 8.4; bw = view; view = 'S'; }
    dm = dm || {}; view = FEET[view] ? view : 'S';
    var INKC = '#1c110b', BONE = '#efe6d0', items = [], fit = dm.fit || 0, shift = bodyShift(dm), hs = 0.9 * (dm.hd || 1);
    var BK = bodyKit(bw, bh, fit, dm.fat || 0, view, '#888888', dm), yt = BK.yt, yb = BK.yb;
    var AJ = armJoints(view, bw, bh, fit, dm), stn = legStance(view, BK.wa), ft = [stn.x[0], stn.x[1], stn.y[0], stn.y[1]];
    var f = function (v) { return Math.round(v * 100) / 100; };
    function E(cx, cy, rx, ry) { return 'M' + f(cx - rx) + ' ' + f(cy) + ' a' + f(rx) + ' ' + f(ry) + ' 0 1 0 ' + f(2 * rx) + ' 0 a' + f(rx) + ' ' + f(ry) + ' 0 1 0 ' + f(-2 * rx) + ' 0Z'; }
    function P(d, fill, stroke, sw, extra) { var o = { d: d, fill: fill || 'none', stroke: stroke || 'none', sw: sw || 0, op: 1, tf: 'rotate(0)' }; if (extra) for (var k in extra) o[k] = extra[k]; items.push(o); }
    function bone(d, w) { P(d, 'none', INKC, w + 1.4); P(d, 'none', BONE, w); }
    function seg(a, b, w) { bone('M' + f(a[0]) + ' ' + f(a[1]) + ' L' + f(b[0]) + ' ' + f(b[1]), w); }
    function joint(a, r) { P(E(a[0], a[1], r, r), BONE, INKC, 0.7); }
    var UP = 'translate(0 ' + f(-shift) + ')', HD = UP + ' translate(0 ' + f(-NECK) + ') translate(24 29.5) scale(' + f(hs) + ') translate(-24 -29.5)';
    function T(from, tf) { for (var i = from; i < items.length; i++) items[i].tf = tf; }
    // half-width of the torso at a body-space height (shoulder, chest, waist): ribs and pelvis stay inside it
    var yS = yt + bh * 0.58, yW = 33 + bh * 0.22;
    function half(y) { if (y <= yS) return BK.sh * (0.78 + 0.22 * Math.max(0, (y - yt) / (yS - yt))); return BK.sh + (BK.wn - BK.sh) * Math.min(1, (y - yS) / (yW - yS)); }
    var n0 = items.length;
    // spine: neck base to the pelvis, one vertebra per step
    var yN = yt + bh * 0.12, yP = yb - bh * 0.3, nv = 7, i, y;
    seg([24, yN], [24, yP], 0.6);
    var yJ = 29.5 + (25 - 29.5) * hs - NECK;
    seg([24, yN], [24, yJ + 0.6], 0.9); P(E(24, (yN + yJ) / 2 + 0.3, 0.95, 0.9), BONE, INKC, 0.6);
    for (i = 0; i <= nv; i++) { y = yN + (yP - yN) * i / nv; P(E(24, y, 0.95, 0.9), BONE, INKC, 0.6); }
    // ribs: three pairs, each sprung out from the spine and ending inside the torso outline
    [0.34, 0.52, 0.70].forEach(function (q, k) {
      var yy = yt + bh * (0.2 + q * 0.95), w = Math.max(2.2, half(yy) - 1.7 - k * 0.2);
      [-1, 1].forEach(function (s) { bone('M24 ' + f(yy) + ' Q' + f(24 + s * w * 0.7) + ' ' + f(yy - 1.2) + ' ' + f(24 + s * w) + ' ' + f(yy + 2.2), 0.9); });
    });
    // clavicles run from the neck to the real shoulder joint, so the arm bones hang from them
    AJ.forEach(function (J) { var s = J.s || 1; seg([24 + s * 1.1, yN + 0.4], J.sh, 1); });
    AJ.forEach(function (J) {
      if (J.far) { /* drawn first so the near arm covers it */ }
      seg(J.sh, J.el, 1.3); seg(J.el, [J.ha[0], J.ha[1] - 1.4], 1.1);
      joint(J.sh, 1.35); joint(J.el, 1.1);
      P(E(J.ha[0], J.ha[1], 1.9, 2.1), BONE, INKC, 0.8);
    });
    // pelvis: a bowl whose width follows the hips, with the hip sockets over the leg tops
    var pw = Math.min(BK.wa * 0.8, bw * 0.85), pyB = yb - bh * 0.34;
    bone('M' + f(24 - pw) + ' ' + f(pyB - 0.6) + ' Q' + f(24 - pw * 0.7) + ' ' + f(pyB + 2.4) + ' 24 ' + f(pyB + 2.8) + ' Q' + f(24 + pw * 0.7) + ' ' + f(pyB + 2.4) + ' ' + f(24 + pw) + ' ' + f(pyB - 0.6), 1.3);
    bone('M' + f(24 - pw * 0.7) + ' ' + f(pyB - 0.2) + ' Q24 ' + f(pyB + 1.4) + ' ' + f(24 + pw * 0.7) + ' ' + f(pyB - 0.2), 0.8);
    T(n0, UP);
    // legs, in world space: hip socket (inside the pelvis), knee, ankle, foot
    var n1 = items.length, hipY = pyB + 2.2 - shift, ankY = 41.2, kneeY = (hipY + ankY) / 2;
    [[ft[0], ft[2]], [ft[1], ft[3]]].forEach(function (q) {
      var x = q[0], hip = [x + (24 - x) * 0.25, hipY], kn = [x, kneeY], an = [x, ankY + (q[1] - 41) * 0.5];
      seg(hip, kn, 1.5); seg(kn, an, 1.3);
      joint(hip, 1.5); joint(kn, 1.3);
      bone(E(x, q[1] + 1.7, stn.hw * 0.9, 0.9), 0.9);
    });
    T(n1, 'translate(0 0)');
    // skull and jaw (head space, scaled with the head)
    var n2 = items.length;
    P(E(24, 18.4, 6.6, 6.2), BONE, INKC, 1);
    P(E(24, 25, 3.7, 1.8), BONE, INKC, 0.9);
    P(E(24, 13.6, 4.6, 2.2), '#f0a8b8', INKC, 0.9, { g: 'organ' });
    P('M22.4 13.2 q0.8 -0.8 1.6 0 M24.6 13.4 q0.8 -0.8 1.6 0', 'none', INKC, 0.5, { g: 'organ' });
    P(E(21.2, 18.9, 1.9, 2.2), INKC); P(E(26.8, 18.9, 1.9, 2.2), INKC);
    P('M24 21.6 l-0.9 1.5 h1.8Z', INKC);
    P('M22.2 24.9 h3.6 M23.4 24.2 v1.4 M24.6 24.2 v1.4', 'none', INKC, 0.55);
    T(n2, HD);
    return items;
  }

  /* ---------- light model: every shadow and every darkening comes from these numbers (1.0 = full daylight) ---------- */
  var LIGHT = {
    maxElev: 70,                 // sun elevation at noon, degrees
    twilight: 0.45,              // light with the sun on the horizon
    moon: 0.07,                  // light on a clear night
    darkK: 0.71,                 // night overlay opacity = darkness x darkK, so a clear night is about 0.66
    refH: 14,                    // px height of the reference object whose shadow length is measured
    maxShadow: 26,               // px cap on shadow length near the horizon
    weather: { clear: 1, rain: 0.7, storm: 0.5, drought: 1, 'cold snap': 0.9, 'heat wave': 1 },
    softness: { clear: 1, rain: 0.45, storm: 0.2, drought: 1, 'cold snap': 0.8, 'heat wave': 1 },
    hearth: { inner: 70, outer: 130, innerLight: 0.85, outerLight: 0.45 },
    cave: { mouth: 0.4, floor: 0.08 }, lamp: 36
  };
  function lightModel(tick, weather) {
    var tph = tick % 50, ssn = Math.sin(2 * Math.PI * ((tick % 2000) - 250) / 2000), elev = (LIGHT.maxElev - 10) * Math.sin(2 * Math.PI * tph / 50) + 12 * ssn, rad = elev * Math.PI / 180;
    var sunF = Math.min(1, Math.max(0, Math.sin(rad)) / Math.sin(LIGHT.maxElev * Math.PI / 180));
    var tw = Math.min(1, Math.max(0, (elev + 50) / 50));
    var base = elev > 0 ? LIGHT.twilight + (1 - LIGHT.twilight) * Math.pow(sunF, 0.5) : LIGHT.moon + (LIGHT.twilight - LIGHT.moon) * tw * tw;
    var wa = LIGHT.weather[weather] ?? 1, ambient = Math.min(1, base * wa);
    var len = elev > 0 ? Math.min(LIGHT.maxShadow, LIGHT.refH / Math.tan(Math.max(rad, 0.01))) : 0, az = 90 + Math.min(1, tph / 25) * 180;
    return { tph, elev, az, sun: sunF * wa, ambient, darkness: 1 - ambient, shadowLen: len, shadowDx: elev > 0 ? -Math.sin(az * Math.PI / 180) * len : 0, shadowDy: elev > 0 ? len * 0.7 + 2 : 0, shadowOp: elev > 0 ? (0.08 + 0.3 * sunF) * (LIGHT.softness[weather] ?? 1) : 0 };
  }
  
  /* ---------- symbol bubbles: every message is 1 to 3 glyphs on a 24 x 24 grid, drawn with the ink outline ---------- */
  function circ(cx, cy, r) { return 'M' + (cx - r) + ' ' + cy + ' a' + r + ' ' + r + ' 0 1 0 ' + (2 * r) + ' 0 a' + r + ' ' + r + ' 0 1 0 ' + (-2 * r) + ' 0Z'; }
  var GLYPH = {
    drink:   [['M12 2.5 C12 2.5 5 10.5 5 15 a7 7 0 0 0 14 0 C19 10.5 12 2.5 12 2.5Z', '#7cacdf'], ['M8.6 14.6 a3.6 3.6 0 0 0 2.2 3.4', 'none', '#e8f2fb', 1.8]],
    thirsty: [['M12 2.5 C12 2.5 5 10.5 5 15 a7 7 0 0 0 14 0 C19 10.5 12 2.5 12 2.5Z', '#b9ccdc'], ['M12 9.5 v5.2', 'none', '#1c110b', 2.2], [circ(12, 18, 1.2), '#1c110b', '#1c110b', 0.6]],
    eat:     [[circ(7.6, 15.4, 4.4), '#c2503c'], [circ(16.4, 15.4, 4.4), '#c2503c'], [circ(12, 9.4, 4.4), '#d8564a'], ['M12 5.4 C12 2 15.4 1 18 2.2 C16.8 4.8 14.6 5.6 12 5.4Z', '#7fbf8e'], [circ(10.6, 8, 1), '#f6d5cf', 'none', 0], [circ(6.2, 14, 1), '#f6d5cf', 'none', 0], [circ(15, 14, 1), '#f6d5cf', 'none', 0]],
    hungry:  [['M9.2 12.2 L5.6 15.6 L8.4 18.4 L11.8 14.8Z', '#f3e8d2'], ['M2.7 17.0 a2.2 2.2 0 1 0 4.4 0 a2.2 2.2 0 1 0 -4.4 0Z', '#f3e8d2'], ['M4.8 19.1 a2.2 2.2 0 1 0 4.4 0 a2.2 2.2 0 1 0 -4.4 0Z', '#f3e8d2'], ['M3.0000000000000004 17.0 a1.9 1.9 0 1 0 3.8 0 a1.9 1.9 0 1 0 -3.8 0Z', '#f3e8d2', 'none', 0], ['M5.1 19.1 a1.9 1.9 0 1 0 3.8 0 a1.9 1.9 0 1 0 -3.8 0Z', '#f3e8d2', 'none', 0], ['M6.2 15.8 L8.2 17.8 L10.6 14.2 L9.6 13.2Z', '#f3e8d2', 'none', 0], ['M10.2 7 C10.4 3.2 15.6 1.8 19 4.4 C22.4 7.2 21 12.8 16.8 14.2 C14 15.2 11.4 13.8 10.2 12 C9.6 10.4 10 8.6 10.2 7Z', '#c8733a'], ['M13.4 5.8 C14.8 4.6 17 4.6 18.2 5.8 C16.6 5.6 15 6.4 14.2 7.8 C13.2 7.6 13 6.6 13.4 5.8Z', '#e8a067', 'none', 0]],
    sleep:   [['M20 14.5A8 8 0 1 1 9.5 4 6.5 6.5 0 0 0 20 14.5Z', '#d4c63c'], ['M14 3 h5 l-5 5.4 h5', 'none', '#1c110b', 1.8]],
    hurt:    [['M9 3 h6 v6 h6 v6 h-6 v6 H9 v-6 H3 V9 h6Z', '#e08a78']],
    cold:    [['M12 2 V22 M3.4 7 L20.6 17 M3.4 17 L20.6 7', 'none', '#9cc4ee', 3.4], ['M12 2 V22 M3.4 7 L20.6 17 M3.4 17 L20.6 7', 'none', '#e8f2fb', 1.3]],
    hot:     [[circ(12, 12, 5.4), '#e8c63c'], ['M12 1.6 V4.4 M12 19.6 V22.4 M1.6 12 H4.4 M19.6 12 H22.4 M4.6 4.6 L6.6 6.6 M17.4 17.4 L19.4 19.4 M19.4 4.6 L17.4 6.6 M6.6 17.4 L4.6 19.4', 'none', '#1c110b', 1.8]],
    fire:    [['M12 2 C13 6 18 8 18 14 a6 6 0 0 1 -12 0 C6 11 8 9.6 8.6 7 C10 8 10.6 9 11 9.6 C11.6 7 11 4.6 12 2Z', '#e0742c'], ['M12 11 C12.6 13 15 13.6 15 16.4 a3 3 0 0 1 -6 0 C9 14.2 11.4 13.4 12 11Z', '#f2d04a']],
    scared:  [['M10 2.5 h4 l-0.8 12 h-2.4Z', '#e0504a'], [circ(12, 19.4, 2.2), '#e0504a']],
    angry:   [['M12 2 L14.6 8 L21 6.6 L17 12 L22 16.6 L15.4 16 L13.4 22 L11 16.2 L4.4 18.6 L7.6 12.6 L2.4 8 L9.2 8.4Z', '#e0504a']],
    love:    [['M12 21 C4 14.6 2.6 9.8 5.2 6.2 C7.6 3.4 11 4.6 12 7.4 C13 4.6 16.4 3.4 18.8 6.2 C21.4 9.8 20 14.6 12 21Z', '#e08a78']],
    idea:    [['M12 2.6 a6.4 6.4 0 0 0 -3.6 11.6 V17 h7.2 v-2.8 A6.4 6.4 0 0 0 12 2.6Z', '#f2d04a'], ['M9.6 20 h4.8 v1.8 H9.6Z', '#aeb6bd'], ['M10.6 9 q1.4 -1.8 3 0', 'none', '#1c110b', 1.4]],
    question:[['M8 8.4 C8 5 10.6 3.4 12.6 3.4 C15.2 3.4 17 5.2 17 7.6 C17 10.8 12.8 11 12.8 14.2', 'none', '#1c110b', 5], ['M8 8.4 C8 5 10.6 3.4 12.6 3.4 C15.2 3.4 17 5.2 17 7.6 C17 10.8 12.8 11 12.8 14.2', 'none', '#ede9e1', 2.4], [circ(12.8, 19, 1.9), '#ede9e1']],
    skull:   [['M12 2.6 C6.4 2.6 3.4 6.6 3.4 10.8 C3.4 13.8 5 15.4 7 16.4 V20.4 H17 V16.4 C19 15.4 20.6 13.8 20.6 10.8 C20.6 6.6 17.6 2.6 12 2.6Z', '#efe6d0'], [circ(8.6, 11, 2.4), '#1c110b', 'none', 0], [circ(15.4, 11, 2.4), '#1c110b', 'none', 0], ['M12 13 l-1.1 2.2 h2.2Z', '#1c110b', 'none', 0], ['M9.6 17 v3.2 M12 17 v3.2 M14.4 17 v3.2', 'none', '#1c110b', 1.2]],
    craft:   [['M5 20 L13 9', 'none', '#1c110b', 5], ['M5 20 L13 9', 'none', '#b98a5a', 2.6], ['M9.6 4.6 L19.8 3.4 L21 9.4 L14.4 12.6 L9.8 9Z', '#aeb6bd']],
    gather:  [['M12 21 C5 18 3.4 10 5.4 3.6 C12 4 19 7.6 18.6 14 C18.2 18 15 20.6 12 21Z', '#7fbf8e'], ['M12 20 C11 14 9.6 10 7.4 6.6', 'none', '#2f6a44', 1.6]],
    hunt:    [[circ(12, 12, 9), '#ede9e1'], [circ(12, 12, 5.4), '#e0504a'], [circ(12, 12, 2), '#ede9e1']],
    sick:    [[circ(12, 12, 9), '#7fbf3c'], [circ(8.6, 10, 1.4), '#1c110b', 'none', 0], [circ(15.4, 10, 1.4), '#1c110b', 'none', 0], ['M8 16.6 q2 -2 4 -0.4 t4 0', 'none', '#1c110b', 1.6]],
    home:    [['M3 12 L12 3.6 L21 12 V20.4 H3Z', '#c9a46a'], ['M9.6 20.4 V14.4 h4.8 v6Z', '#5a3d28']],
    flee:    [['M3 9 H13 V4 L21.4 12 L13 20 V15 H3Z', '#e0956a']],
    arrow:   [['M3 9.4 H13.4 V5 L21 12 L13.4 19 V14.6 H3Z', '#ede9e1']],
    warn:    [['M12 3 L22 20 H2Z', '#e8c63c'], ['M12 9 v5', 'none', '#1c110b', 2.2], [circ(12, 17, 1.1), '#1c110b', 'none', 0]]
  };
  var MSG = {
    dead:    { syms: ['skull'], text: 'Dead', prio: 100 },
    hurt:    { syms: ['hurt'], text: 'Hurt', prio: 90 },
    flee:    { syms: ['scared', 'flee'], text: 'Fleeing danger', prio: 85 },
    sick:    { syms: ['sick'], text: 'Sick', prio: 70 },
    hunt:    { syms: ['hunt'], text: 'Hunting', prio: 60 },
    craft:   { syms: ['craft'], text: 'Crafting', prio: 55 },
    gather:  { syms: ['gather'], text: 'Gathering materials', prio: 50 },
    drink:   { syms: ['drink'], text: 'Drinking', prio: 45 },
    eat:     { syms: ['eat'], text: 'Eating', prio: 45 },
    sleep:   { syms: ['sleep'], text: 'Sleeping', prio: 40 },
    cold:    { syms: ['cold', 'arrow', 'fire'], text: 'Cold, wants warmth', prio: 30 },
    hungry:  { syms: ['hungry', 'arrow', 'eat'], text: 'Hungry, wants food', prio: 25 },
    thirsty: { syms: ['thirsty', 'arrow', 'drink'], text: 'Thirsty, wants water', prio: 25 },
    tired:   { syms: ['sleep', 'arrow', 'home'], text: 'Tired, wants a bed', prio: 20 },
    angry:   { syms: ['angry'], text: 'Angry', prio: 35 },
    social:  { syms: ['love'], text: 'Content with company', prio: 10 },
    idea:    { syms: ['idea'], text: 'Has an idea', prio: 10 }
  };
  function glyphPaths(name) { return (GLYPH[name] || GLYPH.question).map(function (g) { return { d: g[0], f: g[1] || 'none', s: g[2] || '#1c110b', w: g[3] === undefined ? 1.6 : g[3] }; }); }
  /* ---------- the colony's sign vocabulary ----------
     A sign is not given: it is coined, by two people who both needed it at the same moment and
     were standing together. Only the signs you can make against your own body are there from the
     start -- an empty belly, a dry mouth, a closed eye, a wound, fright. Everything else has to be
     agreed on before anyone can mean it, and until it is, the hearer sees only that something was
     meant: a question.

     ARROW is the first piece of grammar rather than another noun. Before it is coined a speaker can
     only name one thing ("cold"); after it, two things can be set in relation ("cold -> fire"),
     which is the whole of what makes a sentence rather than a cry. So compound messages stay
     truncated to their first sign until the colony has coined the arrow. */
  var LEX_SEED = ['hungry', 'thirsty', 'sleep', 'hurt', 'scared'];
  var GLYPH_GLOSS = {
    hungry: 'empty belly', thirsty: 'dry mouth', sleep: 'closed eye', hurt: 'wound', scared: 'fright',
    arrow: 'leads to', eat: 'eating', drink: 'drinking', fire: 'fire', cold: 'cold', hot: 'heat',
    home: 'the camp', flee: 'run', sick: 'sickness', skull: 'death', hunt: 'the hunt', craft: 'making',
    gather: 'gathering', love: 'kinship', angry: 'anger', idea: 'a thought', question: 'unknown'
  };
  function msgSymNames(key) { var m = MSG[key]; return m ? m.syms.slice() : []; }
  function msgText(key) { var m = MSG[key]; return m ? m.text : ''; }
  function lexSeed() { return LEX_SEED.slice(); }
  function glyphGloss(n) { return GLYPH_GLOSS[n] || n; }
  // every sign a message would need if the colony could say all of it, so the coining loop knows what is wanted
  function msgWants(key) { return msgSymNames(key); }
  function msgSyms(key, lex) {
    var m = MSG[key]; if (!m) return [];
    var names = m.syms;
    if (lex) {
      // without the arrow there is no grammar, so only the first sign survives
      if (names.length > 1 && !lex.arrow) names = names.slice(0, 1);
      names = names.map(function (n) { return lex[n] ? n : 'question'; });
      // a bubble of nothing but questions says only that something was meant
      var allQ = names.every(function (n) { return n === 'question'; });
      if (allQ) names = ['question'];
    }
    return names.map(function (n) { return { name: n, paths: glyphPaths(n) }; });
  }

  /* ---------- mind: 8 personality traits (octagon) and 8 thought categories (bar or pie) ---------- */
  var PERS = ['Bravery', 'Sociability', 'Curiosity', 'Patience', 'Aggression', 'Diligence', 'Empathy', 'Greed'];
  var THOUGHTS = [
    { key: 'food', label: 'Food', color: '#e0956a' }, { key: 'water', label: 'Water', color: '#7cacdf' },
    { key: 'rest', label: 'Rest', color: '#b49ae0' }, { key: 'safety', label: 'Safety', color: '#e0504a' },
    { key: 'warmth', label: 'Warmth', color: '#f2d04a' }, { key: 'social', label: 'Social', color: '#e08a78' },
    { key: 'work', label: 'Work', color: '#7fbf8e' }, { key: 'wander', label: 'Wander', color: '#aea69a' }
  ];

  /* ---------- crafting: materials have properties, templates have parts, parts take materials, materials set the effects ---------- */
  var MATS = {
    wood:   { label: 'Wood',   color: '#8a5a38', hard: 3, sharp: 0, weight: 2,   flex: 3, ins: 1, rain: 1, dur: 4, src: 'forest' },
    stone:  { label: 'Stone',  color: '#8f8a80', hard: 5, sharp: 2, weight: 5,   flex: 0, ins: 0, rain: 3, dur: 3, src: 'hills, river cobbles and field stones' },
    flint:  { label: 'Flint',  color: '#4c5560', hard: 6, sharp: 5, weight: 3,   flex: 0, ins: 0, rain: 3, dur: 2, src: 'hills' },
    bone:   { label: 'Bone',   color: '#efe6d0', hard: 4, sharp: 3, weight: 2,   flex: 1, ins: 0, rain: 3, dur: 3, src: 'prey' },
    hide:   { label: 'Hide',   color: '#b98a5a', hard: 1, sharp: 0, weight: 1,   flex: 5, ins: 2, rain: 3, dur: 3, src: 'prey' },
    fur:    { label: 'Fur',    color: '#7a5a3a', hard: 0, sharp: 0, weight: 1,   flex: 4, ins: 5, rain: 2, dur: 2, src: 'prey' },
    fiber:  { label: 'Fiber',  color: '#a8b06a', hard: 0, sharp: 0, weight: 0.5, flex: 5, ins: 1, rain: 0, dur: 1, src: 'grass' },
    sinew:  { label: 'Sinew',  color: '#d9c9a0', hard: 0, sharp: 0, weight: 0.3, flex: 5, ins: 0, rain: 1, dur: 4, src: 'prey' },
    shell:  { label: 'Shell',  color: '#d9cdbb', hard: 4, sharp: 3.5, weight: 1, flex: 0, ins: 0, rain: 3, dur: 2, src: 'shores and river banks' },
    copper: { label: 'Native copper', color: '#b8693a', hard: 4.5, sharp: 3, weight: 4, flex: 1, ins: 0, rain: 3, dur: 4, src: 'stream beds and rocky ground' },
    resin:  { label: 'Resin',  color: '#c89a3c', hard: 1, sharp: 0, weight: 0.3, flex: 2, ins: 1, rain: 5, dur: 3, src: 'conifers and acacia' },
    horn:   { label: 'Horn',   color: '#c9b78a', hard: 4, sharp: 4, weight: 1.5, flex: 1, ins: 0, rain: 3, dur: 3, src: 'antlered and horned herd animals' },
    tusk:   { label: 'Tusk',   color: '#f2ead6', hard: 5, sharp: 4, weight: 2,   flex: 0, ins: 0, rain: 3, dur: 4, src: 'boar, peccary and elephant' },
    claw:   { label: 'Claw',   color: '#3a352e', hard: 4, sharp: 5, weight: 0.3, flex: 0, ins: 0, rain: 3, dur: 3, src: 'cats, bears, raptors and diggers' },
    feather:{ label: 'Feather',color: '#d8d2c4', hard: 0, sharp: 0, weight: 0.1, flex: 4, ins: 2, rain: 2, dur: 1, src: 'birds' },
    hoof:   { label: 'Hoof',   color: '#4a3e34', hard: 3, sharp: 0, weight: 1,   flex: 0, ins: 0, rain: 3, dur: 3, src: 'hoofed herd animals' },
    scale:  { label: 'Scales', color: '#6f8a5a', hard: 4, sharp: 0, weight: 1,   flex: 2, ins: 0, rain: 3, dur: 3, src: 'snakes and lizards' },
    fat:    { label: 'Tallow', color: '#efe2b0', hard: 0, sharp: 0, weight: 0.8, flex: 3, ins: 3, rain: 3, dur: 2, src: 'fat animals' },
    gold:   { label: 'Native gold', color: '#d4a82a', hard: 2, sharp: 0.5, weight: 6, flex: 2, ins: 0, rain: 3, dur: 2, src: 'stream gravels' }
  };
  var MAT_ORDER = ['wood', 'stone', 'flint', 'bone', 'hide', 'fur', 'fiber', 'sinew', 'resin', 'shell', 'horn', 'tusk', 'claw', 'feather', 'hoof', 'scale', 'fat', 'copper', 'gold'];
  var ALLM = MAT_ORDER;
  // part: [key, label, allowed materials, share of the item]
  var TPL = {
    weapon: {
      club:   { label: 'Striker', blunt: true, base: 2, hw: 0.7, sw: 0, reach: 1, wf: 0.5, parts: [['head', 'Head', ['wood', 'stone', 'bone', 'copper', 'shell', 'gold'], 1], ['grip', 'Grip', ['hide', 'fur', 'fiber'], 0.4]] },
      spear:  { label: 'Reacher', base: 1.5, hw: 0.3, sw: 0.9, reach: 2, wf: 0.4, parts: [['head', 'Point', ['flint', 'stone', 'bone', 'wood', 'shell', 'horn', 'tusk', 'claw', 'copper', 'gold'], 0.5], ['shaft', 'Shaft', ['wood', 'bone'], 1], ['bind', 'Binding', ['sinew', 'resin', 'fiber', 'hide'], 0.3]] },
      axe:    { label: 'Splitter', base: 2, hw: 0.5, sw: 0.7, reach: 1, wf: 0.6, parts: [['head', 'Head', ['stone', 'flint', 'bone', 'horn', 'tusk', 'copper', 'shell', 'gold'], 1], ['haft', 'Haft', ['wood', 'bone'], 0.8], ['bind', 'Binding', ['sinew', 'resin', 'fiber', 'hide'], 0.3]] },
      knife:  { label: 'Slicer', base: 1, hw: 0.1, sw: 1.1, reach: 0.5, wf: 0.25, parts: [['head', 'Edge', ['flint', 'bone', 'stone', 'shell', 'claw', 'horn', 'tusk', 'copper', 'gold'], 0.6], ['handle', 'Handle', ['wood', 'bone', 'hide', 'fur'], 0.5], ['bind', 'Binding', ['fiber', 'sinew'], 0.2]] },
      bow:    { label: 'Launcher', ranged: true, base: 1.5, hw: 0.2, sw: 0.5, reach: 5, wf: 0.3, parts: [['head', 'Arm', ['wood', 'bone'], 1], ['string', 'String', ['sinew', 'resin', 'fiber'], 0.3], ['grip', 'Grip', ['hide', 'fur', 'fiber'], 0.3]] }
    },
    armor: {
      vest:   { label: 'Torso cover', slotW: 1, parts: [['plate', 'Plates', ['hide', 'bone', 'scale', 'hoof', 'wood', 'stone'], 1], ['back', 'Backing', ['hide', 'fur', 'fiber'], 0.7], ['lace', 'Lacing', ['sinew', 'resin', 'fiber', 'hide'], 0.2]] },
      helm:   { label: 'Head cover', slotW: 0.4, parts: [['plate', 'Shell', ['bone', 'horn', 'scale', 'wood', 'stone', 'hide'], 1], ['back', 'Lining', ['fur', 'hide', 'fiber'], 0.5], ['lace', 'Strap', ['hide', 'sinew', 'fiber'], 0.2]] },
      greaves:{ label: 'Limb cover', slotW: 0.5, parts: [['plate', 'Plates', ['hide', 'bone', 'scale', 'hoof', 'wood', 'stone'], 1], ['back', 'Backing', ['hide', 'fur', 'fiber'], 0.6], ['lace', 'Lacing', ['sinew', 'resin', 'fiber', 'hide'], 0.2]] },
      shield: { label: 'Ward', slotW: 0.8, parts: [['plate', 'Face', ['wood', 'hide', 'bone'], 1], ['back', 'Rim', ['bone', 'wood', 'stone'], 0.5], ['lace', 'Handle', ['wood', 'hide', 'sinew'], 0.2]] }
    },
    clothing: {
      tunic:  { label: 'Body wrap', parts: [['body', 'Body', ['hide', 'fiber', 'fur'], 1], ['trim', 'Trim', ['fur', 'fiber', 'hide'], 0.3], ['stitch', 'Stitching', ['sinew', 'resin', 'fiber'], 0.15]] },
      cloak:  { label: 'Shoulder wrap', parts: [['body', 'Outer', ['hide', 'fur', 'fiber'], 1], ['trim', 'Lining', ['fur', 'fiber', 'hide'], 0.7], ['stitch', 'Clasp', ['bone', 'wood', 'stone'], 0.1]] },
      boots:  { label: 'Foot wrap', parts: [['body', 'Upper', ['hide', 'fur', 'fiber'], 0.8], ['trim', 'Sole', ['hide', 'hoof', 'wood', 'fur'], 0.6], ['stitch', 'Lacing', ['sinew', 'resin', 'fiber'], 0.15]] },
      gloves: { label: 'Hand wrap', parts: [['body', 'Palm', ['hide', 'fur', 'fiber'], 0.6], ['trim', 'Cuff', ['fur', 'fiber', 'hide'], 0.3], ['stitch', 'Stitching', ['sinew', 'resin', 'fiber'], 0.1]] },
      hat:    { label: 'Head wrap', parts: [['body', 'Crown', ['fur', 'hide', 'fiber'], 0.7], ['trim', 'Band', ['feather', 'fiber', 'hide', 'fur'], 0.2], ['stitch', 'Stitching', ['sinew', 'resin', 'fiber'], 0.1]] }
    }
  };

  /* ---------- worn gear, drawn on the body itself (48x48 pawn space) ----------
     Each spec is "template|mat0|mat1|mat2" (the item's parts in order). Returns shapes split by draw layer:
     feet (over the feet), hands (over the hands), body (over the torso, under the head), head (over the hair), held (over everything, right hand). */
  function wornParts(wa, wc, ww, bw, bh, hlx, hrx, geo) {
    geo = geo || {};
    var FX = geo.feet || [19, 29], FRX = geo.feetRx || 4.0;
    var INK = '#1c110b', out = { feet: [], hands: [], body: [], head: [], held: [] };
    function spec(s) { if (!s) return null; var a = String(s).split('|'); if (!a[0]) return null; var col = function (i) { var m = MATS[a[i + 1]]; return m ? m.color : '#8a5a38'; }; return { k: a[0], c0: col(0), c1: col(1), c2: col(2) }; }
    function E(cx, cy, rx, ry) { return 'M' + (cx - rx).toFixed(2) + ' ' + cy.toFixed(2) + ' a' + rx.toFixed(2) + ' ' + ry.toFixed(2) + ' 0 1 0 ' + (rx * 2).toFixed(2) + ' 0 a' + rx.toFixed(2) + ' ' + ry.toFixed(2) + ' 0 1 0 ' + (-rx * 2).toFixed(2) + ' 0 Z'; }
    var curFoot = -1;
    function P(layer, d, fill, stroke, sw) { var o = { d: d, fill: fill || 'none', stroke: stroke || 'none', sw: sw || 0 }; if (curFoot >= 0) o.f = curFoot; out[layer].push(o); }
    var cl = spec(wc), ar = spec(wa), wp = spec(ww);
    var L = 24 - bw, R = 24 + bw;
    var D = function (c) { return shade(c, 0.74); }, Lt = function (c) { return shade(c, 1.22); };
    if (cl) {
      if (cl.k === 'tunic') {
        P('body', E(24, 33, bw, bh), cl.c0, INK, 2);
        P('body', 'M' + (L + 1) + ' 36 Q24 43 ' + (R - 1) + ' 36 Q' + (R - 0.6) + ' 40.4 24 41 Q' + (L + 0.6) + ' 40.4 ' + (L + 1) + ' 36 Z', D(cl.c0));
        P('body', E(22.2, 30.6, Math.max(2, bw - 4.4), 4.6), Lt(cl.c0));
        P('body', 'M' + (L + 1.4) + ' 37.2 Q24 42.8 ' + (R - 1.4) + ' 37.2', 'none', cl.c1, 2.4);
        P('body', 'M20 25.6 L24 30.2 L28 25.6', 'none', cl.c2, 1.3);
        P('body', 'M24 30.4 L24 35', 'none', cl.c2, 0.9);
        P('body', E(24, 33, bw, bh), 'none', INK, 2);
      } else if (cl.k === 'cloak') {
        P('body', 'M' + (L - 1.6) + ' 37 Q' + (L - 1.2) + ' 25 24 23 Q' + (R + 1.2) + ' 25 ' + (R + 1.6) + ' 37 Q' + (R - 3) + ' 40.2 24 38 Q' + (L + 3) + ' 40.2 ' + (L - 1.6) + ' 37 Z', cl.c0, INK, 2);
        P('body', 'M' + (L - 0.8) + ' 36 Q' + (L + 3) + ' 39 24 37 Q' + (R - 3) + ' 39 ' + (R + 0.8) + ' 36 L' + (R + 1.2) + ' 37.4 Q' + (R - 3) + ' 40.6 24 38.4 Q' + (L + 3) + ' 40.6 ' + (L - 1.2) + ' 37.4 Z', D(cl.c0));
        P('body', 'M' + (L - 0.4) + ' 28 Q' + (L + 0.6) + ' 24.6 ' + (L + 4.4) + ' 25.4', 'none', cl.c1, 3.2);
        P('body', 'M' + (R + 0.4) + ' 28 Q' + (R - 0.6) + ' 24.6 ' + (R - 4.4) + ' 25.4', 'none', cl.c1, 3.2);
        P('body', 'M24 30 Q22.6 34 21 37.2 M24 30 Q25.4 34 27 37.2', 'none', D(cl.c0), 1);
        P('body', E(24, 29.4, 1.4, 1.4), cl.c2, INK, 1);
      } else if (cl.k === 'boots') {
        FX.forEach(function (x, fi) {
          curFoot = fi;
          P('body', E(x, 41.8, FRX, 3), cl.c0, INK, 1.8);
          P('body', 'M' + (x - 4.2) + ' 42.8 Q' + x + ' 45.8 ' + (x + 4.2) + ' 42.8', 'none', D(cl.c1), 2);
          P('body', 'M' + (x - 3.6) + ' 40 Q' + x + ' 38.2 ' + (x + 3.6) + ' 40', 'none', cl.c1, 2);
          P('body', 'M' + (x - 1.6) + ' 41.2 l1.2 0.6 M' + (x + 0.4) + ' 41.2 l1.2 0.6', 'none', cl.c2, 0.8);
          curFoot = -1;
        });
      } else if (cl.k === 'gloves') {
        (geo.gloves || [hlx, hrx]).forEach(function (x) {
          var gy = geo.gloveY || 33;
          P('body', E(x, gy, 3.7, 3.7), cl.c0, INK, 1.6);
          P('body', 'M' + (x - 3.4) + ' ' + (gy - 2.2) + ' Q' + x + ' ' + (gy - 4.4) + ' ' + (x + 3.4) + ' ' + (gy - 2.2), 'none', cl.c1, 2.2);
          P('body', 'M' + (x - 1.4) + ' ' + (gy + 0.6) + ' l0 1.8 M' + (x + 0.6) + ' ' + (gy + 0.6) + ' l0 1.8', 'none', D(cl.c0), 0.8);
        });
      } else if (cl.k === 'hat') {
        P('head', 'M13 16.2 Q12.6 8.8 24 8.6 Q35.4 8.8 35 16.2 Q24 19.2 13 16.2 Z', cl.c0, INK, 1.8);
        P('head', 'M15.4 13 Q18.6 9.8 23 9.6', 'none', Lt(cl.c0), 1.1);
        P('head', 'M13.4 15.9 Q24 19.3 34.6 15.9', 'none', cl.c1, 3);
        [15.4, 19.2, 24, 28.8, 32.6].forEach(function (x2) { var yy = 15.9 + (1 - Math.pow((x2 - 24) / 10.6, 2)) * 1.5; P('head', E(x2, yy + 0.3, 1.1, 1.1), Lt(cl.c1), INK, 0.6); });
        P('head', 'M24 9 L24 12.4', 'none', cl.c2, 1.1);
      }
    }
    if (ar) {
      if (ar.k === 'vest') {
        var rx = bw * 0.9, ry = bh * 0.86, cy = 33.2, rows = 3, rh = (2 * ry) / rows;
        P('body', E(24, cy, rx, ry), ar.c1, INK, 1.9);
        for (var j = 0; j < rows; j++) {
          var yt = cy - ry + j * rh, yc = yt + rh / 2, w = rx * Math.sqrt(Math.max(0.05, 1 - Math.pow((yc - cy) / ry, 2))) * 0.97;
          P('body', 'M' + (24 - w) + ' ' + (yt + 0.6) + ' h' + (2 * w) + ' v' + (rh - 0.4) + ' h' + (-2 * w) + ' Z', ar.c0, INK, 1.1);
          P('body', 'M' + (24 - w + 1.2) + ' ' + (yt + 1.5) + ' h' + (2 * w - 2.4), 'none', Lt(ar.c0), 0.9);
          P('body', 'M' + (24 - w + 1.2) + ' ' + (yt + rh - 0.4) + ' h' + (2 * w - 2.4), 'none', D(ar.c0), 0.9);
          [-0.5, 0.5].forEach(function (f) { P('body', 'M' + (24 + f * w) + ' ' + (yt + 0.8) + ' v' + (rh - 0.8), 'none', D(ar.c0), 0.7); });
        }
        P('body', 'M24 ' + (cy - ry + 0.6) + ' L24 ' + (cy + ry - 0.6), 'none', ar.c2, 1.1);
        for (var q = 0; q < rows; q++) P('body', 'M22.8 ' + (cy - ry + q * rh + rh * 0.3) + ' L25.2 ' + (cy - ry + q * rh + rh * 0.7) + ' M25.2 ' + (cy - ry + q * rh + rh * 0.3) + ' L22.8 ' + (cy - ry + q * rh + rh * 0.7), 'none', ar.c2, 0.7);
        P('body', E(24, cy, rx, ry), 'none', INK, 1.9);
      } else if (ar.k === 'greaves') {
        FX.forEach(function (x, fi) {
          curFoot = fi;
          P('body', 'M' + (x - 3.6) + ' 37 h7.2 v4.2 h-7.2 Z', ar.c0, INK, 1.4);
          P('body', 'M' + (x - 3) + ' 38 h6', 'none', Lt(ar.c0), 0.8);
          P('body', 'M' + (x - 3.6) + ' 39.2 h7.2', 'none', ar.c2, 1);
          P('body', E(x, 37, 2.2, 1.3), ar.c1, INK, 0.9);
          curFoot = -1;
        });
      } else if (ar.k === 'helm') {
        P('head', 'M13.6 20.4 A10.4 10.4 0 0 1 34.4 20.4 L33 17.4 Q24 14.2 15 17.4 Z', ar.c0, INK, 1.9);
        P('head', 'M16.2 14.8 Q19.6 11.4 24 11', 'none', Lt(ar.c0), 1.1);
        P('head', 'M14.6 17.6 Q24 14.4 33.4 17.6', 'none', ar.c1, 2.6);
        P('head', 'M24 10.4 L24 14.8', 'none', ar.c1, 1.5);
        [17.4, 24, 30.6].forEach(function (x2) { P('head', E(x2, 15.9 + (Math.abs(x2 - 24) / 6.6) * 0.8, 0.55, 0.55), ar.c2, INK, 0.4); });
        P('head', 'M14.2 19 L13.8 25.6 Q16.4 27.8 18.6 25.4 L17.8 19.8 Z', ar.c0, INK, 1.5);
        P('head', 'M33.8 19 L34.2 25.6 Q31.6 27.8 29.4 25.4 L30.2 19.8 Z', ar.c0, INK, 1.5);
        P('head', 'M15.4 22 L16.4 25 M32.6 22 L31.6 25', 'none', ar.c2, 0.8);
      } else if (ar.k === 'shield') {
        var sx = geo.shield != null ? geo.shield : hlx - 1.6, sy = 32.2, srx = geo.shieldRx || 6.6, kx = srx / 6.6;
        P('body', E(sx, sy, srx, 7.4), ar.c0, INK, 1.9);
        P('body', 'M' + (sx - 2.2 * kx) + ' ' + (sy - 6.2) + ' v12.4 M' + (sx + 2.2 * kx) + ' ' + (sy - 6.2) + ' v12.4', 'none', D(ar.c0), 0.8);
        P('body', E(sx, sy, 5.2 * kx, 6), 'none', ar.c1, 1.7);
        P('body', 'M' + (sx - 4) + ' ' + (sy - 3.4) + ' Q' + (sx - 2.4) + ' ' + (sy - 5.4) + ' ' + (sx - 0.4) + ' ' + (sy - 5.6), 'none', Lt(ar.c0), 0.9);
        P('body', E(sx, sy, 1.7, 1.7), ar.c2, INK, 1);
      }
    }
    if (wp) {
      var x = geo.held != null ? geo.held : hrx + 0.4;
      if (wp.k === 'spear') {
        P('held', 'M' + x + ' 42 L' + x + ' 11.5', 'none', INK, 3.4);
        P('held', 'M' + x + ' 42 L' + x + ' 11.5', 'none', wp.c1, 1.7);
        P('held', 'M' + (x - 2) + ' 12.4 L' + x + ' 5.6 L' + (x + 2) + ' 12.4 Z', wp.c0, INK, 1.3);
        P('held', 'M' + (x - 1.7) + ' 13.4 h3.4', 'none', wp.c2, 1.2);
      } else if (wp.k === 'axe') {
        P('held', 'M' + x + ' 41 L' + x + ' 20', 'none', INK, 3.4);
        P('held', 'M' + x + ' 41 L' + x + ' 20', 'none', wp.c1, 1.7);
        P('held', 'M' + (x - 0.8) + ' 20.6 Q' + (x + 5.8) + ' 17.4 ' + (x + 5.6) + ' 25.8 Q' + (x + 2.6) + ' 24.4 ' + (x - 0.8) + ' 25.2 Z', wp.c0, INK, 1.4);
        P('held', 'M' + (x - 1.4) + ' 24 h2.8', 'none', wp.c2, 1.1);
      } else if (wp.k === 'club') {
        P('held', 'M' + x + ' 41 L' + x + ' 22', 'none', INK, 4);
        P('held', 'M' + x + ' 41 L' + x + ' 22', 'none', wp.c1, 2.2);
        P('held', E(x, 20.8, 3.2, 4), wp.c0, INK, 1.5);
      } else if (wp.k === 'knife') {
        P('held', 'M' + x + ' 34 L' + x + ' 28.5', 'none', INK, 3);
        P('held', 'M' + x + ' 34 L' + x + ' 28.5', 'none', wp.c1, 1.5);
        P('held', 'M' + (x - 1.4) + ' 28.6 L' + x + ' 21 L' + (x + 1.4) + ' 28.6 Z', wp.c0, INK, 1.3);
      } else if (wp.k === 'bow') {
        P('held', 'M' + x + ' 17 Q' + (x + 7.4) + ' 31 ' + x + ' 45', 'none', INK, 3.4);
        P('held', 'M' + x + ' 17 Q' + (x + 7.4) + ' 31 ' + x + ' 45', 'none', wp.c0, 1.8);
        P('held', 'M' + x + ' 17 L' + x + ' 45', 'none', wp.c1, 0.8);
      }
      if (geo.skin && geo.handY) P('held', E(x, geo.handY, 2.1, 2.4), geo.skin, INK, 1.6);
    }
    return out;
  }
  function wornKey(it) { return it ? [it.tplKey].concat(it.mats || []).join('|') : ''; }
  var QUALITY = [['Crude', 0.8, 0], ['Fair', 1, 1], ['Fine', 1.15, 3], ['Masterwork', 1.3, 5]];
  function qualityOf(trainedYears, roll) { var i = 0; var eff = trainedYears + (roll || 0) * 1.6 - 0.6; for (var k = 0; k < QUALITY.length; k++) if (eff >= QUALITY[k][2]) i = k; return i; }
  var EFFECTS = [
    { key: 'keen',   label: 'Keen edge',   cats: ['weapon'], test: function (s) { return s.partMax('head', 'sharp') >= 4; }, text: 'Cuts deep: raises hunting kill chance' },
    { key: 'crush',  label: 'Crushing',    cats: ['weapon'], test: function (s) { return s.tpl.blunt && s.partMax('head', 'hard') >= 4; }, text: 'Stuns: predators back off for longer' },
    { key: 'reach',  label: 'Long reach',  cats: ['weapon'], test: function (s) { return s.tpl.reach >= 2; }, text: 'Hits before the animal is adjacent' },
    { key: 'ranged', label: 'Ranged',      cats: ['weapon'], test: function (s) { return !!s.tpl.ranged; }, text: 'Hunts prey from five tiles' },
    { key: 'light',  label: 'Light',       cats: ['weapon', 'armor'], test: function (s) { return s.weight <= 3.4; }, text: 'No speed penalty' },
    { key: 'heavy',  label: 'Heavy',       cats: ['weapon', 'armor'], test: function (s) { return s.weight >= 9; }, text: 'Slows the wearer' },
    { key: 'sturdy', label: 'Sturdy',      cats: ['weapon', 'armor', 'clothing'], test: function (s) { return s.dur >= 7; }, text: 'Lasts a long time' },
    { key: 'brittle',label: 'Brittle',     cats: ['weapon', 'armor', 'clothing'], test: function (s) { return s.dur <= 3.2; }, text: 'Wears out fast' },
    { key: 'padded', label: 'Padded',      cats: ['armor'], test: function (s) { return s.partMax('back', 'flex') >= 4 && s.partMax('back', 'ins') >= 2; }, text: 'Soaks up blunt hits' },
    { key: 'plated', label: 'Hard plates', cats: ['armor'], test: function (s) { return s.partMax('plate', 'hard') >= 4; }, text: 'Strong against piercing' },
    { key: 'warm',   label: 'Warm',        cats: ['armor', 'clothing'], test: function (s) { return s.ins >= 4.5; }, text: 'Extra insulation in the cold' },
    { key: 'dry',    label: 'Waterproof',  cats: ['clothing', 'armor'], test: function (s) { return s.rain >= 2.6; }, text: 'No chill in rain' },
    { key: 'quiet',  label: 'Quiet',       cats: ['weapon', 'clothing'], test: function (s) { return s.mats.some(function (m) { return m === 'fur' || m === 'hide'; }) && s.weight <= 4; }, text: 'Predators notice less' },
    { key: 'camo',   label: 'Camouflage',  cats: ['clothing', 'armor'], test: function (s) { return s.mats.filter(function (m) { return m === 'fur' || m === 'hide' || m === 'fiber'; }).length >= 2; }, text: 'Earth tones blend with the ground' }
  ];
  function craftStats(cat, tplKey, mats, q, qRoll) {
    var T = TPL[cat] && TPL[cat][tplKey]; if (!T) return null;
    var M = mats.map(function (k) { return MATS[k] || MATS.fiber; });
    var by = {}; T.parts.forEach(function (p, i) { by[p[0]] = M[i]; });
    var qi = typeof q === 'number' ? q : 1, qm = QUALITY[Math.max(0, Math.min(3, qi))][1];
    var weight = 0, ins = 0, rain = 0, flexAvg = 0, durMin = 99, sharp = 0, hard = 0, tw = 0;
    T.parts.forEach(function (p, i) { var m = M[i], w = p[3]; weight += m.weight * w * 2; ins += m.ins * w; rain += m.rain * w; flexAvg += m.flex * w; tw += w; durMin = Math.min(durMin, m.dur * (p[0] === 'bind' || p[0] === 'lace' || p[0] === 'stitch' || p[0] === 'string' ? 1.4 : 1)); });
    ins = ins / tw * 1.6; rain = rain / tw; flexAvg = flexAvg / tw;
    var s = { cat: cat, tpl: T, mats: mats, weight: Math.round(weight * 10) / 10, ins: 0, rain: 0, dur: 0, q: qi, partMax: function (part, prop) { return (by[part] || {})[prop] || 0; } };
    var out = { cat: cat, tplKey: tplKey, label: T.label, mats: mats.slice(), q: qi, quality: QUALITY[qi][0], weight: s.weight, effects: [], stats: [] };
    s.dur = Math.round(durMin * 2 * qm * 10) / 10; out.dur = s.dur; out.durMax = s.dur;
    if (cat === 'weapon') {
      var h = by.head; sharp = h.sharp; hard = h.hard;
      out.dmg = Math.round((T.base + hard * T.hw + sharp * T.sw) * qm * 10) / 10;
      out.reach = T.reach; out.speed = Math.round(10 / (3 + s.weight * T.wf) * 10) / 10;
      out.stats = [['Damage', out.dmg], ['Reach', out.reach], ['Speed', out.speed], ['Weight', s.weight], ['Durability', s.dur]];
    } else if (cat === 'armor') {
      var pl = by.plate, bk = by.back;
      out.blunt = Math.round((pl.hard * 0.5 + bk.flex * 0.45 + bk.ins * 0.2) * T.slotW * qm * 10) / 10;
      out.pierce = Math.round((pl.hard * 0.8 + pl.sharp * 0.1) * T.slotW * qm * 10) / 10;
      out.cut = Math.round((pl.hard * 0.6 + pl.flex * 0.3) * T.slotW * qm * 10) / 10;
      out.prot = Math.round((out.blunt + out.pierce + out.cut) / 3 * 10) / 10;
      s.ins = ins * T.slotW; s.rain = rain; out.ins = Math.round(s.ins * 10) / 10; out.rain = Math.round(rain * 10) / 10;
      out.stats = [['Blunt', out.blunt], ['Pierce', out.pierce], ['Cut', out.cut], ['Warmth', out.ins], ['Weight', s.weight], ['Durability', s.dur]];
    } else {
      s.ins = ins * qm; s.rain = rain; out.ins = Math.round(s.ins * 10) / 10; out.rain = Math.round(rain * 10) / 10;
      out.comfort = Math.round((flexAvg * 1.1 + 1) * qm * 10) / 10;
      out.stats = [['Warmth', out.ins], ['Rain guard', out.rain], ['Comfort', out.comfort], ['Weight', s.weight], ['Durability', s.dur]];
    }
    out.stats = out.stats.map(function (r) { return { k: r[0], v: r[1] }; });
    EFFECTS.forEach(function (e) { if (e.cats.indexOf(cat) >= 0 && e.test(s)) out.effects.push({ key: e.key, label: e.label, text: e.text }); });
    out.name = QUALITY[qi][0] + ' ' + (by[T.parts[0][0]] ? by[T.parts[0][0]].label.toLowerCase() + ' ' : '') + T.label.toLowerCase();
    return out;
  }
  // choose the best allowed material per part from what is on hand (counts), scoring by what the part needs
  function pickMats(cat, tplKey, avail) {
    var T = TPL[cat] && TPL[cat][tplKey]; if (!T) return null;
    var left = {}; for (var k in avail) left[k] = avail[k];
    var out = [];
    for (var i = 0; i < T.parts.length; i++) {
      var p = T.parts[i], best = null, bs = -1;
      p[2].forEach(function (mk) {
        if (!(left[mk] > 0)) return;
        var m = MATS[mk], sc;
        if (p[0] === 'head' && cat === 'weapon') sc = m.sharp * 1.2 + m.hard;
        else if (p[0] === 'plate') sc = m.hard * 1.2 + m.rain * 0.2;
        else if (p[0] === 'back' || p[0] === 'trim' || p[0] === 'body') sc = (cat === 'clothing' ? m.ins * 1.2 + m.rain * 0.6 : m.flex + m.ins * 0.5);
        else sc = m.dur + m.flex * 0.4;
        sc -= p[2].indexOf(mk) * 0.01;
        if (sc > bs) { bs = sc; best = mk; }
      });
      if (!best) return null;
      left[best] -= 1; out.push(best);
    }
    return out;
  }
  // 48 x 48 drawing of an item, one shape per part (so the look follows the materials)
  function itemShapes(cat, tplKey, mats) {
    var T = TPL[cat] && TPL[cat][tplKey], sh = [], INKC = '#1c110b';
    if (!T) return sh;
    var C = mats.map(function (k) { return (MATS[k] || MATS.fiber).color; });
    function add(d, fill, sw, stroke) { sh.push({ d: d, fill: fill || 'none', stroke: stroke || INKC, sw: sw === undefined ? 1.6 : sw }); }
    function line(d, col, w) { sh.push({ d: d, fill: 'none', stroke: INKC, sw: w + 1.6 }); sh.push({ d: d, fill: 'none', stroke: col, sw: w }); }
    if (cat === 'weapon') {
      if (tplKey === 'club') { line('M10 41 L30 15', C[1] === C[0] ? shade(C[1], 0.8) : C[1], 4.2); add('M25 9 C31 4 40 8 38 15 C36 21 29 22 25 18 C22 15 22 12 25 9Z', C[0]); line('M12.6 37.6 L17 31.8', C[1], 4.6); }
      if (tplKey === 'spear') { line('M7 43 L36 10', C[1], 3); line('M10.4 39 L14 35', C[2], 4); line('M31 14.6 L33.6 11.8', C[2], 4.4); add('M34 14 L44 3 L40 17Z', C[0]); }
      if (tplKey === 'axe') { line('M9 42 L33 12', C[1], 3.6); add('M28 6 C38 4 44 10 42 20 C37 20 31 17 29 14 C27 11 27 8 28 6Z', C[0]); line('M27 15.4 L31.6 19', C[2], 3.4); }
      if (tplKey === 'knife') { line('M10 40 L22 26', C[1], 5); add('M21 28 L40 6 C42 14 38 24 27 32Z', C[0]); line('M18 31.4 L23 26.6', C[2], 3.4); }
      if (tplKey === 'bow') { line('M16 6 C34 16 34 32 16 42', C[0], 3.6); line('M16 6 L16 42', C[1], 1.2); line('M24.6 21 L26.4 27', C[2], 4.4); }
    } else if (cat === 'armor') {
      if (tplKey === 'vest') { add('M12 8 L20 6 L24 10 L28 6 L36 8 L38 18 L34 40 L14 40 L10 18Z', C[1]); add('M14 14 L34 14 L33 24 L15 24Z', C[0]); add('M15 26 L33 26 L32 36 L16 36Z', C[0]); line('M24 10 L24 40', C[2], 1.4); }
      if (tplKey === 'helm') { add('M9 30 C9 12 17 6 24 6 C31 6 39 12 39 30 L33 34 L15 34Z', C[0]); add('M13 30 C13 22 17 18 24 18 C31 18 35 22 35 30 L24 34Z', C[1]); line('M13 34 L35 34', C[2], 1.6); }
      if (tplKey === 'greaves') { add('M10 6 L21 6 L20 40 L10 40Z', C[0]); add('M27 6 L38 6 L38 40 L28 40Z', C[0]); add('M10 6 L21 6 L21 12 L10 12Z', C[1]); add('M27 6 L38 6 L38 12 L27 12Z', C[1]); line('M10 24 L21 24 M27 24 L38 24', C[2], 1.6); }
      if (tplKey === 'shield') { add('M24 4 C34 4 42 8 42 20 C42 33 33 42 24 45 C15 42 6 33 6 20 C6 8 14 4 24 4Z', C[1]); add('M24 9 C32 9 37 12 37 20 C37 30 31 37 24 40 C17 37 11 30 11 20 C11 12 16 9 24 9Z', C[0]); add(circ(24, 22, 3.4), C[2]); }
    } else {
      if (tplKey === 'tunic') { add('M8 12 L18 7 Q24 12 30 7 L40 12 L37 22 L32 20 L33 41 L15 41 L16 20 L11 22Z', C[0]); add('M18 7 Q24 12 30 7 L28 9 Q24 14 20 9Z', C[1]); line('M16 32 L32 32', C[2], 1.4); }
      if (tplKey === 'cloak') { add('M14 8 Q24 3 34 8 L42 42 L6 42Z', C[0]); add('M14 8 Q24 3 34 8 L33 14 Q24 10 15 14Z', C[1]); add(circ(24, 12, 2.2), shade(C[2], 1.05)); }
      if (tplKey === 'boots') { add('M8 10 L19 10 L20 30 L27 33 L27 41 L8 41Z', C[0]); add('M8 36 L27 36 L27 41 L8 41Z', C[1]); add('M28 10 L39 10 L40 30 L45 33 L45 41 L28 41Z', C[0]); add('M28 36 L45 36 L45 41 L28 41Z', C[1]); line('M10 18 L18 18 M30 18 L38 18', C[2], 1.2); }
      if (tplKey === 'gloves') { add('M9 20 L19 20 L20 12 L23 12 L23 22 L24 42 L11 42Z', C[0]); add('M11 36 L24 36 L24 42 L11 42Z', C[1]); add('M26 20 L36 20 L37 12 L40 12 L40 22 L41 42 L28 42Z', C[0]); line('M11 32 L22 32', C[2], 1.2); }
      if (tplKey === 'hat') { add('M10 30 C10 14 17 8 24 8 C31 8 38 14 38 30Z', C[0]); add('M6 32 C6 28 42 28 42 32 C42 37 6 37 6 32Z', C[1]); line('M10 30 L38 30', C[2], 1.6); }
    }
    return sh;
  }

  /* ---------- survival colours: one colour per value, used by bars, icons, floating text and particles ---------- */
  var NEEDCOL = {
    sat:    { label: 'Satiety',   color: '#e0956a', low: 'Hungry',    glyph: 'hungry' },
    hyd:    { label: 'Hydration', color: '#5aa0e0', low: 'Thirsty',   glyph: 'thirsty' },
    ene:    { label: 'Energy',    color: '#b49ae0', low: 'Exhausted', glyph: 'sleep' },
    hea:    { label: 'Health',    color: '#6fcf8a', low: 'Failing',   glyph: 'hurt' },
    stress: { label: 'Stress',    color: '#f2d04a', low: 'Panicked',  glyph: 'scared' },
    inj:    { label: 'Injury',    color: '#e0504a', low: 'Wounded',   glyph: 'hurt' },
    cold:   { label: 'Cold',      color: '#9cdcf0', low: 'Freezing',  glyph: 'cold' },
    hot:    { label: 'Heat',      color: '#ff8a3c', low: 'Overheated', glyph: 'hot' },
    sick:   { label: 'Infection', color: '#9acd32', low: 'Infected',  glyph: 'sick' },
    san:    { label: 'Sanity',    color: '#d45ee0', low: 'Unhinged',  glyph: 'scared' },
    hap:    { label: 'Happiness', color: '#ffb8d0', low: 'Miserable', glyph: 'love' }
  };

  /* ---------- people: names, archetypes, traits (RimWorld-style), strengths and flaws ---------- */

  /* ---------- naming: every person gets a unique given name built from their people's syllables (four tongues), and a house name that passes down the mother's line ---------- */
  var TONGUES = [
    { on: ['b', 'd', 'k', 'l', 'm', 'n', 'r', 's', 't', 'v', 'br', 'kr', 'th'], vo: ['a', 'e', 'i', 'o', 'u', 'ae'], co: ['', '', 'n', 'r', 'l', 's', 'th'] },
    { on: ['g', 'h', 'j', 'k', 'm', 'n', 'p', 'sh', 'ch', 'z', 'y'], vo: ['a', 'i', 'o', 'u', 'oa', 'ei'], co: ['', '', 'k', 'n', 'm', 'sh'] },
    { on: ['f', 'l', 'm', 'n', 'r', 's', 'w', 'y', 'fr', 'gl', 'sl'], vo: ['a', 'e', 'i', 'y', 'ia', 'io'], co: ['', 'l', 'n', 'rn', 'ss', 'nd'] },
    { on: ['d', 'q', 't', 'x', 'z', 'v', 'r', 'kh', 'zh', 'tr'], vo: ['a', 'o', 'u', 'ai', 'au'], co: ['', 'r', 'x', 'th', 'rk', 'm'] }
  ];
  var HOUSE_A = ['Ash', 'Black', 'Bright', 'Cold', 'Crow', 'Dun', 'Elder', 'Fair', 'Grey', 'Hollow', 'Iron', 'Kestrel', 'Long', 'Mist', 'Oak', 'Raven', 'Red', 'Salt', 'Stone', 'Thorn', 'White', 'Wolf', 'Winter', 'Amber', 'Fen', 'Gold', 'Hawk', 'Moss'];
  var HOUSE_B = ['bend', 'brook', 'crest', 'dale', 'fall', 'field', 'ford', 'gate', 'hall', 'holm', 'mere', 'moor', 'reach', 'run', 'shaw', 'stead', 'vale', 'watch', 'wick', 'wood', 'wold', 'combe', 'tarn', 'ridge'];
  function genName(rng, used, tongue) {
    var T = TONGUES[(tongue || 0) % TONGUES.length], pk = function (a) { return a[Math.floor(rng() * a.length)]; };
    for (var t = 0; t < 60; t++) {
      var n = rng() < 0.78 ? 2 : 3, w = '';
      for (var i = 0; i < n; i++) w += pk(T.on) + pk(T.vo) + (i === n - 1 ? pk(T.co) : (rng() < 0.2 ? pk(T.co) : ''));
      w = w.charAt(0).toUpperCase() + w.slice(1);
      if (w.length >= 3 && w.length <= 8 && !(used && used.has(w))) return w;
    }
    var b = pk(T.on) + pk(T.vo), k = 2; b = b.charAt(0).toUpperCase() + b.slice(1);
    while (used && used.has(b + ' ' + ['II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][k - 2])) k++;
    return b + ' ' + ['II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'][Math.min(8, k - 2)];
  }
  function genHouse(rng, used) {
    for (var t = 0; t < 80; t++) { var h = HOUSE_A[Math.floor(rng() * HOUSE_A.length)] + HOUSE_B[Math.floor(rng() * HOUSE_B.length)]; if (!(used && used.has(h))) return h; }
    return HOUSE_A[0] + HOUSE_B[0] + Math.floor(rng() * 99);
  }
  var NAMES = ['Ash', 'Bryn', 'Cael', 'Dara', 'Eno', 'Fenn', 'Gale', 'Hale', 'Isla', 'Jory', 'Kest', 'Lio', 'Mara', 'Nox', 'Orin', 'Pell', 'Quin', 'Rue', 'Sable', 'Tamsin', 'Ulla', 'Vex', 'Wren', 'Yara', 'Zed'];
  var ARCH = [
    { key: 'hunter',   label: 'Hunter',   score: function (g, p) { return g.strength + g.speed * 0.5 + p[0] * 2 + p[4] * 1.5; } },
    { key: 'gatherer', label: 'Gatherer', score: function (g, p) { return g.metabolism * 0.6 + p[3] * 2 + p[5] * 1.5 + p[7]; } },
    { key: 'crafter',  label: 'Crafter',  score: function (g, p) { return (g.intel || 1) + p[5] * 2.5 + p[3]; } },
    { key: 'scout',    label: 'Scout',    score: function (g, p) { return g.speed * 1.4 + p[2] * 2 + p[0]; } },
    { key: 'guardian', label: 'Guardian', score: function (g, p) { return g.resist + p[0] * 2 + p[6] * 1.5; } },
    { key: 'healer',   label: 'Healer',   score: function (g, p) { return (g.intel || 1) * 0.8 + p[6] * 2.5 + p[3] * 1.2; } }
  ];
  var TRAITS = [
    { key: 'tough',    label: 'Tough',          desc: 'Takes 30% less injury.',                     fx: { injK: 0.7 }, vs: ['fragile'] },
    { key: 'fragile',  label: 'Fragile',        desc: 'Takes 30% more injury.',                     fx: { injK: 1.3 }, vs: ['tough'] },
    { key: 'glutton',  label: 'Glutton',        desc: 'Burns food 30% faster.',                     fx: { satK: 1.3 }, vs: ['ascetic'] },
    { key: 'ascetic',  label: 'Ascetic',        desc: 'Needs 20% less food and 10% less water.',    fx: { satK: 0.8, hydK: 0.9 }, vs: ['glutton'] },
    { key: 'quick',    label: 'Fast learner',   desc: 'Trains 50% faster.',                         fx: { learnK: 1.5 }, vs: ['slow'] },
    { key: 'slow',     label: 'Slow learner',   desc: 'Trains 30% slower.',                         fx: { learnK: 0.7 }, vs: ['quick'] },
    { key: 'neurotic', label: 'Neurotic',       desc: 'Stress lingers 50% longer.',                 fx: { stressK: 1.5 }, vs: ['steady'] },
    { key: 'steady',   label: 'Steady',         desc: 'Stress fades 60% faster.',                   fx: { stressK: 0.6 }, vs: ['neurotic'] },
    { key: 'brave',    label: 'Brave',          desc: '+30% bravery.',                              pers: [0, 0.3], vs: ['coward'] },
    { key: 'coward',   label: 'Coward',         desc: '-30% bravery.',                              pers: [0, -0.3], vs: ['brave'] },
    { key: 'sociable', label: 'Sociable',       desc: '+30% sociability.',                          pers: [1, 0.3], vs: ['loner'] },
    { key: 'loner',    label: 'Loner',          desc: '-30% sociability.',                          pers: [1, -0.3], vs: ['sociable'] },
    { key: 'curious',  label: 'Curious',        desc: '+30% curiosity. Finds new technology faster.', pers: [2, 0.3], vs: [] },
    { key: 'patient',  label: 'Patient',        desc: '+30% patience.',                             pers: [3, 0.3], vs: ['hothead'] },
    { key: 'hothead',  label: 'Hothead',        desc: '+30% aggression.',            pers: [4, 0.3], vs: ['patient', 'kind'] },
    { key: 'industrious', label: 'Industrious', desc: '+30% diligence.',                            pers: [5, 0.3], vs: ['lazy'] },
    { key: 'lazy',     label: 'Lazy',           desc: '-30% diligence.',                            pers: [5, -0.3], vs: ['industrious'] },
    { key: 'kind',     label: 'Kind',           desc: '+30% empathy.',                              pers: [6, 0.3], vs: ['cruel', 'hothead'] },
    { key: 'cruel',    label: 'Cruel',          desc: '-30% empathy.',                              pers: [6, -0.3], vs: ['kind'] },
    { key: 'greedy',   label: 'Greedy',         desc: '+30% greed.',              pers: [7, 0.3], vs: [] }
  ];
  function traitFx(a, k) { var v = 1; (a.traits || []).forEach(function (t) { for (var i = 0; i < TRAITS.length; i++) if (TRAITS[i].key === t && TRAITS[i].fx && TRAITS[i].fx[k] !== undefined) v *= TRAITS[i].fx[k]; }); return v; }
  function archetypesOf(g, pers) { return ARCH.map(function (r) { return { label: r.label, s: r.score(g, pers) }; }).sort(function (a, b) { return b.s - a.s; }).slice(0, 2).map(function (r) { return r.label; }); }
  function strengthsFlaws(g, p) {
    var S = [], F = [], intel = g.intel || 1;
    function two(v, hi, lo, hiL, loL) { if (v >= hi) S.push(hiL); else if (v <= lo) F.push(loL); }
    two(g.strength, 1.3, 0.75, 'Strong', 'Weak'); two(g.speed, 1.3, 0.75, 'Swift', 'Slow'); two(g.resist, 1.3, 0.7, 'Resilient', 'Susceptible');
    two(g.insulation, 1.3, 0.75, 'Cold-proof', 'Chills easily'); two(g.metabolism, 0.8, 0, 'Efficient', 'x'); two(1 / g.metabolism, 1.3, 0, 'x', 'Ravenous'); two(g.lifespan, 1.3, 0.75, 'Long-lived', 'Short-lived'); two(intel, 1.4, 0.7, 'Sharp', 'Dull');
    two(p[5], 0.75, 0.25, 'Diligent', 'Idle-minded'); two(p[0], 0.75, 0.25, 'Fearless', 'Timid'); two(p[3], 0.75, 0.25, 'Patient', 'Impulsive'); two(p[6], 0.75, 0.25, 'Compassionate', 'Callous');
    return { strengths: S.filter(function (x) { return x !== 'x'; }).slice(0, 4), flaws: F.filter(function (x) { return x !== 'x'; }).slice(0, 4) };
  }

  /* ---------- schematics: primitives named by function, discoverable ----------
     tier 0 = foundations, 1 and 2 build on them. In "preset" mode everything is known; in "emergent" mode agents discover them, rolled against intellect. */
  var SCHEM = {
    'weapon.club':     { tier: 0, diff: 1, req: [] },
    'weapon.knife':    { tier: 0, diff: 1, req: [] },
    'clothing.tunic':  { tier: 0, diff: 1, req: [] },
    'weapon.spear':    { tier: 1, diff: 2, req: ['weapon.knife'] },
    'weapon.axe':      { tier: 1, diff: 2, req: ['weapon.club', 'weapon.knife'] },
    'armor.vest':      { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'armor.helm':      { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'clothing.cloak':  { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'clothing.hat':    { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'clothing.gloves': { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'clothing.boots':  { tier: 1, diff: 2, req: ['clothing.tunic'] },
    'weapon.bow':      { tier: 2, diff: 3, req: ['weapon.spear'] },
    'armor.shield':    { tier: 2, diff: 3, req: ['armor.vest'] },
    'armor.greaves':   { tier: 2, diff: 3, req: ['armor.vest'] }
  };
  function discoverable(known) { return Object.keys(SCHEM).filter(function (k) { return known.indexOf(k) < 0 && SCHEM[k].req.every(function (r) { return known.indexOf(r) >= 0; }); }); }
  function discoverChance(key, intel, curiosity, k) { return Math.min(0.5, 0.02 * intel * (0.5 + curiosity) * (k === undefined ? 1 : k) / SCHEM[key].diff); }

  /* ---------- machinery schema (data only for now): mechanisms assembled from components, unlocked by intellect and known schematics ---------- */
  var MACHINES = [
    { key: 'lever',   label: 'Lever frame',     tier: 1, power: 'hand',   parts: ['Fulcrum', 'Beam'],                  needs: 14, effect: 'Crafting speed x1.2' },
    { key: 'roller',  label: 'Roller sled',     tier: 1, power: 'hand',   parts: ['Rollers', 'Deck', 'Cord'],           needs: 14, effect: 'Haul x2 per trip' },
    { key: 'spindle', label: 'Spindle',         tier: 1, power: 'hand',   parts: ['Shaft', 'Whorl', 'Fiber'],           needs: 16, effect: 'Cord from fiber, x2' },
    { key: 'press',   label: 'Screw press',     tier: 2, power: 'hand',   parts: ['Screw', 'Plate', 'Frame'],           needs: 20, effect: 'Extra yield from hide and berries' },
    { key: 'wheel',   label: 'Drive wheel',     tier: 2, power: 'animal', parts: ['Wheel', 'Axle', 'Frame'],            needs: 22, effect: 'Powers one machine per wheel' },
    { key: 'gears',   label: 'Gear train',      tier: 2, power: 'any',    parts: ['Gears', 'Axle', 'Housing'],          needs: 26, effect: 'Links machines, trades speed for force' },
    { key: 'kiln',    label: 'Kiln',            tier: 2, power: 'fuel',   parts: ['Chamber', 'Flue', 'Fuel'],           needs: 24, effect: 'Fired clay and tougher parts' },
    { key: 'loom',    label: 'Loom',            tier: 2, power: 'hand',   parts: ['Frame', 'Heddle', 'Fiber'],          needs: 26, effect: 'Woven wraps, warmth x1.3' },
    { key: 'millwheel', label: 'Water wheel',   tier: 3, power: 'water',  parts: ['Wheel', 'Axle', 'Race'],             needs: 32, effect: 'Free power beside water' },
    { key: 'engine',  label: 'Heat engine',     tier: 3, power: 'fuel',   parts: ['Boiler', 'Piston', 'Gears', 'Flue'], needs: 44, effect: 'Powers a whole workshop' }
  ];


  /* ---------- genes: 18 multipliers around 1.0 in four groups. "on" genes already change the simulation, the rest are latent until a system reads them ---------- */
  var GENES = [
    { key: 'frame', label: 'Frame size', group: 'Frame', on: 1, drives: 'Height and body mass' }, { key: 'muscle', label: 'Muscle', group: 'Frame', on: 0, drives: 'Carry capacity, strike force (latent)' },
    { key: 'boneDensity', label: 'Bone density', group: 'Frame', on: 1, fx: 'injK', drives: 'Injury taken' }, { key: 'fatStore', label: 'Fat reserve', group: 'Frame', on: 1, fx: 'satK', drives: 'Slower hunger' },
    { key: 'strength', label: 'Strength', group: 'Body', on: 1, drives: 'Damage, hauling, build' }, { key: 'speed', label: 'Speed', group: 'Body', on: 1, drives: 'Movement' },
    { key: 'resist', label: 'Resistance', group: 'Body', on: 1, drives: 'Infection recovery, venom' }, { key: 'insulation', label: 'Insulation', group: 'Body', on: 1, drives: 'Cold and heat' },
    { key: 'metabolism', label: 'Metabolism', group: 'Body', on: 1, fx: 'satK', drives: 'Food burn' }, { key: 'lifespan', label: 'Lifespan', group: 'Body', on: 1, drives: 'Years until old age' },
    { key: 'stamina', label: 'Stamina', group: 'Body', on: 1, drives: 'Energy drain' }, { key: 'lungs', label: 'Lung capacity', group: 'Body', on: 1, drives: 'Energy drain' },
    { key: 'hydration', label: 'Water economy', group: 'Body', on: 1, fx: 'hydK', drives: 'Thirst rate' }, { key: 'healing', label: 'Healing', group: 'Body', on: 1, drives: 'Health and injury recovery' },
    { key: 'painTol', label: 'Pain tolerance', group: 'Body', on: 1, fx: 'injK', drives: 'Injury taken' }, { key: 'immunity', label: 'Immunity', group: 'Body', on: 1, drives: 'Infection chance and recovery' },
    { key: 'clotting', label: 'Clotting', group: 'Body', on: 1, drives: 'How fast a bleeding wound seals' }, { key: 'regen', label: 'Regeneration', group: 'Body', on: 1, drives: 'Wound repair speed, on top of general healing' },
    { key: 'shock', label: 'Shock resistance', group: 'Body', on: 1, drives: 'Consciousness lost to pain' }, { key: 'scarring', label: 'Scar resistance', group: 'Body', on: 1, drives: 'Chance a healed wound leaves a scar' },
    { key: 'vision', label: 'Vision', group: 'Senses', on: 0, drives: 'Spotting prey and threats (latent)' }, { key: 'hearing', label: 'Hearing', group: 'Senses', on: 0, drives: 'Alertness (latent)' }, { key: 'smell', label: 'Smell', group: 'Senses', on: 0, drives: 'Food and predators (latent)' },
    { key: 'intel', label: 'Intellect', group: 'Mind', on: 1, fx: 'learnK', drives: 'Schematic discovery, learning' }, { key: 'memory', label: 'Memory', group: 'Mind', on: 0, drives: 'Remembering places (latent)' }, { key: 'focus', label: 'Focus', group: 'Mind', on: 0, fx: 'learnK', drives: 'Training speed (latent)' },
    { key: 'fertility', label: 'Fertility', group: 'Life', on: 0, drives: 'Births (latent)' }, { key: 'growth', label: 'Growth rate', group: 'Life', on: 0, drives: 'Childhood length (latent)' }
  ];
  var GENE_GROUPS = { Frame: '#e0b46a', Body: '#d4c63c', Senses: '#7cacdf', Mind: '#b49ae0', Life: '#e0956a', Temperament: '#e08aa8' };
  function gv(g, k) { return g && g[k] !== undefined ? g[k] : 1; }


  /* ---------- technology: a prefabbed tree of primitives. Nothing historic is hard-coded as a finished item; each node unlocks schematics (the
     function-named primitives), machines and small global effects. Preset mode knows every node; Emergent mode researches them with insight. ---------- */
  var TECH = [
    { key: 'shaping',   label: 'Shaping',      tier: 0, cost: 4,   req: [], schem: ['weapon.club', 'weapon.knife'], machines: [], fx: {}, text: 'Work stone, wood and bone by striking and scraping.' },
    { key: 'binding',   label: 'Binding',      tier: 0, cost: 4,   req: [], schem: ['clothing.tunic'], machines: [], fx: {}, text: 'Tie and wrap with fiber, sinew and hide.' },
    { key: 'hafting',   label: 'Hafting',      tier: 1, cost: 10,  req: ['shaping', 'binding'], schem: ['weapon.spear', 'weapon.axe'], machines: [], fx: { skill: 0.5 }, text: 'Fix a working head to a handle.' },
    { key: 'tanning',   label: 'Tanning',      tier: 1, cost: 10,  req: ['binding'], schem: ['armor.vest', 'armor.helm', 'clothing.cloak', 'clothing.hat', 'clothing.gloves', 'clothing.boots'], machines: [], fx: { skill: 0.5 }, text: 'Cure hide and fur so they last and shape.' },
    { key: 'spearfishing', label: 'Spear fishing', tier: 2, cost: 26, req: ['hafting'], schem: [], machines: [], fx: { fish: 1 }, text: 'Stand braced at the bank and strike down at a fish with a hafted spear. Needs a spear in hand. Takes the big fish (trout, pike, carp, catfish, eel, salmon, sturgeon) that bare hands cannot.' },
    { key: 'polefishing', label: 'Pole fishing', tier: 2, cost: 34, req: ['spearfishing', 'spinning'], schem: [], machines: [], fx: { fish: 2 }, text: 'A long wooden pole, a cord line and a bone or claw hook: wait for a bite instead of striking. Slower than the spear but it takes perch, trout and carp without spooking the shoal. Needs a pole (wood), cord (fiber) and a hook (bone, claw or horn) in the store; the line wears out now and then.' },
    { key: 'leverage',  label: 'Leverage',     tier: 1, cost: 12,  req: ['shaping'], schem: [], machines: ['lever', 'roller'], fx: { carry: 2 }, text: 'Beam on a fulcrum, logs under a load.' },
    { key: 'spinning',  label: 'Spinning',     tier: 1, cost: 12,  req: ['binding'], schem: [], machines: ['spindle'], fx: {}, text: 'Twist fiber into cord.' },
    { key: 'tension',   label: 'Tension',      tier: 2, cost: 42,  req: ['hafting', 'spinning'], schem: ['weapon.bow'], machines: [], fx: {}, text: 'Store force in a flexed limb.' },
    { key: 'shielding', label: 'Shielding',    tier: 2, cost: 40,  req: ['tanning', 'hafting'], schem: ['armor.shield', 'armor.greaves'], machines: [], fx: { skill: 0.5 }, text: 'Layered cover that turns blows.' },
    { key: 'weaving',   label: 'Weaving',      tier: 2, cost: 44,  req: ['spinning', 'tanning'], schem: [], machines: ['loom'], fx: {}, text: 'Interlace cord into cloth.' },
    { key: 'metalwork', label: 'Cold metalwork', tier: 2, cost: 40, req: ['shaping', 'firecraft'], schem: [], machines: [], fx: { skill: 0.3 }, text: 'Hammer native copper and gold into sheet and edge; heat and quench to keep it workable.' },
    { key: 'kilning',   label: 'Kilning',      tier: 2, cost: 46,  req: ['hafting', 'leverage'], schem: [], machines: ['kiln'], fx: { skill: 0.5 }, text: 'Contain fire to harden clay and ore.' },
    { key: 'wheelwork', label: 'Wheelwork',    tier: 2, cost: 52,  req: ['leverage', 'spinning'], schem: [], machines: ['wheel', 'press', 'gears'], fx: { carry: 2 }, text: 'Rotating parts and screw threads.' },
    { key: 'wells',     label: 'Wells',        tier: 1, cost: 10,  req: ['shaping'], schem: [], machines: [], fx: {}, text: 'Break ground and lay stone round a seep: a well in camp, so no one walks to the stream to drink or fill a skin.' },
    { key: 'adornment', label: 'Adornment',   tier: 0, cost: 7,  req: [], schem: [], machines: [], fx: {}, text: 'Thread shell, bone, native copper and gold into beads and pendants: marks of kin, rank and luck. (Soft gold is no tool and no weapon; it is for wearing.) No mechanical effect yet.' },
    { key: 'markmaking', label: 'Marks and tallies', tier: 2, cost: 30, req: ['firecraft'], schem: [], machines: [], fx: {}, text: 'Charcoal, ochre and cuts in bone, wood and stone keep a record outside any one head: a colonist can set down what they know, and others can read it back after the maker is gone.' },
    { key: 'shelter',   label: 'Shelter',      tier: 1, cost: 10,  req: [], schem: [], machines: [], fx: {}, text: 'Stack stone, set posts and lash roofs: walls, floors, doors and windows. Until known there is no house; people sleep under the sky.' },
    { key: 'bedding',   label: 'Bedding',      tier: 1, cost: 7,  req: [], schem: [], machines: [], fx: {}, text: 'Pile fiber, hide and fur into a bed: sleep off the cold ground. Until known no one has a bed.' },
    { key: 'storage',   label: 'Storage',      tier: 1, cost: 8,  req: [], schem: [], machines: [], fx: {}, text: 'Woven baskets, lashed crates and hide sacks: crates, barrels and shelves for the camp store.' },
    { key: 'carpentry', label: 'Carpentry',    tier: 2, cost: 30,  req: ['shaping'], schem: [], machines: [], fx: {}, text: 'Split, trim and join wood: a table, stools and a workbench in camp.' },
    { key: 'firecraft', label: 'Firecraft',    tier: 0, cost: 4,   req: [], schem: [], machines: [], fx: {}, text: 'Strike a spark and keep a flame: the hearth warms everyone within 3 tiles (+10 degrees). Without it the hearth is only stones.' },
    { key: 'cooking',   label: 'Cooking',      tier: 1, cost: 9,  req: ['firecraft'], schem: [], machines: [], fx: {}, text: 'Heat food over the fire: a meal at the table feeds 12 points more and cannot spoil.' },
    { key: 'torchcraft', label: 'Torchcraft',  tier: 0, cost: 4,   req: ['firecraft'], schem: [], machines: [], fx: {}, text: 'Wrap fiber round a wooden shaft: a torch burns about 90 ticks and lights 4 tiles of cave. Without one, caves are dark.' },
    { key: 'footwork',  label: 'Footwork',     tier: 1, cost: 10,  req: ['shaping'], schem: [], machines: [], moves: ['side', 'feint', 'taunt'], fx: {}, text: 'Balance, spacing and timing: sidesteps, feints, kiting.' },
    { key: 'climbing',  label: 'Climbing',     tier: 2, cost: 36,  req: ['spinning', 'footwork'], schem: [], machines: [], fx: {}, text: 'Cord, pegs and hand-holds: with a tool on the belt, agonists can scale sheer faces and cross steep scree.' },
    { key: 'tactics',   label: 'Tactics',      tier: 2, cost: 48,  req: ['footwork', 'hafting'], schem: [], machines: [], moves: ['guard'], fx: {}, text: 'Fight as a group: guard and counter, shield work, covering allies.' },
    { key: 'herbalism', label: 'Herbalism',  tier: 1, cost: 10,  req: ['binding'], schem: [], machines: [], fx: { tend: 0.3 }, text: 'Know which leaves stop bleeding and which ease pain: tended wounds heal 30% faster.' },
    { key: 'cultivation', label: 'Cultivation', tier: 1, cost: 12, req: ['shaping', 'binding'], schem: [], machines: [], fx: { crop: 0.35 }, text: 'Dig, seed and weed in rows: crop plots give 35% more.' },
    { key: 'curing',    label: 'Curing',       tier: 2, cost: 40,  req: ['tanning', 'herbalism'], schem: [], machines: [], fx: { keep: 0.4 }, text: 'Smoke, salt and dry: stored food spoils 40% slower.' },
    { key: 'medicine',  label: 'Medicine',     tier: 3, cost: 70,  req: ['herbalism', 'kilning'], schem: [], machines: [], fx: { tend: 0.4 }, text: 'Boiled dressings, cauterising and set bones: tended wounds heal a further 40% faster.' },
    { key: 'waterpower', label: 'Water power', tier: 3, cost: 90,  req: ['wheelwork'], schem: [], machines: ['millwheel'], fx: {}, text: 'Let a stream turn the wheel.' },
    { key: 'heatengine', label: 'Heat engine', tier: 3, cost: 120, req: ['kilning', 'wheelwork'], schem: [], machines: ['engine'], fx: { skill: 1 }, text: 'Turn fire into steady motion.' }
  ];
  /* ---------- action system: PRIMITIVES (one tick each) -> MOVES (sequences of primitives) -> STYLES (weighted move preferences).
     Moves are gated by weapon template, armour piece and technology, so new techniques are data: add a primitive, add a move, add it to a style. ---------- */
  var PRIM = {
    step:   { cat: 'move', label: 'Step',     cost: 0.0004, text: 'One tile toward the target.' },
    dash:   { cat: 'move', label: 'Dash',     cost: 0.0016, steps: 2, text: 'Two tiles at once; tiring.' },
    side:   { cat: 'move', label: 'Sidestep', cost: 0.0006, lateral: 1, evade: 0.35, text: 'Slip sideways; harder to hit this tick.' },
    back:   { cat: 'move', label: 'Back off', cost: 0.0005, away: 1, text: 'Retreat one tile.' },
    wind:   { cat: 'act',  label: 'Wind up',  cost: 0.0006, mod: { dmg: 0.5, acc: -0.1 }, text: 'Load the next blow: more damage, less accurate.' },
    feint:  { cat: 'act',  label: 'Feint',    cost: 0.0005, mod: { acc: 0.2 }, text: 'Fake one; the next blow lands easier.' },
    guard:  { cat: 'act',  label: 'Guard',    cost: 0.0004, block: 0.5, text: 'Brace: halves damage taken this tick and the next.' },
    taunt:  { cat: 'act',  label: 'Taunt',    cost: 0, tauntFx: 0.15, text: 'Provoke: the target\'s anger rises.' },
    jab:    { cat: 'hit',  label: 'Jab',      cost: 0.001, dmg: 0.7, acc: 0.6, text: 'Quick short blow.' },
    swing:  { cat: 'hit',  label: 'Swing',    cost: 0.0014, dmg: 1, acc: 0.55, text: 'Wide arc.' },
    thrust: { cat: 'hit',  label: 'Thrust',   cost: 0.0012, dmg: 1, acc: 0.6, reach: 1, pierce: 0.3, text: 'Straight point; +1 reach, ignores 30% armour.' },
    slash:  { cat: 'hit',  label: 'Slash',    cost: 0.0012, dmg: 0.8, acc: 0.6, bleed: 0.5, text: 'Cutting edge: extra injury.' },
    bash:   { cat: 'hit',  label: 'Bash',     cost: 0.0016, dmg: 1.1, acc: 0.5, stagger: 0.4, text: 'Heavy blunt blow; may stagger.' },
    shove:  { cat: 'hit',  label: 'Shove',    cost: 0.001, dmg: 0.15, acc: 0.7, knock: 1, text: 'Push the target one tile back.' },
    grapple:{ cat: 'hit',  label: 'Grapple',  cost: 0.0014, dmg: 0.3, acc: 0.6, hold: 2, text: 'Seize: target cannot move for 2 ticks.' },
    shoot:  { cat: 'hit',  label: 'Shoot',    cost: 0.0012, dmg: 1, acc: 0.5, ranged: 1, text: 'Launch from up to 4 tiles.' },
    throw:  { cat: 'hit',  label: 'Throw',    cost: 0.0011, dmg: 0.8, acc: 0.45, ranged: 1, thrown: 1, range: 3, text: 'Hurl a carried stone from up to 3 tiles. The stone is spent. Runs through the interaction layer as hit(use(throw(rock)), on(person)).' }
  };
  var MOVES = {
    punch:    { label: 'Punch',        seq: ['jab'],                 req: {} },
    haymaker: { label: 'Haymaker',     seq: ['wind', 'swing'],       req: {} },
    shove:    { label: 'Shove',        seq: ['shove'],               req: {} },
    tackle:   { label: 'Tackle',       seq: ['dash', 'grapple'],     req: {} },
    smash:    { label: 'Smash',        seq: ['wind', 'bash'],        req: { weapon: ['club', 'axe'] } },
    chop:     { label: 'Chop',         seq: ['swing'],               req: { weapon: ['axe', 'club'] } },
    poke:     { label: 'Poke',         seq: ['thrust'],              req: { weapon: ['spear'] } },
    lunge:    { label: 'Lunge',        seq: ['dash', 'thrust'],      req: { weapon: ['spear'], tech: 'hafting' } },
    cut:      { label: 'Cut',          seq: ['slash'],               req: { weapon: ['knife', 'axe'] } },
    feintcut: { label: 'Feint and cut', seq: ['feint', 'slash'],     req: { weapon: ['knife', 'axe'], tech: 'footwork' } },
    skirmish: { label: 'Slip and jab', seq: ['side', 'jab'],         req: { tech: 'footwork' } },
    taunting: { label: 'Bait',         seq: ['taunt', 'side'],       req: { tech: 'footwork' } },
    volley:   { label: 'Shoot',        seq: ['shoot'],               req: { weapon: ['bow'] } },
    kite:     { label: 'Shoot and fall back', seq: ['shoot', 'back'], req: { weapon: ['bow'], tech: 'footwork' } },
    bulwark:  { label: 'Shield bash',  seq: ['guard', 'bash'],       req: { armor: ['shield'], tech: 'tactics' } },
    counter:  { label: 'Guard and counter', seq: ['guard', 'swing'], req: { tech: 'tactics' } },
    rockthrow: { label: 'Throw a rock', seq: ['throw'],            req: { stone: 1 } }
  };
  var STYLES = {
    brawler:    { label: 'Brawler',    w: { rockthrow: 0.4, punch: 1, haymaker: 1.2, shove: 0.6, tackle: 0.9, smash: 1.4, chop: 1 } },
    duelist:    { label: 'Duelist',    w: { rockthrow: 0.3, poke: 1.2, lunge: 1.2, cut: 1, feintcut: 1.4, skirmish: 0.8, haymaker: 0.3 } },
    skirmisher: { label: 'Skirmisher', w: { rockthrow: 1.1, volley: 1.2, kite: 1.6, skirmish: 1, taunting: 0.6, cut: 0.5 } },
    guardian:   { label: 'Guardian',   w: { bulwark: 1.5, counter: 1.4, smash: 0.8, poke: 0.6, shove: 0.8 } }
  };
  function moveOK(k, ctx) {
    var m = MOVES[k], q = m.req || {};
    if (q.weapon && q.weapon.indexOf(ctx.tpl) < 0) return false;
    if (q.armor && q.armor.indexOf(ctx.armor) < 0) return false;
    if (q.tech && ctx.techs.indexOf(q.tech) < 0) return false;
    if (q.stone && !(ctx.stone > 0)) return false;
    if (m.seq.some(function (p) { return PRIM[p].ranged && !PRIM[p].thrown && ctx.tpl !== 'bow'; })) return false;
    return m.seq.every(function (p) { return !tkNeed(p) || ctx.techs.indexOf(tkNeed(p)) >= 0; });
  }
  function tkNeed(p) { for (var i = 0; i < TECH.length; i++) if (TECH[i].moves && TECH[i].moves.indexOf(p) >= 0) return TECH[i].key; return null; }
  function styleOf(pers, gear) {
    var wt = gear && gear.weapon ? gear.weapon.tplKey : null, ar = gear && gear.armor ? gear.armor.tplKey : null;
    if (wt === 'bow') return 'skirmisher';
    if (ar === 'shield' || (ar && pers[0] > 0.65 && pers[3] > 0.5)) return 'guardian';
    if (wt === 'spear' || wt === 'knife') return 'duelist';
    return 'brawler';
  }
  function pickMove(style, ctx, rnd) {
    var st = STYLES[style], ks = Object.keys(MOVES).filter(function (k) { return moveOK(k, ctx); }), tot = 0, ws = ks.map(function (k) { var w = st.w[k] || (k === 'punch' || k === 'shove' ? 0.25 : 0.05); tot += w; return w; });
    var x = rnd() * tot; for (var i = 0; i < ks.length; i++) { x -= ws[i]; if (x <= 0) return ks[i]; } return ks[0] || 'punch';
  }
  function moveStats(k) {
    var m = MOVES[k], mod = 0, dmg = 0, acc = 0, n = 0, reach = 0, cost = 0, fx = [];
    m.seq.forEach(function (pk) { var P = PRIM[pk]; cost += P.cost; if (P.mod && P.mod.dmg) mod += P.mod.dmg; if (P.cat === 'hit') { dmg += P.dmg; acc += P.acc; n++; reach = Math.max(reach, P.thrown ? P.range : P.ranged ? 4 : (P.reach || 0)); ['pierce', 'bleed', 'stagger', 'knock', 'hold'].forEach(function (f) { if (P[f]) fx.push(f); }); } if (P.block) fx.push('block'); if (P.evade) fx.push('evade'); });
    return { ticks: m.seq.length, dmg: Math.round(dmg * (1 + mod) * 100) / 100, acc: n ? Math.round(acc / n * 100) / 100 : 0, reach: reach, cost: Math.round(cost * 10000) / 10000, fx: fx };
  }
  /* ---------- universal interaction layer ----------
     Every physical thing (person, animal, rock, tree, wall, fire, tool, container...) is a THING: tags + numeric props (0..1), the verbs it can
     be USED for (use) and the verbs that can be done TO it (on). A VERB is a primitive that reads the props of its use-operand and on-operand and
     returns effects. Operands may themselves be primitives, so composites nest: prim_z = [verb, { use: X, on: Y }], e.g.
     hit(use(throw(rock)), on(person)). evalNode() runs a tree bottom-up. Props are 0..1 unless noted; mass 1 is roughly one carried person-load. */
  var THINGS = {
    person:   { label: 'Person',  tags: ['living', 'solid', 'warm'],        p: { hard: 0.25, mass: 1,   sharp: 0, flam: 0.3, integ: 1 }, use: ['hit', 'shove', 'carry', 'throw', 'open', 'climb', 'talk', 'heal', 'feed', 'light', 'craft', 'drink', 'eat'], on: ['hit', 'cut', 'pierce', 'shove', 'carry', 'burn', 'talk', 'heal', 'feed', 'grapple'] },
    prey:     { label: 'Prey animal', tags: ['living', 'solid', 'edible'],  p: { hard: 0.2, mass: 0.6, sharp: 0, flam: 0.3, integ: 0.6, food: 0.8 }, use: ['hit'], on: ['hit', 'cut', 'pierce', 'burn', 'eat', 'carry', 'feed', 'tame'] },
    predator: { label: 'Predator',  tags: ['living', 'solid'],              p: { hard: 0.3, mass: 0.9, sharp: 0.5, flam: 0.3, integ: 0.9 }, use: ['hit'], on: ['hit', 'cut', 'pierce', 'burn', 'feed', 'tame', 'scare'] },
    rock:     { label: 'Rock',    tags: ['solid', 'stone', 'hard'],         p: { hard: 0.85, mass: 0.5, sharp: 0.2, flam: 0, integ: 1 }, use: ['hit', 'throw', 'carry', 'block', 'sit', 'craft'], on: ['hit', 'carry', 'throw', 'climb', 'shove'] },
    boulder:  { label: 'Boulder', tags: ['solid', 'stone', 'hard', 'heavy'], p: { hard: 0.95, mass: 3,  sharp: 0, flam: 0, integ: 1 }, use: ['block', 'sit', 'climb'], on: ['hit', 'climb', 'shove'] },
    flint:    { label: 'Flint',   tags: ['solid', 'stone', 'hard', 'sharp'], p: { hard: 0.95, mass: 0.3, sharp: 0.85, flam: 0, integ: 0.8 }, use: ['cut', 'hit', 'throw', 'light', 'craft'], on: ['carry', 'hit', 'craft'] },
    stick:    { label: 'Stick',   tags: ['solid', 'wood', 'flammable'],     p: { hard: 0.4, mass: 0.3, sharp: 0, flam: 0.9, integ: 0.5 }, use: ['hit', 'light', 'carry', 'craft', 'climb', 'throw'], on: ['hit', 'burn', 'carry', 'craft', 'break'] },
    tree:     { label: 'Tree',    tags: ['solid', 'wood', 'flammable', 'plant'], p: { hard: 0.5, mass: 3, sharp: 0, flam: 0.6, integ: 2 }, use: ['climb', 'block', 'shelter'], on: ['hit', 'cut', 'burn', 'climb'], yields: { cut: [['wood', 3]], burn: [['ash', 1]] } },
    bush:     { label: 'Berry bush', tags: ['solid', 'plant', 'flammable', 'edible'], p: { hard: 0.1, mass: 0.6, sharp: 0, flam: 0.8, integ: 0.4, food: 0.6 }, use: ['block'], on: ['gather', 'cut', 'burn', 'hit'], yields: { gather: [['berries', 1]], cut: [['fiber', 1]] } },
    water:    { label: 'Water',   tags: ['liquid'],                         p: { hard: 0, mass: 0, sharp: 0, flam: 0 }, use: ['douse', 'drink'], on: ['drink', 'fill', 'douse', 'cross'] },
    fire:     { label: 'Fire',    tags: ['hot', 'flame'],                   p: { hard: 0, mass: 0, sharp: 0, flam: 0, heat: 1 }, use: ['burn', 'light', 'warm', 'cook', 'scare'], on: ['douse', 'feed'] },
    wall:     { label: 'Wall',    tags: ['solid', 'structure', 'wood'],     p: { hard: 0.7, mass: 5, sharp: 0, flam: 0.4, integ: 3 }, use: ['block', 'shelter', 'climb'], on: ['hit', 'burn', 'break'] },
    door:     { label: 'Door',    tags: ['solid', 'structure', 'wood', 'portal'], p: { hard: 0.5, mass: 2, sharp: 0, flam: 0.6, integ: 1.5 }, use: ['block'], on: ['open', 'hit', 'burn', 'break'] },
    window:   { label: 'Window',  tags: ['structure', 'portal'],            p: { hard: 0.3, mass: 1, sharp: 0, flam: 0.1, integ: 0.5 }, use: ['look'], on: ['hit', 'break', 'open'] },
    table:    { label: 'Table',   tags: ['solid', 'wood', 'furniture', 'container'], p: { hard: 0.5, mass: 2, sharp: 0, flam: 0.6, integ: 1.2, cap: 10 }, use: ['store', 'eat', 'craft', 'block'], on: ['hit', 'burn', 'break', 'store', 'climb'] },
    bench:    { label: 'Workbench', tags: ['solid', 'wood', 'furniture'],   p: { hard: 0.6, mass: 2.5, sharp: 0, flam: 0.5, integ: 1.4 }, use: ['craft', 'research', 'sit'], on: ['hit', 'burn', 'break'] },
    stool:    { label: 'Stool',   tags: ['solid', 'wood', 'furniture'],     p: { hard: 0.4, mass: 0.6, sharp: 0, flam: 0.6, integ: 0.5 }, use: ['sit', 'climb', 'hit', 'throw'], on: ['hit', 'burn', 'break', 'carry', 'shove'] },
    bed:      { label: 'Bed',     tags: ['solid', 'furniture', 'soft'],     p: { hard: 0.2, mass: 2, sharp: 0, flam: 0.7, integ: 0.8, warm: 0.5 }, use: ['sleep', 'sit'], on: ['hit', 'burn', 'break'] },
    barrel:   { label: 'Barrel',  tags: ['solid', 'wood', 'container', 'liquid-safe'], p: { hard: 0.5, mass: 1.5, sharp: 0, flam: 0.6, integ: 0.9, cap: 20 }, use: ['store', 'block', 'sit'], on: ['hit', 'burn', 'break', 'fill', 'store', 'carry'] },
    crate:    { label: 'Crate',   tags: ['solid', 'wood', 'container'],     p: { hard: 0.5, mass: 1.2, sharp: 0, flam: 0.6, integ: 0.8, cap: 12 }, use: ['store', 'block', 'sit', 'climb'], on: ['hit', 'burn', 'break', 'open', 'store', 'carry'] },
    shelf:    { label: 'Shelf',   tags: ['solid', 'wood', 'furniture', 'container'], p: { hard: 0.4, mass: 1.5, sharp: 0, flam: 0.6, integ: 0.8, cap: 16 }, use: ['store'], on: ['hit', 'burn', 'break', 'store'] },
    rug:      { label: 'Rug',     tags: ['soft', 'furniture', 'flammable'], p: { hard: 0, mass: 0.5, sharp: 0, flam: 0.8, integ: 0.4, warm: 0.4 }, use: ['sit', 'warm'], on: ['burn', 'carry', 'cut'] },
    cave:     { label: 'Cave',    tags: ['structure', 'stone', 'shelter'],  p: { hard: 1, mass: 9, sharp: 0, flam: 0, integ: 9 }, use: ['shelter', 'sleep', 'hide'], on: ['enter'] },
    hearth:   { label: 'Hearth',  tags: ['structure', 'stone', 'hot'],      p: { hard: 0.8, mass: 3, sharp: 0, flam: 0, integ: 3, heat: 0.8 }, use: ['warm', 'cook', 'light'], on: ['feed', 'douse'] },
    club:     { label: 'Striker', tags: ['solid', 'wood', 'tool', 'weapon', 'blunt'],  p: { hard: 0.6, mass: 0.6, sharp: 0, flam: 0.5, integ: 0.5, dmg: 3 }, use: ['hit', 'shove', 'block'], on: ['carry', 'break', 'craft', 'burn'] },
    spear:    { label: 'Reacher', tags: ['solid', 'wood', 'tool', 'weapon', 'sharp', 'long'], p: { hard: 0.6, mass: 0.5, sharp: 0.7, flam: 0.4, integ: 0.4, dmg: 3, reach: 2 }, use: ['pierce', 'hit', 'throw', 'block'], on: ['carry', 'break', 'craft', 'burn'] },
    axe:      { label: 'Splitter', tags: ['solid', 'tool', 'weapon', 'sharp'], p: { hard: 0.8, mass: 0.8, sharp: 0.6, flam: 0.2, integ: 0.6, dmg: 3.5 }, use: ['cut', 'hit', 'throw'], on: ['carry', 'break', 'craft'] },
    knife:    { label: 'Slicer',  tags: ['solid', 'tool', 'weapon', 'sharp'], p: { hard: 0.8, mass: 0.2, sharp: 0.9, flam: 0.1, integ: 0.4, dmg: 2.2 }, use: ['cut', 'pierce', 'hit'], on: ['carry', 'break', 'craft'] },
    bow:      { label: 'Launcher', tags: ['tool', 'weapon', 'ranged', 'wood'], p: { hard: 0.4, mass: 0.4, sharp: 0, flam: 0.5, integ: 0.4, dmg: 2.5, reach: 5 }, use: ['shoot', 'hit'], on: ['carry', 'break', 'craft', 'burn'] },
    vest:     { label: 'Armour',  tags: ['solid', 'worn', 'armor'],         p: { hard: 0.6, mass: 0.7, sharp: 0, flam: 0.4, integ: 0.6, prot: 1 }, use: ['block', 'shelter'], on: ['carry', 'break', 'craft', 'burn', 'wear'] },
    cloak:    { label: 'Clothing', tags: ['soft', 'worn', 'cloth', 'flammable'], p: { hard: 0, mass: 0.3, sharp: 0, flam: 0.8, integ: 0.3, warm: 0.6 }, use: ['warm', 'shelter'], on: ['carry', 'burn', 'cut', 'craft', 'wear'] },
    waterskin: { label: 'Waterskin', tags: ['soft', 'container', 'liquid-safe'], p: { hard: 0.1, mass: 0.3, sharp: 0, flam: 0.4, integ: 0.3, cap: 1 }, use: ['store', 'drink'], on: ['fill', 'carry', 'cut', 'burn'] },
    berries:  { label: 'Berries', tags: ['soft', 'edible', 'plant'],        p: { hard: 0, mass: 0.05, sharp: 0, flam: 0.2, integ: 0.05, food: 0.3 }, use: ['eat', 'feed', 'throw'], on: ['eat', 'carry', 'feed', 'store', 'burn'] },
    hide:     { label: 'Hide',    tags: ['soft', 'material', 'flammable'],  p: { hard: 0.15, mass: 0.2, sharp: 0, flam: 0.5, integ: 0.4 }, use: ['craft', 'warm'], on: ['carry', 'cut', 'burn', 'craft'] },
    fiber:    { label: 'Fiber',   tags: ['soft', 'material', 'flammable'],  p: { hard: 0, mass: 0.05, sharp: 0, flam: 0.9, integ: 0.1 }, use: ['craft', 'light'], on: ['carry', 'cut', 'burn', 'craft'] },
    bone:     { label: 'Bone',    tags: ['solid', 'material', 'hard'],      p: { hard: 0.65, mass: 0.3, sharp: 0.5, flam: 0.05, integ: 0.5 }, use: ['hit', 'cut', 'craft', 'throw'], on: ['carry', 'break', 'craft'] },
    shell:    { label: 'Shell',   tags: ['solid', 'material', 'hard', 'sharp'], p: { hard: 0.6, mass: 0.15, sharp: 0.6, flam: 0, integ: 0.4 }, use: ['cut', 'hit', 'craft', 'throw'], on: ['carry', 'hit', 'craft', 'break'] },
    copper:   { label: 'Native copper', tags: ['solid', 'material', 'hard', 'metal', 'malleable'], p: { hard: 0.55, mass: 0.6, sharp: 0.25, flam: 0, integ: 1.2 }, use: ['hit', 'cut', 'craft', 'throw'], on: ['carry', 'hit', 'burn', 'craft', 'cut'] },
    gold:     { label: 'Native gold', tags: ['solid', 'material', 'metal', 'malleable', 'precious'], p: { hard: 0.25, mass: 0.9, sharp: 0.1, flam: 0, integ: 1 }, use: ['hit', 'craft'], on: ['carry', 'hit', 'burn', 'craft', 'cut'] }
  };
  // verbs: needs.use / needs.on are tag lists (any one satisfies); run(use, on, f) returns effects; kind 'xform' returns a derived operand instead
  function opOf(x) { return typeof x === 'string' ? { k: x, tags: THINGS[x].tags.slice(), p: Object.assign({}, THINGS[x].p) } : x; }
  function hasTag(o, ts) { if (!ts) return true; return ts.some(function (t) { return o.tags.indexOf(t) >= 0; }); }
  function clamp01(v) { return Math.max(0, Math.min(1, v)); }
  function force(o) { var p = o.p; return p.dmg !== undefined ? p.dmg : Math.round((3 * p.hard * Math.min(1, p.mass * 1.4) + 2.5 * p.sharp) * 10) / 10; }
  var VERBS = {
    hit:    { label: 'Hit',    text: 'Strike with force (hardness, mass, edge).', needs: { use: ['solid', 'living'], on: ['solid', 'living'] }, run: function (u, o, f) {
      var raw = (0.035 + force(u) * 0.012) * (f.k || 1) * (u.airborne ? 0.8 : 1), arm = f.armor ? Math.min(0.65, f.armor / 12) : 0, d = raw * (1 - arm) * (1 - 0.35 * (o.p.hard || 0) * (o.tags.indexOf('living') < 0 ? 1 : 0));
      if (o.tags.indexOf('living') >= 0) return [{ on: 'on', stat: 'hea', d: -d }, { on: 'on', stat: 'inj', d: d * 4 }, { on: 'on', stat: 'stress', d: 0.4 }, { on: 'use', stat: 'wear', d: 0.015 }];
      return [{ on: 'on', stat: 'integ', d: -d * 2.4 / Math.max(0.2, o.p.integ || 1) }, { on: 'use', stat: 'wear', d: 0.015 }]; } },
    pierce: { label: 'Pierce', text: 'Thrust a point; ignores a third of armour.', needs: { use: ['sharp'], on: ['solid', 'living'] }, run: function (u, o, f) { return VERBS.hit.run(u, o, { k: (f.k || 1) * 1.05, armor: (f.armor || 0) * 0.7 }); } },
    cut:    { label: 'Cut',    text: 'Draw an edge through; yields materials from plants, hides and carcasses.', needs: { use: ['sharp'], on: ['plant', 'soft', 'living', 'wood'] }, run: function (u, o, f) {
      var eff = (u.p.sharp || 0) * 1.1 + (u.p.hard || 0) * 0.3, y = (THINGS[o.k] && THINGS[o.k].yields && THINGS[o.k].yields.cut) || null, e = o.tags.indexOf('living') >= 0 ? [{ on: 'on', stat: 'hea', d: -0.03 * eff }, { on: 'on', stat: 'inj', d: 0.12 * eff }] : [{ on: 'on', stat: 'integ', d: -0.4 * eff / Math.max(0.2, o.p.integ || 1) }];
      e.push({ on: 'use', stat: 'wear', d: 0.02 }); if (y && eff > 0.5) e.push({ yields: y }); return e; } },
    shove:  { label: 'Shove',  text: 'Push away; moves light things, staggers people.', needs: { use: ['solid', 'living'], on: ['solid', 'living'] }, run: function (u, o) { return [{ on: 'on', stat: 'pos', d: clamp01((u.p.mass || 1) / Math.max(0.2, o.p.mass || 1)) > 0.5 ? 1 : 0 }, { on: 'on', stat: 'hea', d: -0.01 }]; } },
    grapple: { label: 'Grapple', text: 'Seize and hold.', needs: { use: ['living'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'held', d: 2 }, { on: 'on', stat: 'hea', d: -0.01 }]; } },
    throw:  { kind: 'xform', label: 'Throw', text: 'Launch a light solid; it becomes an airborne projectile (reach 4).', needs: { use: ['solid'] }, xform: function (u, f) {
      if ((u.p.mass || 0) > 1.2) return null; var n = { k: u.k, tags: u.tags.concat(['airborne']), p: Object.assign({}, u.p), airborne: true }; n.p.dmg = force(u); return n; } },
    swing:  { kind: 'xform', label: 'Swing', text: 'Wind a tool through an arc: heavier head, more force.', needs: { use: ['tool', 'solid'] }, xform: function (u) { var n = { k: u.k, tags: u.tags.slice(), p: Object.assign({}, u.p) }; n.p.dmg = force(u) * (1 + 0.25 * (u.p.mass || 0)); return n; } },
    light:  { kind: 'xform', label: 'Light', text: 'Ignite a flammable thing from a fire or by striking flint.', needs: { use: ['flammable'] }, xform: function (u) { var n = { k: u.k, tags: u.tags.concat(['hot', 'flame']), p: Object.assign({}, u.p, { heat: 0.7 }) }; return n; } },
    carry:  { label: 'Carry', text: 'Lift and move; limited by mass.', needs: { use: ['living'], on: ['solid', 'soft', 'liquid-safe', 'living', 'edible', 'material'] }, run: function (u, o) { var ok = (o.p.mass || 0) <= 1.5; return ok ? [{ on: 'on', stat: 'carried', d: 1 }, { on: 'use', stat: 'ene', d: -0.002 * (o.p.mass || 0) }] : [{ fail: 'too heavy' }]; } },
    burn:   { label: 'Burn', text: 'Fire spreads into flammable things and hurts living ones.', needs: { use: ['hot', 'flame'], on: ['flammable', 'living', 'wood', 'cloth', 'soft', 'plant'] }, run: function (u, o) {
      var fl = o.p.flam || 0, e = o.tags.indexOf('living') >= 0 ? [{ on: 'on', stat: 'hea', d: -0.08 }, { on: 'on', stat: 'inj', d: 0.25 }] : [{ on: 'on', stat: 'integ', d: -0.5 * fl }]; if (fl > 0.3) e.push({ on: 'on', stat: 'burning', d: 1 }); return e; } },
    douse:  { label: 'Douse', text: 'Water puts fire out.', needs: { use: ['liquid'], on: ['hot', 'flame'] }, run: function () { return [{ on: 'on', stat: 'burning', d: -1 }]; } },
    eat:    { label: 'Eat', text: 'Edible things feed the eater.', needs: { use: ['living'], on: ['edible'] }, run: function (u, o) { return [{ on: 'use', stat: 'sat', d: o.p.food || 0.2 }, { on: 'on', stat: 'consumed', d: 1 }]; } },
    drink:  { label: 'Drink', text: 'Drink from water or a skin.', needs: { use: ['living'], on: ['liquid', 'container'] }, run: function () { return [{ on: 'use', stat: 'hyd', d: 0.4 }]; } },
    fill:   { label: 'Fill', text: 'Fill a container from water.', needs: { use: ['liquid'], on: ['container'] }, run: function () { return [{ on: 'on', stat: 'filled', d: 1 }]; } },
    store:  { label: 'Store', text: 'Place something in a container up to its capacity.', needs: { use: ['container'], on: ['solid', 'soft', 'edible', 'material'] }, run: function (u, o) { return (u.p.cap || 0) >= (o.p.mass || 0.1) ? [{ on: 'use', stat: 'stored', d: o.p.mass || 0.1 }] : [{ fail: 'full' }]; } },
    open:   { label: 'Open', text: 'Open a door, window or crate.', needs: { use: ['living'], on: ['portal', 'container'] }, run: function () { return [{ on: 'on', stat: 'open', d: 1 }]; } },
    climb:  { label: 'Climb', text: 'Get up onto something solid; gives height.', needs: { use: ['living'], on: ['solid'] }, run: function (u, o) { return (o.p.mass || 0) >= 0.5 ? [{ on: 'use', stat: 'height', d: 1 }, { on: 'use', stat: 'ene', d: -0.003 }] : [{ fail: 'too small to stand on' }]; } },
    sit:    { label: 'Sit', text: 'Rest on something stable.', needs: { use: ['living'], on: ['solid', 'soft', 'furniture'] }, run: function (u, o) { return [{ on: 'use', stat: 'ene', d: 0.004 }]; } },
    sleep:  { label: 'Sleep', text: 'Recover energy; better on a bed or in shelter.', needs: { use: ['living'], on: ['furniture', 'shelter', 'soft', 'structure'] }, run: function (u, o) { return [{ on: 'use', stat: 'ene', d: 0.01 + 0.01 * (o.p.warm || 0) * 2 }]; } },
    shelter: { label: 'Shelter', text: 'Standing inside walls, trees or a cave cuts weather and cold.', needs: { use: ['living'], on: ['structure', 'shelter', 'plant'] }, run: function (u, o) { return [{ on: 'use', stat: 'temp', d: 0.05 }]; } },
    warm:   { label: 'Warm', text: 'Standing by fire or wearing insulation warms the body.', needs: { use: ['living'], on: ['hot', 'cloth', 'soft'] }, run: function (u, o) { return [{ on: 'use', stat: 'temp', d: 0.08 * (o.p.heat || o.p.warm || 0.3) }]; } },
    block:  { label: 'Block', text: 'Put something solid in the way; absorbs a blow.', needs: { use: ['solid', 'structure'], on: ['solid', 'living'] }, run: function (u) { return [{ on: 'use', stat: 'absorb', d: clamp01((u.p.hard || 0) * 0.6 + (u.p.mass || 0) * 0.1) }]; } },
    break:  { label: 'Break', text: 'Smash through; needs force above the thing\'s integrity.', needs: { use: ['solid'], on: ['solid', 'structure', 'wood'] }, run: function (u, o) { var f = force(u) / 3; return f > (o.p.integ || 1) * 0.8 ? [{ on: 'on', stat: 'integ', d: -9 }] : [{ on: 'on', stat: 'integ', d: -f * 0.3 }]; } },
    craft:  { label: 'Craft', text: 'Combine materials at hand or a bench into tools and gear.', needs: { use: ['living', 'solid'], on: ['material', 'solid', 'soft'] }, run: function () { return [{ on: 'on', stat: 'crafted', d: 1 }]; } },
    shoot:  { label: 'Shoot', text: 'Launch a projectile at range.', needs: { use: ['ranged'], on: ['solid', 'living'] }, run: function (u, o, f) { return VERBS.hit.run({ k: 'arrow', tags: ['solid'], p: { hard: 0.5, mass: 0.05, sharp: 0.6, dmg: force(u) } }, o, f); } },
    gather: { label: 'Gather', text: 'Pick fruit or fibre from plants.', needs: { use: ['living'], on: ['plant'] }, run: function (u, o) { var y = (THINGS[o.k] && THINGS[o.k].yields && THINGS[o.k].yields.gather) || null; return y ? [{ yields: y }] : [{ fail: 'nothing to gather' }]; } },
    talk:   { label: 'Talk', text: 'Calm or provoke; shifts stress and anger.', needs: { use: ['living'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'stress', d: -0.05 }]; } },
    heal:   { label: 'Heal', text: 'Tend wounds; speeds recovery.', needs: { use: ['living'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'inj', d: -0.1 }, { on: 'on', stat: 'hea', d: 0.02 }]; } },
    feed:   { label: 'Feed', text: 'Give food, or fuel to a fire; tames animals over time.', needs: { use: ['edible', 'flammable'], on: ['living', 'hot', 'flame'] }, run: function (u, o) { return o.tags.indexOf('living') >= 0 ? [{ on: 'on', stat: 'sat', d: u.p.food || 0.1 }, { on: 'on', stat: 'trust', d: 0.1 }] : [{ on: 'on', stat: 'fuel', d: 1 }]; } },
    tame:   { label: 'Tame', text: 'Build trust with an animal through food.', needs: { use: ['living'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'trust', d: 0.15 }]; } },
    scare:  { label: 'Scare', text: 'Fire and loud predators drive animals away.', needs: { use: ['flame', 'living'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'flee', d: 1 }]; } },
    cook:   { label: 'Cook', text: 'Heat makes food safer and richer.', needs: { use: ['hot', 'flame'], on: ['edible'] }, run: function () { return [{ on: 'on', stat: 'food', d: 0.1 }]; } },
    look:   { label: 'Look', text: 'See through; gives light and vision.', needs: { use: ['portal'], on: ['living', 'solid'] }, run: function () { return []; } },
    hide:   { label: 'Hide', text: 'Be out of sight inside a shelter.', needs: { use: ['shelter'], on: ['living'] }, run: function () { return [{ on: 'on', stat: 'hidden', d: 1 }]; } },
    enter:  { label: 'Enter', text: 'Walk into a cave or building.', needs: { use: ['living'], on: ['shelter', 'portal', 'structure'] }, run: function () { return [{ on: 'use', stat: 'inside', d: 1 }]; } },
    wear:   { label: 'Wear', text: 'Put on gear; armour absorbs hits, cloth insulates.', needs: { use: ['living'], on: ['worn'] }, run: function (u, o) { return [{ on: 'use', stat: o.tags.indexOf('armor') >= 0 ? 'prot' : 'insul', d: o.p.prot || o.p.warm || 0.3 }]; } },
    cross:  { label: 'Cross', text: 'Wade or swim.', needs: { use: ['living'], on: ['liquid'] }, run: function () { return [{ on: 'use', stat: 'ene', d: -0.01 }]; } },
    research: { label: 'Research', text: 'Work a bench to build insight.', needs: { use: ['living'], on: ['furniture'] }, run: function () { return [{ on: 'use', stat: 'insight', d: 0.25 }]; } }
  };
  // run a verb between two operands (thing keys or derived operands); returns { ok, why, effects, yields }
  function interact(verb, use, on, f) {
    var V = VERBS[verb]; if (!V) return { ok: false, why: 'unknown verb ' + verb, effects: [], yields: [] };
    var u = opOf(use), o = on === undefined || on === null ? null : opOf(on); f = f || {};
    if (V.kind === 'xform') { if (!hasTag(u, V.needs.use)) return { ok: false, why: u.k + ' cannot be ' + V.label.toLowerCase() + 'n', effects: [], yields: [] }; var x = V.xform(u, f); return x ? { ok: true, operand: x, effects: [], yields: [] } : { ok: false, why: u.k + ' is too heavy to ' + V.label.toLowerCase(), effects: [], yields: [] }; }
    if (!hasTag(u, V.needs.use)) return { ok: false, why: u.k + ' cannot be used to ' + verb, effects: [], yields: [] };
    if (!o) return { ok: false, why: 'needs a target', effects: [], yields: [] };
    if (!hasTag(o, V.needs.on)) return { ok: false, why: verb + ' does nothing to ' + o.k, effects: [], yields: [] };
    var raw = V.run(u, o, f) || [], effects = [], yields = [], why = '';
    raw.forEach(function (e) { if (e.yields) e.yields.forEach(function (y) { yields.push(y); }); else if (e.fail) why = e.fail; else effects.push(e); });
    return { ok: !why, why: why, effects: effects, yields: yields, use: u, on: o };
  }
  // composite trees: ['hit', {use: ['throw', {use: 'rock'}], on: 'person'}]
  function evalNode(node, f) {
    if (typeof node === 'string') return { ok: true, operand: opOf(node), effects: [], yields: [], trace: node };
    var verb = node[0], slots = node[1] || {}, trace = [];
    function side(x) { if (x === undefined) return undefined; if (typeof x === 'string') return x; var r = evalNode(x, f); trace.push(r.trace); if (!r.ok || !r.operand) return r; return r.operand; }
    var u = side(slots.use), o = side(slots.on);
    if (u && u.ok === false) return u; if (o && o.ok === false) return o;
    var r = interact(verb, u, o, f);
    r.trace = verb + '(use(' + (typeof slots.use === 'string' ? slots.use : (slots.use ? slots.use[0] + '(' + (slots.use[1] && slots.use[1].use) + ')' : '')) + ')' + (slots.on !== undefined ? ', on(' + (typeof slots.on === 'string' ? slots.on : slots.on[0] + '(..)') + '))' : ')');
    return r;
  }
  var COMP = {
    rock_throw:  { label: 'Throw a rock',        tree: ['hit', { use: ['throw', { use: 'rock' }], on: 'person' }] },
    club_smash:  { label: 'Swing a club',        tree: ['hit', { use: ['swing', { use: 'club' }], on: 'person' }] },
    spear_poke:  { label: 'Spear thrust',        tree: ['pierce', { use: 'spear', on: 'prey' }] },
    axe_fell:    { label: 'Fell a tree',         tree: ['cut', { use: 'axe', on: 'tree' }] },
    knife_skin:  { label: 'Skin a kill',         tree: ['cut', { use: 'knife', on: 'prey' }] },
    firebrand:   { label: 'Burn a bush with a lit stick', tree: ['burn', { use: ['light', { use: 'stick' }], on: 'bush' }] },
    flint_spark: { label: 'Strike a spark into fibre', tree: ['burn', { use: ['light', { use: 'fiber' }], on: 'stick' }] },
    break_door:  { label: 'Break a door with a boulder-weight club', tree: ['break', { use: ['swing', { use: 'club' }], on: 'door' }] },
    sleep_cave:  { label: 'Sleep in a cave',     tree: ['sleep', { use: 'person', on: 'cave' }] },
    fill_skin:   { label: 'Fill a waterskin',    tree: ['fill', { use: 'water', on: 'waterskin' }] },
    eat_berries: { label: 'Eat berries',         tree: ['eat', { use: 'person', on: 'berries' }] },
    stack_store: { label: 'Store a rock in a crate', tree: ['store', { use: 'crate', on: 'rock' }] }
  };
  function affordances(k) { var T = THINGS[k]; return T ? { use: T.use.slice(), on: T.on.slice() } : null; }
  // matrix of thing x verb: 'U' used for, 'O' done to, 'B' both
  function matrix() { var vs = Object.keys(VERBS); return { verbs: vs, rows: Object.keys(THINGS).map(function (k) { var T = THINGS[k]; return { key: k, label: T.label, cells: vs.map(function (v) { var u = T.use.indexOf(v) >= 0, o = T.on.indexOf(v) >= 0; return u && o ? 'B' : u ? 'U' : o ? 'O' : ''; }) }; }) }; }
  function techOf(k) { for (var i = 0; i < TECH.length; i++) if (TECH[i].key === k) return TECH[i]; return null; }
  function schemOf(techs) { var o = []; TECH.forEach(function (t) { if (techs.indexOf(t.key) >= 0) t.schem.forEach(function (k) { if (o.indexOf(k) < 0) o.push(k); }); }); return o; }
  function machinesOf(techs) { var o = []; TECH.forEach(function (t) { if (techs.indexOf(t.key) >= 0) t.machines.forEach(function (k) { if (o.indexOf(k) < 0) o.push(k); }); }); return o; }
  function techAvail(techs) { return TECH.filter(function (t) { return techs.indexOf(t.key) < 0 && t.req.every(function (r) { return techs.indexOf(r) >= 0; }); }); }
  function nextTech(techs) { var a = techAvail(techs).sort(function (x, y) { return x.cost - y.cost; }); return a[0] || null; }
  function techFx(techs, k) { var v = 0; TECH.forEach(function (t) { if (techs.indexOf(t.key) >= 0 && t.fx[k]) v += t.fx[k]; }); return v; }


  /* ---------- discovery by experiment: nothing is researched; colonists try verb x tool x target combinations through the interaction layer.
     A combination that works is learned by that person, can be shown to others, watched and copied, and is forgotten when the last one who knows it dies.
     A technology exists only while the colony's living knowledge still satisfies its pattern. ---------- */
  var MAT2T = { wood: 'stick', stone: 'rock', flint: 'flint', bone: 'bone', hide: 'hide', fur: 'hide', fiber: 'fiber', sinew: 'fiber', resin: 'fiber', horn: 'bone', tusk: 'bone', claw: 'bone', feather: 'fiber', hoof: 'bone', scale: 'shell', fat: 'hide', shell: 'shell', copper: 'copper', gold: 'gold' };
  var XVERBS = ['hit', 'cut', 'pierce', 'break', 'craft', 'burn', 'douse', 'fill', 'cook', 'gather', 'climb'], PVERBS = { climb: 1, gather: 1 };
  function opSig(sg) { var p = sg.split(':'), o = THINGS[p[p.length - 1]] ? opOf(p[p.length - 1]) : null; if (!o) return null; for (var i = p.length - 2; i >= 0; i--) { var r = interact(p[i], o); if (!r.ok || !r.operand) return null; o = r.operand; } return o; }
  var SIGC = {};
  function sigInfo(sig) { if (SIGC[sig]) return SIGC[sig]; var p = sig.split('|'), u = opSig(p[1]), o = opSig(p[2]); return (SIGC[sig] = { v: p[0], u: u, o: o, ut: u ? u.tags.concat([u.k]) : [], ot: o ? o.tags.concat([o.k]) : [] }); }
  function sigTry(sig) { var I = sigInfo(sig); if (!I.u || !I.o) return null; var r = interact(I.v, I.u, I.o, { k: 1 }); return r.ok && (r.effects.length || r.yields.length) ? r : null; }
  function sigLabel(sig) { var p = sig.split('|'), nm = function (x) { if (x === 'person') return 'bare hands'; var q = x.split(':'), b = THINGS[q[q.length - 1]] ? THINGS[q[q.length - 1]].label.toLowerCase() : q[q.length - 1]; return q.length > 1 ? 'lit ' + b : b; }; return (VERBS[p[0]] ? VERBS[p[0]].label : p[0]) + ' ' + nm(p[1]) + ' on ' + nm(p[2]); }
  function expCands(has, flame) {
    var out = [], seen = {}, uses = has.filter(function (k) { return THINGS[k] && THINGS[k].tags.indexOf('plant') < 0 || k === 'berries'; }), lit = [];
    if (flame) { uses.push('fire'); if (has.indexOf('fire') < 0) has = has.concat(['fire']); }
    if (flame || has.indexOf('flint') >= 0) uses.forEach(function (k) { if (THINGS[k].tags.indexOf('flammable') >= 0) lit.push('light:' + k); });
    var U = uses.concat(lit, ['person']);
    XVERBS.forEach(function (v) { U.forEach(function (u) { has.forEach(function (o) {
      if (u === 'person' && !PVERBS[v]) return; if (u === o && v !== 'hit') return;
      var sg = v + '|' + u + '|' + o; if (seen[sg]) return; seen[sg] = 1; if (sigTry(sg)) out.push(sg); }); }); });
    return out;
  }
  // goals and memory steer which experiment is tried: needs of the body, the frontier of technologies nearly within reach, curiosity about what nobody knows, and what the person already tried
  // what an experiment is about: material physics (sci), making and joining (craft), food and water (food), fire (fire), or going out to the terrain itself (explore)
  var XCLS = { hit: 'sci', cut: 'sci', pierce: 'sci', 'break': 'sci', craft: 'craft', burn: 'fire', douse: 'fire', fill: 'food', cook: 'food', gather: 'food', climb: 'explore' }, XTERR = { tree: 1, rock: 1, water: 1, bush: 1 };
  function expClass(sg) { var p = sg.split('|'), v = p[0].indexOf('light:') === 0 ? 'burn' : p[0]; return XTERR[p[2]] && v !== 'cook' && v !== 'gather' && v !== 'fill' ? 'explore' : (XCLS[v] || 'sci'); }
  function expScore(sg, c) {
    var I = sigInfo(sg), s = 1;
    if (!c.know[sg]) s += 1.5; if (!c.colony[sg]) s += 1; else if (c.know[sg]) s *= 0.25;
    if (c.need.food && (I.v === 'gather' || I.v === 'cook')) s += 2; if (c.need.cold && (I.v === 'burn' || sg.indexOf('light:') > 0)) s += 2; if (c.need.hurt && I.v === 'burn') s += 0.5;
    for (var i = 0; i < c.front.length; i++) if (patHit(sg, c.front[i])) { s += 2.5; break; }
    if (c.foc) s *= Math.max(0.1, Math.pow(c.foc[expClass(sg)] || 1, (typeof window !== 'undefined' && window.AGON_P && window.AGON_P.focP) || 1));
    var x = c.xp[sg]; if (x && !c.know[sg]) { var xw = (typeof window !== 'undefined' && window.AGON_P && window.AGON_P.xpW != null) ? window.AGON_P.xpW : 0.3; s = Math.max(0.2, s + Math.max(-0.8, Math.min(1.5, xw * x.n))); }
    return s;
  }
  function expPick(cands, c, rnd) {
    if (typeof window !== 'undefined' && typeof window.AGON_POLICY === 'function') { try { var ch = window.AGON_POLICY({ id: c.id, tick: c.tick, candidates: cands.slice(), know: Object.keys(c.know), colony: Object.keys(c.colony), need: c.need }); if (ch && cands.indexOf(ch) >= 0) return ch; } catch (e) { } }
    var w = cands.map(function (sg) { return expScore(sg, c); }), t = w.reduce(function (a, b) { return a + b; }, 0), x = rnd() * t;
    for (var i = 0; i < cands.length; i++) { x -= w[i]; if (x <= 0) return cands[i]; } return cands[cands.length - 1];
  }
  function expOdds(sg, c) { var x = c.xp[sg], rel = 1, I = sigInfo(sg); for (var k in c.know) { var J = sigInfo(k); if (J.v === I.v || (J.u && I.u && J.u.k === I.u.k)) { rel = 1.3; break; } }
    return Math.min(0.6, ((typeof window !== 'undefined' && window.AGON_P && window.AGON_P.oddsK) || 1) * 0.15 * c.intel * (0.6 + 0.8 * c.curio) * (c.bench ? 1.4 : 1) * rel * (1 + 0.15 * Math.min(5, x ? x.n : 0))); }
  // technology = a pattern over discovered combinations (verb, user tag, target tag) with a minimum number met, plus its prerequisites
  var LORE = {
    firecraft:  { min: 1, p: [['burn', 'flame', 'wood'], ['burn', 'flame', 'fiber'], ['burn', 'flame', 'plant']] },
    // Shelter was reachable one way only: craft wood against something, twice. That made every
    // colony arrive at it from the workbench, and a band that spent a winter shivering in a cave
    // learned nothing from it. The four 'ev' lines below are the other ways a person works out
    // that walls are worth having -- being cold, sitting out a storm, taking refuge in a rock
    // overhang, and seeing someone else's house. They are counted per colonist and carried in
    // a.work, so like every other kind of evidence here they die with the people who hold them.
    shelter:    { min: 2, p: [['craft', 'wood', 'wood'], ['craft', 'stone', 'wood'], ['craft', 'wood', 'stone'], ['craft', 'wood', 'fiber'], ['craft', 'wood', 'hide'], ['hit', 'wood', 'hard'], ['craft', 'wood', 'bone'], ['ev', 'cold', 2000], ['ev', 'stormed', 1800], ['ev', 'sheltered', 25], ['ev', 'sawwall', 60]] },
    bedding:    { min: 2, p: [['cut', 'sharp', 'fiber'], ['cut', 'sharp', 'hide'], ['craft', 'bone', 'fiber'], ['craft', 'bone', 'hide'], ['craft', 'stone', 'hide'], ['craft', 'stone', 'fiber'], ['craft', 'wood', 'hide']] },
    storage:    { min: 2, p: [['craft', 'wood', 'fiber'], ['craft', 'bone', 'hide'], ['craft', 'shell', 'fiber'], ['craft', 'shell', 'hide'], ['craft', 'wood', 'shell'], ['craft', 'wood', 'wood'], ['craft', 'wood', 'hide']] },
    carpentry:  { min: 2, p: [['cut', 'sharp', 'wood'], ['craft', 'wood', 'wood'], ['hit', 'wood', 'wood'], ['hit', 'hard', 'wood'], ['craft', 'stone', 'wood'], ['craft', 'sharp', 'wood']] },
    adornment:  { min: 1, p: [['craft', 'precious', 'fiber'], ['craft', 'shell', 'fiber'], ['craft', 'precious', 'hide'], ['craft', 'shell', 'hide']] },
    markmaking: { min: 2, p: [['cut', 'sharp', 'solid'], ['cut', 'sharp', 'hide'], ['burn', 'flame', 'wood'], ['hit', 'hard', 'solid']] },
    wells:      { min: 2, p: [['fill', 'liquid', 'container'], ['break', 'hard', 'solid'], ['hit', 'hard', 'hard'], ['hit', 'wood', 'hard']] },
    shaping:    { min: 1, p: [['hit', 'hard', 'hard'], ['hit', 'hard', 'wood'], ['cut', 'sharp', 'wood']] },
    metalwork:  { min: 2, p: [['hit', 'hard', 'metal'], ['burn', 'flame', 'metal'], ['cut', 'sharp', 'metal'], ['hit', 'metal', 'hard']] },
    binding:    { min: 1, p: [['craft', 'wood', 'fiber'], ['craft', 'wood', 'hide'], ['craft', 'bone', 'hide']] },
    torchcraft: { min: 2, p: [['craft', 'wood', 'fiber'], ['burn', 'flame', 'wood'], ['burn', 'flame', 'fiber']] },
    cooking:    { min: 1, p: [['cook', 'flame', 'edible'], ['cook', 'hot', 'edible']] },
    hafting:    { min: 2, p: [['craft', 'stone', 'wood'], ['craft', 'sharp', 'wood'], ['craft', 'wood', 'fiber']] },
    tanning:    { min: 2, p: [['cut', 'sharp', 'soft'], ['craft', 'bone', 'hide'], ['hit', 'stone', 'hide']] },
    leverage:   { min: 2, p: [['hit', 'wood', 'stone'], ['break', 'wood', 'solid'], ['break', 'stone', 'wood']] },
    spinning:   { min: 2, p: [['craft', 'bone', 'fiber'], ['cut', 'sharp', 'fiber'], ['burn', 'flame', 'fiber']] },
    footwork:   { min: 1, p: [['ev', 'hunt', 4]] },
    spearfishing: { min: 1, p: [['ev', 'fish', 3]] },
    polefishing: { min: 1, p: [['ev', 'spearfish', 4]] },
    herbalism:  { min: 2, p: [['gather', 'living', 'plant'], ['eat', 'living', 'edible'], ['cut', 'sharp', 'plant']] },
    cultivation: { min: 2, p: [['gather', 'living', 'plant'], ['cut', 'sharp', 'plant'], ['hit', 'wood', 'plant']] },
    tension:    { min: 2, p: [['hit', 'wood', 'wood'], ['break', 'wood', 'wood'], ['craft', 'wood', 'fiber']] },
    shielding:  { min: 2, p: [['craft', 'wood', 'hide'], ['block', 'solid', 'solid'], ['hit', 'solid', 'hide']] },
    weaving:    { min: 3, p: [['craft', 'wood', 'fiber'], ['craft', 'bone', 'fiber'], ['craft', 'wood', 'wood']] },
    kilning:    { min: 3, p: [['burn', 'flame', 'wood'], ['douse', 'liquid', 'flame'], ['hit', 'hard', 'hard']] },
    wheelwork:  { min: 3, p: [['hit', 'wood', 'wood'], ['break', 'stone', 'wood'], ['craft', 'wood', 'stone'], ['craft', 'stone', 'wood']] },
    climbing:   { min: 2, p: [['climb', 'living', 'solid'], ['craft', 'wood', 'fiber'], ['hit', 'wood', 'stone']] },
    tactics:    { min: 1, p: [['ev', 'hunt', 14]] },
    curing:     { min: 2, p: [['burn', 'flame', 'hide'], ['cook', 'flame', 'edible'], ['cut', 'sharp', 'soft']] },
    medicine:   { min: 3, p: [['burn', 'flame', 'plant'], ['fill', 'liquid', 'container'], ['douse', 'liquid', 'flame'], ['cook', 'flame', 'edible']] },
    waterpower: { min: 3, p: [['fill', 'liquid', 'container'], ['craft', 'wood', 'wood'], ['hit', 'wood', 'wood']] },
    heatengine: { min: 4, p: [['burn', 'flame', 'wood'], ['fill', 'liquid', 'container'], ['douse', 'liquid', 'flame'], ['cook', 'flame', 'edible']] }
  };
  function mt(list, t) { return t === '*' || list.indexOf(t) >= 0; }
  function patHit(sg, pat) { if (pat[0] === 'ev') return false; var I = sigInfo(sg); return I.v === pat[0] && !!I.u && !!I.o && mt(I.ut, pat[1]) && mt(I.ot, pat[2]); }
  function loreHave(key, know, ev) { var L = LORE[key], n = 0; if (!L) return { have: 0, min: 0 }; L.p.forEach(function (pat) { if (pat[0] === 'ev') { if (((ev || {})[pat[1]] || 0) >= pat[2]) n++; return; } for (var sg in know) if (patHit(sg, pat)) { n++; break; } }); return { have: n, min: L.min }; }
  function loreDerive(know, ev, forced) { var out = (forced || []).slice(), ch = true; while (ch) { ch = false; TECH.forEach(function (t) { if (out.indexOf(t.key) >= 0 || !t.req.every(function (r) { return out.indexOf(r) >= 0; })) return; var h = loreHave(t.key, know, ev); if (LORE[t.key] && h.have >= h.min) { out.push(t.key); ch = true; } }); } return out; }
  // patterns of technologies whose prerequisites are met but whose evidence is incomplete: what a curious mind would work toward
  function loreFront(techs, know, ev) { var out = []; TECH.forEach(function (t) { if (techs.indexOf(t.key) >= 0 || !LORE[t.key] || !t.req.every(function (r) { return techs.indexOf(r) >= 0; })) return; LORE[t.key].p.forEach(function (pat) { if (pat[0] === 'ev') return; var got = false; for (var sg in know) if (patHit(sg, pat)) { got = true; break; } if (!got) out.push(pat); }); }); return out; }
  function loreTeach(tKnow, lKnow, colony, rnd) { var c = Object.keys(tKnow).filter(function (k) { return !lKnow[k]; }); if (!c.length) return null; var w = c.map(function (k) { return Object.keys(LORE).some(function (t) { return LORE[t].p.some(function (p) { return patHit(k, p); }); }) ? 3 : 1; }), tot = w.reduce(function (a, b) { return a + b; }, 0), x = rnd() * tot; for (var i = 0; i < c.length; i++) { x -= w[i]; if (x <= 0) return c[i]; } return c[0]; }
  function sigYields(sg) { var r = sigTry(sg); return r ? r.yields : []; }

  /* ---------- thoughts as words: plain first-person lines built from the agent's real state, no language model ---------- */
  function thoughtLine(key, c) {
    var pick = function (arr) { return arr[(c.seed || 0) % arr.length]; };
    if (!c.alive) return 'Nothing.';
    if (key === 'food') {
      if (c.sat < 0.15) return pick(['I am starving. I need food now.', 'My stomach is cramping. Anything to eat.']);
      if (c.berries > 0) return pick(['I have ' + c.berries + ' berries in my pack. I will eat soon.', 'Berries in the pack. Eat when it gets worse.']);
      if (c.tableFood > 0) return 'There are ' + c.tableFood + ' berries on the table.';
      if (c.bushes > 0) return 'There are still ' + c.bushes + ' bushes with berries out there.';
      return pick(['No food anywhere I know of.', 'I should look for something to hunt.']);
    }
    if (key === 'water') {
      if (c.hyd < 0.2) return 'My mouth is dry. I must reach water.';
      return pick(['My waterskin is ' + Math.round(c.skin * 100) + '% full.', 'The water is not far.']);
    }
    if (key === 'rest') {
      if (c.sleeping) return pick(['Warm enough. Sleep.', 'Dreaming of somewhere green.']);
      if (c.ene < 0.2) return 'I can barely keep my eyes open.';
      return c.bed >= 0 ? 'I should go back to my bed.' : 'I have no bed. I will sleep rough.';
    }
    if (key === 'safety') {
      if (c.pred) return 'A ' + c.pred + ' is close. Keep away from it.';
      if (c.injured) return pick(['My wound hurts.', 'I need to stay still and heal.']);
      if (c.infected) return 'Something is wrong with my wound.';
      return 'I do not feel safe.';
    }
    if (key === 'warmth') return c.temp < 0 ? pick(['I am freezing.', 'I need the fire.']) : pick(['It is too hot out here.', 'I need shade and water.']);
    if (key === 'social') return c.near.length ? pick([c.near[0] + ' is close. I would like to talk.', 'Good to have ' + c.near[0] + ' nearby.']) : 'I am alone. I miss the others.';
    if (key === 'work') {
      if (c.crafting) return 'This ' + (c.crafting || 'thing') + ' is coming together.';
      if (c.mats) return 'I carry ' + c.mats + '. It should go on the shelves.';
      if (c.research) return 'If I try ' + c.research + ' differently, it might work.';
      return pick(['There is always something to make.', 'I should gather wood and stone.']);
    }
    return pick(['I wonder what is past the ' + (c.terrain || 'ridge') + '.', 'Just walking.', 'Nothing urgent.']);
  }


  /* ---------- gene details: what a value means in numbers, for the helix hover ---------- */
  function sg(x) { return (x >= 0 ? '+' : '') + Math.round(x * 100) + '%'; }
  var GENE_FX = {
    frame: [['Height', function (v) { return sg(0.9 + 0.1 * v - 1); }], ['Body mass', function (v) { return sg(v - 1); }]],
    muscle: [['Carry capacity', function (v) { return sg(v - 1); }], ['Strike force', function (v) { return sg((v - 1) * 0.5); }]],
    boneDensity: [['Injury taken', function (v) { return sg(1 / Math.sqrt(v) - 1); }]],
    fatStore: [['Hunger rate', function (v) { return sg(1 / Math.sqrt(v) - 1); }]],
    strength: [['Strike force', function (v) { return sg(v - 1); }], ['Hauling', function (v) { return sg(v - 1); }]],
    speed: [['Move speed', function (v) { return sg(v - 1); }]],
    resist: [['Infection recovery', function (v) { return sg(v - 1); }], ['Venom resistance', function (v) { return sg(v - 1); }]],
    insulation: [['Warmth in the cold', function (v) { return sg(v - 1); }]],
    metabolism: [['Food burn', function (v) { return sg(v - 1); }]],
    lifespan: [['Years of life', function (v) { return sg(v - 1); }]],
    stamina: [['Energy drain', function (v) { return sg(1 / v - 1); }]],
    lungs: [['Energy drain', function (v) { return sg(1 / Math.sqrt(v) - 1); }]],
    hydration: [['Thirst rate', function (v) { return sg(1 / v - 1); }]],
    healing: [['Health regeneration', function (v) { return sg(v - 1); }], ['Injury recovery', function (v) { return sg(v - 1); }]],
    painTol: [['Injury taken', function (v) { return sg(1 / v - 1); }]],
    immunity: [['Infection chance', function (v) { return sg(1 / v - 1); }], ['Infection recovery', function (v) { return sg(v - 1); }]],
    vision: [['Sight range', function (v) { return sg(v - 1); }]], hearing: [['Alert range', function (v) { return sg(v - 1); }]], smell: [['Scent range', function (v) { return sg(v - 1); }]],
    intel: [['Research insight', function (v) { return sg(v - 1); }], ['Craft skill gain', function (v) { return sg((v - 1) * 0.5); }]],
    memory: [['Place memory', function (v) { return sg(v - 1); }]], focus: [['Training speed', function (v) { return sg((v - 1) * 0.5); }]],
    fertility: [['Birth chance', function (v) { return sg(v - 1); }]], growth: [['Childhood length', function (v) { return sg(1 / v - 1); }]]
  };
  var PERS_FX = [
    'Stands ground against threats instead of fleeing', 'Seeks company, shares food and camp time', 'Explores, tinkers, speeds research',
    'Waits out hunger and bad weather', 'Fights back, hits harder, picks quarrels', 'Keeps working at crafting and hauling',
    'Helps the hurt and shares', 'Hoards materials and food'];

  /* ---------- mutations: heritable gene variants. Emergence-lite: rare spontaneous variants plus small drift on every birth, recombined from two parents;
     the sim then selects (a bad variant starves or gets caught, a good one survives and breeds). fx are multipliers on gene values. ---------- */
  var MUTATIONS = [
    { key: 'giant',    label: 'Gigantism',      kind: 'mixed', w: 1,   fx: { frame: 1.4, strength: 1.25, metabolism: 1.25, speed: 0.92 }, text: 'Bigger and stronger, hungrier and slower.' },
    { key: 'denseBone',label: 'Dense bone',     kind: 'mixed', w: 1,   fx: { boneDensity: 1.6, speed: 0.92 }, text: 'Takes less injury, moves a little slower.' },
    { key: 'glass',    label: 'Brittle bone',   kind: 'bad',   w: 1,   fx: { boneDensity: 0.55 }, text: 'Injuries land harder.' },
    { key: 'pelt',     label: 'Thick pelt',     kind: 'good',  w: 0.6, fx: { insulation: 1.5, metabolism: 1.1 }, text: 'Stays warmer, eats a little more.' },
    { key: 'thin',     label: 'Thin blood',     kind: 'bad',   w: 1,   fx: { insulation: 0.6 }, text: 'Feels the cold sooner.' },
    { key: 'bellows',  label: 'Big lungs',      kind: 'good',  w: 0.6, fx: { lungs: 1.5, stamina: 1.2 }, text: 'Tires slowly.' },
    { key: 'ironGut',  label: 'Iron gut',       kind: 'good',  w: 0.6, fx: { resist: 1.4, immunity: 1.4 }, text: 'Shrugs off infection and venom.' },
    { key: 'slowBurn', label: 'Slow burn',      kind: 'good',  w: 0.7, fx: { metabolism: 0.75, fatStore: 1.3, speed: 0.95 }, text: 'Goes hungry slowly.' },
    { key: 'twitch',   label: 'Quick twitch',   kind: 'mixed', w: 0.8, fx: { speed: 1.25, stamina: 0.85, lifespan: 0.9 }, text: 'Fast, tires sooner, ages sooner.' },
    { key: 'elder',    label: 'Long-lived',     kind: 'mixed', w: 0.7, fx: { lifespan: 1.35, fertility: 0.8 }, text: 'Lives long, fewer births.' },
    { key: 'wits',     label: 'Sharp wits',     kind: 'mixed', w: 0.6, fx: { intel: 1.3, metabolism: 1.15 }, text: 'Learns and researches faster, burns more food.' },
    { key: 'frail',    label: 'Frail',          kind: 'bad',   w: 1,   fx: { healing: 0.6, immunity: 0.7 }, text: 'Recovers slowly.' }
  ];
  var MUT_BY = {}; MUTATIONS.forEach(function (m) { MUT_BY[m.key] = m; });
  var GENE_KEYS = ['strength', 'speed', 'metabolism', 'lifespan', 'resist', 'insulation', 'intel', 'frame', 'muscle', 'boneDensity', 'fatStore', 'lungs', 'immunity', 'stamina', 'hydration', 'healing', 'painTol', 'clotting', 'regen', 'shock', 'scarring', 'vision', 'hearing', 'smell', 'memory', 'focus', 'fertility', 'growth'];
  function clampG(k, v) { return k === 'intel' ? Math.min(3, Math.max(0.4, v)) : Math.min(4, Math.max(0.25, v)); }
  function applyMutation(g, key) {
    var m = MUT_BY[key], o = {}, k; for (k in g) o[k] = g[k]; if (!m) return o;
    for (k in m.fx) o[k] = clampG(k, (o[k] === undefined ? 1 : o[k]) * m.fx[k]); return o;
  }
  // roll new mutations: each has weight*rate chance (rate is per birth or per exposure); never double-rolls one a carrier already has
  function rollMutations(rng, rate, have) {
    var out = []; MUTATIONS.forEach(function (m) { if (have && have.indexOf(m.key) >= 0) return; if (rng() < m.w * rate) out.push(m.key); }); return out;
  }
  // child genes from two parents: each gene comes from one parent or the midpoint, with a little drift; inherited mutations copy the carrier's gene values

  /* ---------- functional genetics. Each of the 24 body genes is carried as two alleles; the body shows a dominance blend of the pair (DOM above 0.5: the higher
     allele dominates, below 0.5 the lower one does). A child takes one allele from each parent. Five recessive disorders ride along: a carrier has one copy and is well,
     two copies show the disorder. Related parents are likelier to both carry the same copy, so close kin produce affected children far more often (the inbreeding coefficient F is
     worked out from the shared ancestors of the parents). ---------- */
  var DOM = { immunity: 0.75, resist: 0.7, healing: 0.65, boneDensity: 0.65, lifespan: 0.6, stamina: 0.55, lungs: 0.55, metabolism: 0.35, hydration: 0.4, fatStore: 0.45, painTol: 0.5 };
  var DISORDERS = [
    { key: 'brittle', label: 'Brittle bones', fx: { boneDensity: 0.6 } }, { key: 'clot', label: 'Slow clotting', fx: { healing: 0.55 } },
    { key: 'lung', label: 'Weak lungs', fx: { lungs: 0.6, stamina: 0.8 } }, { key: 'chill', label: 'Cold intolerance', fx: { insulation: 0.6 } },
    { key: 'frail', label: 'Frail immunity', fx: { immunity: 0.55, resist: 0.75 } }
  ];
  function domMix(k, a, b) { var h = DOM[k] === undefined ? 0.5 : DOM[k], hi = Math.max(a, b), lo = Math.min(a, b); return h * hi + (1 - h) * lo; }
  function founderAlleles(g, r) { var o = {}; GENE_KEYS.forEach(function (k) { var v = g[k] === undefined ? 1 : g[k], e = 0.06 * (r() + r() - 1); o[k] = [clampG(k, v * (1 + e)), clampG(k, v * (1 - e))]; }); return o; }
  function founderDis(r, q) { q = q === undefined ? 0.08 : q; return DISORDERS.map(function () { return [r() < q ? 1 : 0, r() < q ? 1 : 0]; }); }
  function disAffected(dis) { return dis ? DISORDERS.filter(function (d, i) { return dis[i] && dis[i][0] && dis[i][1]; }) : []; }
  function disCarrier(dis) { return dis ? DISORDERS.filter(function (d, i) { return dis[i] && dis[i][0] + dis[i][1] === 1; }) : []; }
  function applyDis(g, dis) { var o = {}, k; for (k in g) o[k] = g[k]; disAffected(dis).forEach(function (d) { for (k in d.fx) o[k] = clampG(k, (o[k] === undefined ? 1 : o[k]) * d.fx[k]); }); return o; }
  /* inbreeding coefficient of a child of m and d: lin maps an ancestor id to the generations up from that person (self = 0) */
  function inbreeding(lm, ld) { var f = 0, id; if (!lm || !ld) return 0; for (id in lm) if (ld[id] !== undefined) f += Math.pow(0.5, lm[id] + ld[id] + 1); return f; }
  function linChild(m, d, id) { var o = {}, k; [m.lin || {}, d.lin || {}].forEach(function (L) { for (k in L) if (L[k] < 4 && (o[k] === undefined || L[k] + 1 < o[k])) o[k] = L[k] + 1; }); o[m.id] = Math.min(o[m.id] === undefined ? 1 : o[m.id], 1); o[d.id] = Math.min(o[d.id] === undefined ? 1 : o[d.id], 1); o[id] = 0; return o; }
  /* child of two parent objects ({g, ga, dis, lin, id, muts}): returns genes, alleles, disorders, lineage and F */
  function breed2(m, d, rng, opts) {
    opts = opts || {}; var drift = opts.drift === undefined ? 0.05 : opts.drift, rate = opts.rate === undefined ? 0.06 : opts.rate, g = {}, ga = {}, muts = [];
    var pm = m.ga || founderAlleles(m.g, rng), pd = d.ga || founderAlleles(d.g, rng);
    GENE_KEYS.forEach(function (k) {
      var a = pm[k][rng() < 0.5 ? 0 : 1], b = pd[k][rng() < 0.5 ? 0 : 1];
      a = clampG(k, a * Math.exp(drift * 0.6 * (rng() + rng() - 1))); b = clampG(k, b * Math.exp(drift * 0.6 * (rng() + rng() - 1)));
      ga[k] = [a, b]; g[k] = clampG(k, domMix(k, a, b));
    });
    [[m.muts || [], m.g], [d.muts || [], d.g]].forEach(function (pr) {
      pr[0].forEach(function (key) { if (muts.indexOf(key) >= 0 || rng() > 0.5) return; muts.push(key); var mu = MUT_BY[key]; if (mu) for (var k in mu.fx) { g[k] = clampG(k, (pr[1][k] === undefined ? 1 : pr[1][k]) * Math.exp(drift * (rng() - 0.5))); } });
    });
    rollMutations(rng, rate, muts).forEach(function (key) { muts.push(key); g = applyMutation(g, key); });
    var dm = m.dis || founderDis(rng), dd = d.dis || founderDis(rng), dis = DISORDERS.map(function (x, i) {
      var a = dm[i][rng() < 0.5 ? 0 : 1], b = dd[i][rng() < 0.5 ? 0 : 1]; if (rng() < 0.004) a = 1; if (rng() < 0.004) b = 1; return [a, b]; });
    var F = 0; if (m.lin && d.lin) F = inbreeding(m.lin, d.lin);
    g = applyDis(g, dis);
    if (F > 0) { var f = Math.min(0.5, F); g.immunity = clampG('immunity', g.immunity * (1 - 0.3 * f)); g.lifespan = clampG('lifespan', g.lifespan * (1 - 0.25 * f)); g.healing = clampG('healing', g.healing * (1 - 0.2 * f)); }
    return { g: g, ga: ga, muts: muts, dis: dis, F: F };
  }
  function skinOf(ge) { var G = parseGe(ge); if (!G) return 0.5; return (G[0] + G[1] + G[2] + G[3] + G[4] + G[5]) / 6; }
  function disRows(dis, F) {
    var rows = [], af = disAffected(dis), ca = disCarrier(dis);
    af.forEach(function (d) { rows.push({ k: d.label, v: 'affected (two copies)' }); });
    if (ca.length) rows.push({ k: 'Carries', v: ca.map(function (d) { return d.label.toLowerCase(); }).join(', ') });
    if (!af.length && !ca.length) rows.push({ k: 'Hereditary', v: 'no known disorders' });
    if (F > 0.001) rows.push({ k: 'Inbreeding F', v: F.toFixed(3) + (F >= 0.125 ? ' (high)' : F >= 0.0625 ? ' (raised)' : '') });
    return rows;
  }
  /* an animal coat's brightness (0 dark, 1 white), against a background brightness, decides how far away a hunter picks it out */
  function coatLum(c) { var f = coatPheno(c); return !f ? 0.4 : f.alb ? 0.93 : f.mel ? 0.1 : 0.4 + (f.tone - 0.5) * 0.4; }
  function visibility(c, bg) { var k = Math.abs(coatLum(c) - bg); return Math.max(0.7, Math.min(1.6, 1 + 1.2 * (k - 0.2))); }
  function breed(gA, gB, mA, mB, rng, opts) {
    opts = opts || {}; var drift = opts.drift === undefined ? 0.05 : opts.drift, rate = opts.rate === undefined ? 0.06 : opts.rate, g = {}, muts = [];
    GENE_KEYS.forEach(function (k) {
      var a = gA[k] === undefined ? 1 : gA[k], b = gB[k] === undefined ? 1 : gB[k], t = rng();
      var v = t < 0.45 ? a : t < 0.9 ? b : (a + b) / 2;
      g[k] = clampG(k, v * Math.exp(drift * (rng() + rng() + rng() - 1.5) * 1.6));
    });
    [[mA || [], gA], [mB || [], gB]].forEach(function (pr) {
      pr[0].forEach(function (key) { if (muts.indexOf(key) >= 0 || rng() > 0.5) return; muts.push(key); var m = MUT_BY[key]; if (m) for (var k in m.fx) g[k] = clampG(k, (pr[1][k] === undefined ? 1 : pr[1][k]) * Math.exp(drift * (rng() - 0.5))); });
    });
    rollMutations(rng, rate, muts).forEach(function (key) { muts.push(key); g = applyMutation(g, key); });
    return { g: g, muts: muts };
  }
  function mutationLines(keys) { return (keys || []).map(function (k) { return MUT_BY[k]; }).filter(Boolean); }

  /* ---------- LLM agents: each agonist can be driven by a language model. The system prompt is built from the same traits, personality, genes and
     mutations the sim models; the model answers with one goal from LLM_GOALS and the sim's own movement and action code carries it out. ---------- */
  var LLM_GOALS = [
    ['eat', 'Eat: use berries you carry, food on the camp table, a bush, or hunt if you are armed and hungry.'],
    ['hunt', 'Hunt: go after nearby prey with whatever weapon you have, even if you are not starving.'],
    ['drink', 'Drink: go to water, drink and refill your waterskin.'],
    ['sleep', 'Sleep: go to your bed (or lie down where you are) and recover energy.'],
    ['shelter', 'Shelter: get into a cave out of storms and bitter cold.'],
    ['warm', 'Warm: go to the hearth or a roofed floor and stay there until you are warm again.'],
    ['gather', 'Gather: pick berries, wood, stone, flint, fiber; carry them home.'],
    ['craft', 'Craft: work at the bench: make a weapon, armor or clothing, or work out new know-how.'],
    ['tame', 'Tame: approach a wild animal with berries and build its trust until it follows you.'],
    ['socialize', 'Socialize: walk to the nearest person and spend time together.'],
    ['wander', 'Wander: explore on foot.'],
    ['rest', 'Rest: stand still and catch your breath.']
  ];
  function pct(v) { return Math.round(Math.max(0, Math.min(1, v)) * 100); }
  var PERS_ADJ = [['bold', 'fearful'], ['sociable', 'solitary'], ['curious', 'incurious'], ['patient', 'impatient'], ['aggressive', 'gentle'], ['diligent', 'idle-minded'], ['empathic', 'callous'], ['greedy', 'generous']];
  function persLine(p) {
    var out = [];
    PERS_ADJ.forEach(function (pr, i) { var v = p[i] === undefined ? 0.5 : p[i]; if (v >= 0.7) out.push('very ' + pr[0]); else if (v >= 0.58) out.push('somewhat ' + pr[0]); else if (v <= 0.3) out.push('very ' + pr[1]); else if (v <= 0.42) out.push('somewhat ' + pr[1]); });
    return out.join(', ') || 'average in everything';
  }

  /* ---------- beliefs, social transmission, norms (design: design/belief_culture.md) ---------- */
  var MIND = (function () {
    var ST = { tries: 0, ex: 0, dv: 0, harm: 0 };
    function clamp01(x) { return x < 0 ? 0 : x > 1 ? 1 : x; }
    function bold(a) { return a.pers && a.pers[0] !== undefined ? a.pers[0] : 0.5; }
    function bv(a, k) { var b = a.bel && a.bel[k]; return b ? b.v : 0; }
    // expected harm of a context, scaled by temperament: bold people discount it, timid people inflate it
    function fear(a, k) { return clamp01(bv(a, k) * (0.7 + 0.6 * (1 - bold(a)))); }
    function ensure(a, k, tick, src) { if (!a.bel) a.bel = {}; var b = a.bel[k]; if (!b) b = a.bel[k] = { v: 0, n: 0, t: tick, src: src || 'self' }; return b; }
    // Rescorla-Wagner update toward outcome o with learning rate al
    function learn(a, k, o, al, tick, src) { var b = ensure(a, k, tick, src); b.v = clamp01(b.v + al * (o - b.v)); b.n += 1; b.t = tick; if (src === 'self') b.src = 'self'; return b; }
    function harmEv(a, k, sev, tick) { ST.harm++; a.survived = (a.survived || 0) + 0.2; return learn(a, k, clamp01(sev), 0.35 * (1 + 0.5 * (a.stress || 0)), tick, 'self'); }
    function safeEv(a, k, tick) { if (!a.bel || !a.bel[k]) return; learn(a, k, 0, 0.04, tick, 'self'); }
    function prior(a) { var cp = (typeof window !== 'undefined' && window.AGON_P && window.AGON_P.coldPrior != null) ? window.AGON_P.coldPrior : 0.12, p = cp * (1 - bold(a)); ensure(a, 'cold:night', 0, 'born').v = p; ensure(a, 'cold:night', 0, 'born').n = 0; }
    function decay(a) { if (!a.bel) return; for (var k in a.bel) { var b = a.bel[k]; if (b.src !== 'self' && b.v > 0) b.v *= 0.9995; } }
    // everyone alive and adult counts once; rank in [0,1] by age plus harms survived
    function prestigeOf(a, agents) { var sc = function (q) { return (q.age || 0) / 2000 + 20 * (q.survived || 0); }, me = sc(a), n = 0, lo = 0; for (var i = 0; i < agents.length; i++) { var q = agents[i]; if (!q.alive || q.fac || q.build === 'child') continue; n++; if (sc(q) < me) lo++; } return 0.6 + 0.4 * (n > 1 ? lo / (n - 1) : 0.5); }
    function trust(t, l) { if (t.mother && t.mother === l.id || l.mother && l.mother === t.id || t.father && t.father === l.id || l.father && l.father === t.id || t.mother && t.mother === l.mother) return 1; if (t.house && t.house === l.house) return 0.8; return 0.5; }
    // teller shares the belief it is surest of; deviants from a house norm are told first
    function exchange(t, l, tick, r, agents, norms) {
      ST.tries++; if (!t.bel) return null;
      var best = null, bs = 0.05;
      for (var k in t.bel) { var b = t.bel[k], w = b.v * Math.min(1, b.n / 3); if (norms && norms[k] && bv(l, k) < 0.15) w *= 3; if (w > bs) { bs = w; best = k; } }
      if (!best) return null;
      var tb = t.bel[best], lb = ensure(l, best, tick, 'told'), norm = norms && norms[best] ? 1.5 : 1;
      var beta = 0.25 * trust(t, l) * prestigeOf(t, agents) * (l.build === 'child' ? 1.5 : 1) * norm, vt = clamp01(tb.v + (r() + r() + r() - 1.5) * 0.1), before = lb.v;
      lb.v = clamp01(lb.v + beta * (vt - lb.v)); lb.t = tick; if (lb.src !== 'self') lb.src = 'told';
      ST.ex++; ST.dv += lb.v - before; return { k: best, d: lb.v - before };
    }
    function inherit(c, m, d, tick) {
      var ks = {}; [m, d].forEach(function (p) { if (p && p.bel) for (var k in p.bel) ks[k] = 1; });
      for (var k in ks) { var vm = m && m.bel && m.bel[k] ? m.bel[k].v : 0, vd = d && d.bel && d.bel[k] ? d.bel[k].v : 0, v = 0.5 * Math.max(vm, vd); if (v > 0.15) { var b = ensure(c, k, tick, 'born'); b.v = Math.max(b.v, clamp01(v)); b.n = 0; } }
    }
    // a norm is a belief most living adults of a house hold; returns { house: { key: {share, since} } }
    function norms(agents, tick, prev) {
      var by = {}, out = {}; agents.forEach(function (a) { if (!a.alive || a.build === 'child') return; var h = a.fac ? 'R' + a.fac : 'camp'; (by[h] = by[h] || []).push(a); });
      Object.keys(by).forEach(function (h) {
        var L = by[h], ks = {}; L.forEach(function (a) { if (a.bel) for (var k in a.bel) ks[k] = 1; }); out[h] = {};
        Object.keys(ks).forEach(function (k) { var n = 0; L.forEach(function (a) { if (bv(a, k) > 0.4) n++; }); var sh = n / L.length, was = prev && prev[h] && prev[h][k]; if (L.length >= 3 && (sh >= 0.6 || (was && sh >= 0.45))) out[h][k] = { share: Math.round(sh * 100) / 100, since: was ? was.since : tick }; });
      });
      return out;
    }
    function diverge(agents) {
      var by = {}; agents.forEach(function (a) { if (!a.alive || a.fac || !a.bel) return; var h = a.house || 'camp'; (by[h] = by[h] || []).push(a); });
      var hs = Object.keys(by); if (hs.length < 2) return 0;
      var vec = function (L) { var o = {}; L.forEach(function (a) { for (var k in a.bel) o[k] = (o[k] || 0) + a.bel[k].v / L.length; }); return o; }, va = vec(by[hs[0]]), vb = vec(by[hs[1]]), dot = 0, na = 0, nb = 0, ks = Object.assign({}, va, vb);
      for (var k in ks) { var x = va[k] || 0, y = vb[k] || 0; dot += x * y; na += x * x; nb += y * y; }
      return na && nb ? 1 - dot / Math.sqrt(na * nb) : 0;
    }
    // band-level divergence: cosine distance between the mean belief vectors of the player camp and each rival faction (0 = same culture, 1 = nothing shared)
    function divergeBands(agents) {
      var by = {}; agents.forEach(function (a) { if (!a.alive || !a.bel) return; var h = a.fac ? 'R' + a.fac : 'camp'; (by[h] = by[h] || []).push(a); });
      var vec = function (L) { var o = {}; L.forEach(function (a) { for (var k in a.bel) o[k] = (o[k] || 0) + a.bel[k].v / L.length; }); return o; }, out = {}, hs = Object.keys(by), vc = by.camp ? vec(by.camp) : null;
      hs.forEach(function (h) { if (h === 'camp' || !vc) return; var vb = vec(by[h]), dot = 0, na = 0, nb = 0, ks = Object.assign({}, vc, vb); for (var k in ks) { var x = vc[k] || 0, y = vb[k] || 0; dot += x * y; na += x * x; nb += y * y; } out[h] = na && nb ? +(1 - dot / Math.sqrt(na * nb)).toFixed(3) : null; });
      return out;
    }
    // regret matching over tied work goals: each colonist keeps a running estimate U[g] of how much better its body and output got over the 20 ticks it spent on goal g; the chance of picking g is proportional to its positive regret max(0, U[g] - mean U) plus a small exploration floor
    function regretUpdate(a, g, reward) { if (!a.rgU) a.rgU = {}; a.rgB = a.rgB === undefined ? reward : 0.95 * a.rgB + 0.05 * reward; reward -= a.rgB; var o = a.rgU[g]; a.rgU[g] = o === undefined ? reward : 0.8 * o + 0.2 * reward; }
    function regretPick(a, goals, r) {
      var U = a.rgU || {}, vals = goals.map(function (g) { return U[g] === undefined ? 0 : U[g]; }), m = vals.reduce(function (x, y) { return x + y; }, 0) / vals.length, w = vals.map(function (v) { return Math.max(0, v - m) + 0.05; }), tot = w.reduce(function (x, y) { return x + y; }, 0), x = r() * tot;
      for (var i = 0; i < goals.length; i++) { x -= w[i]; if (x <= 0) return goals[i]; } return goals[goals.length - 1];
    }
    // does following a custom pay? mean cold-harm ticks of colonists who hold the fear (followers) against those who do not (deviators)
    function obedience(agents, key) {
      var f = [], d = []; agents.forEach(function (a) { if (!a.alive || a.fac || a.build === 'child') return; (bv(a, key) > 0.4 ? f : d).push(a.cold25 || 0); });
      var mean = function (L) { return L.length ? L.reduce(function (x, y) { return x + y; }, 0) / L.length : null; };
      return { fol: mean(f), dev: mean(d), nf: f.length, nd: d.length };
    }
    function label(k) { var p = k.split(':'), n = { cold: 'the cold', pred: 'predators', pois: 'bad plants', fire: 'fire' }[p[0]] || p[0]; return n + (p[1] ? ' (' + p[1] + ')' : ''); }
    return { divergeBands: divergeBands, regretPick: regretPick, regretUpdate: regretUpdate, obedience: obedience, stats: ST, fear: fear, v: bv, harm: harmEv, safe: safeEv, prior: prior, decay: decay, exchange: exchange, inherit: inherit, norms: norms, diverge: diverge, label: label, prestige: prestigeOf };
  })();
  function llmSystem(a) {
    var g = a.g || {}, sf = strengthsFlaws(g, a.pers || []), muts = mutationLines(a.muts), tr = (a.traits || []).map(function (k) { for (var i = 0; i < TRAITS.length; i++) if (TRAITS[i].key === k) return TRAITS[i]; return null; }).filter(Boolean);
    var who = (a.name || a.id) + ', a ' + (a.build === 'child' ? 'child' : 'grown') + ' ' + (a.sex === 'F' ? 'woman' : 'man') + ' of about ' + (a.age / 2000).toFixed(1) + ' years';
    var L = [];
    L.push('You are ' + who + ', one of a small band of early people living at a camp with a hearth, beds, a table and a workbench. You have no memory of anything beyond what you are told each turn. Think and choose as this person would.');
    L.push('Your nature:');
    L.push('- Personality: ' + persLine(a.pers || []) + '.');
    L.push('- Traits: ' + (tr.length ? tr.map(function (t) { return t.label + ' (' + t.desc.replace(/\.$/, '') + ')'; }).join('; ') : 'none') + '.');
    L.push('- Body: ' + (sf.strengths.length ? 'strengths ' + sf.strengths.join(', ') : 'no standout strengths') + '; ' + (sf.flaws.length ? 'weaknesses ' + sf.flaws.join(', ') : 'no standout weaknesses') + '.');
    if (muts.length) L.push('- Mutations you were born with: ' + muts.map(function (m) { return m.label + ' (' + m.text.replace(/\.$/, '') + ')'; }).join('; ') + '.');
    L.push('- You know ' + (a.gear && a.gear.weapon ? 'how to use your ' + a.gear.weapon.label.toLowerCase() : 'only bare hands') + '.');
    L.push('Cold kills: at night and in winter your body loses heat fast. Your "warmth" reading goes comfortable, cold, freezing; frozen people take damage and can die. Warmth comes from the hearth, a roofed floor, a cave or clothing. If warmth reads cold or worse and you are not at a warm place, pick shelter or rest near the hearth before anything else. "coldFear" (0-100) is how afraid of the cold you have learned to be; the higher it is, the sooner you should head in.');
    L.push('Staying alive matters, but you are a person, not a machine: let your personality and traits shape what you do and why. If you are in real danger (starving, dying of thirst, freezing, badly hurt) your body will insist.');
    L.push('Each turn you receive a JSON description of what you feel and see. Pick exactly one goal from this list:');
    LLM_GOALS.forEach(function (q) { L.push('- ' + q[0] + ': ' + q[1].replace(/^[A-Za-z]+: /, '')); });
    L.push('Reply with ONLY a JSON object: {"goal": "<one goal>", "say": "<at most 8 words you say aloud, or empty>", "thought": "<one short first-person sentence>"}. No other text.');
    return L.join('\n');
  }
  function parseDecision(txt) {
    var m = /\{[\s\S]*\}/.exec(String(txt || '')); if (!m) return null;
    var o; try { o = JSON.parse(m[0]); } catch (e) { return null; }
    var ok = LLM_GOALS.some(function (q) { return q[0] === o.goal; }); if (!ok) return null;
    return { goal: o.goal, say: String(o.say || '').slice(0, 80), thought: String(o.thought || '').slice(0, 200) };
  }
  // offline stand-in for a model: reads the same observation and the persona, answers in the same JSON, after a short delay
  function mockDecide(a, obs) {
    var p = a.pers || [], y = obs.you || {}, sn = obs.seen || {}, c = obs.carrying || {}, pick = function (arr) { return arr[Math.floor(Math.random() * arr.length)]; };
    var goal = 'wander', say = '', th = '';
    if (y.energy < 25 || (obs.time === 'night' && y.energy < 75)) { goal = 'sleep'; th = 'I am worn out. Bed.'; }
    else if (y.thirst < 45) { goal = 'drink'; th = 'My mouth is dry.'; }
    else if (y.hunger < 55) { goal = (sn.prey && sn.prey.length && (c.weapon || p[4] > 0.6) && y.hunger < 40) ? 'hunt' : 'eat'; th = goal === 'hunt' ? 'Meat will do more than berries.' : 'I need to eat.'; }
    else if ((y.warmth === 'cold' || y.warmth === 'freezing') && obs.you && obs.warmPlace === false) { goal = 'warm'; th = 'I am cold. Back to the fire.'; }
    else if (obs.weather === 'storm' || (y.warmth === 'cold' && p[0] < 0.4)) { goal = 'shelter'; th = 'This weather is no place for me.'; }
    else if (c.berries >= 1 && sn.tameable && sn.tameable.length && p[6] > 0.45) { goal = 'tame'; th = 'That ' + sn.tameable[0].kind + ' might trust me.'; say = 'Easy now'; }
    else if (p[1] > 0.6 && sn.people && sn.people.length) { goal = 'socialize'; th = 'I would rather not be alone.'; say = 'Hello'; }
    else if (p[5] > 0.5 && c.materials > 0) { goal = 'craft'; th = 'Time to make something useful.'; }
    else if (p[5] > 0.4 && sn.berryBushes && sn.berryBushes.count) { goal = 'gather'; th = 'There is food to bring home.'; }
    else if (p[2] > 0.6) { goal = pick(['wander', 'craft']); th = 'I wonder what is over there.'; }
    else { goal = pick(['gather', 'rest', 'wander']); th = 'Nothing pressing.'; }
    var delay = 150 + Math.random() * 450;
    return new Promise(function (res) { setTimeout(function () { res(JSON.stringify({ goal: goal, say: say, thought: th })); }, delay); });
  }
  function geneDetail(it) {
    // it: { key, temp, v, idx }
    if (it.temp) return { lines: [['Share of the camp norm', sg(it.v / 0.5 - 1)], ['Tends to', PERS_FX[it.idx] || '']] };
    var fx = GENE_FX[it.key] || [];
    return { lines: [['Versus species average', sg(it.v - 1)]].concat(fx.map(function (f) { return [f[0], f[1](it.v)]; })).concat([['Typical range', 'x0.25 to x4.00']]) };
  }
  var THOUGHT_ICONS = { food: ['hungry'], water: ['thirsty'], rest: ['sleep'], safety: ['scared'], warmth: ['cold'], social: ['love'], work: ['craft'], wander: ['arrow'] };
  function thoughtIcons(key, c) {
    var o = (THOUGHT_ICONS[key] || ['question']).slice();
    if (key === 'safety' && c.pred) o.push('flee'); if (key === 'safety' && !c.pred && c.injured) o = ['hurt']; if (key === 'safety' && c.infected) o = ['sick'];
    if (key === 'warmth' && c.temp > 0) o = ['hot']; if (key === 'food' && c.berries > 0) o.push('eat'); if (key === 'water') o.push('drink');
    if (key === 'work' && c.research) o = ['idea']; if (key === 'rest' && !c.sleeping && c.bed >= 0) o.push('home'); if (key === 'social' && !c.near.length) o = ['question'];
    return o;
  }

  /* ---------- hover information: one glossary, matched by the visible label, with live numbers for the selected agonist ----------
     An entry is { t: title, k: kind, b: body, r: [rule lines], by: [affected by], then: [consequences], live: function (c) -> [lines] }.
     c is { sel: agent or null, tick, known: [tech keys], wp: work priorities }. Rates are per tick (600 ms). 50 ticks = 1 day, 2000 ticks = 1 year. */
  var TIPS = {};
  var KINDCOL = { Need: '#e0956a', Mind: '#d45ee0', Gene: '#d4c63c', Trait: '#e08aa8', Order: '#7fbf8e', Work: '#7fbf8e', Tech: '#7cacdf', Move: '#e0504a', Gear: '#c9a45a', Info: '#aea69a', Tool: '#ffd23f', World: '#6fcf8a', Social: '#ffb8d0', Material: '#b98a5a', Body: '#b57e55' };
  function tkTime(n) { return n < 1 ? 'now' : Math.round(n) + ' ticks (' + (n / 50).toFixed(1) + ' days)'; }
  function pc(v) { return Math.round((v || 0) * 100) + '%'; }
  function T(keys, t, k, b, x) { var e = { t: t, k: k, b: b }; if (x) for (var i in x) e[i] = x[i]; (Array.isArray(keys) ? keys : [keys]).forEach(function (key) { TIPS[key] = e; }); }
  function burnOf(a, which) {
    var g = a.g || {};
    if (which === 'sat') return 0.01 * (g.metabolism || 1) * traitFx(a, 'satK') / Math.sqrt(gv(g, 'fatStore'));
    if (which === 'hyd') return 0.015 * traitFx(a, 'hydK') / gv(g, 'hydration');
    return 0.005 / (gv(g, 'stamina') * Math.sqrt(gv(g, 'lungs')));
  }
  function liveNeed(which, label) {
    return function (c) {
      var a = c.sel; if (!a || a[which] === undefined) return [];
      var b = burnOf(a, which), out = [label + ': ' + pc(a[which]), 'Burn for this agonist: ' + (b * 100).toFixed(2) + '% per tick'];
      if (b > 0 && a[which] > 0) out.push('Empty in ' + tkTime(a[which] / b) + ' if nothing changes');
      return out;
    };
  }
  /* needs and mind */
  T(['satiety', 'hungry'], 'Satiety', 'Need', 'How fed the agonist is. 100% is full, 0% is starving.', {
    r: ['Burns 1.0% per tick at average metabolism: empty in about 100 ticks (2 days).', 'Berries +30%, bush +35%, hunted meat +45%, table food +30%.'],
    by: ['Metabolism gene (faster burn)', 'Fat reserve gene (slower burn)', 'Glutton and Ascetic traits'],
    then: ['At 0% health drains 2% per tick and stress climbs 5% per tick.'], live: liveNeed('sat', 'Satiety') });
  T(['hydration', 'thirsty'], 'Hydration', 'Need', 'How much water the agonist has. Drains faster than food.', {
    r: ['Burns 1.5% per tick at average: empty in about 67 ticks (1.3 days).', 'Drinking at water +50% and refills the waterskin. Drinking from the waterskin +40% and uses 35% of it.'],
    by: ['Water economy gene', 'Ascetic trait'], then: ['At 0% health drains 2% per tick.'], live: liveNeed('hyd', 'Hydration') });
  T(['energy', 'exhausted'], 'Energy', 'Need', 'Tiredness. Recovered by sleeping, resting and idling.', {
    r: ['Drains 0.5% per tick at average: empty in about 200 ticks (4 days).', 'Sleeping +6% per tick (and +0.4% health). Idling, resting or sheltering +1% per tick.', 'Dashing and combat moves spend extra energy.'],
    by: ['Stamina gene', 'Lung capacity gene'], then: ['At 0% health drains 2% per tick.'], live: liveNeed('ene', 'Energy') });
  T(['health', 'failing'], 'Health', 'Need', 'Total bodily integrity. At 0% the agonist dies.', {
    r: ['Heals +0.2% per tick (x Healing gene) while satiety, hydration and energy are all above 0. Asleep it heals +0.4% more.', 'Drains: starvation 2% per tick, infection 0.3% per tick, freezing 0.4% per tick, standing in fire 8% per hit, lightning 60%, plus blows from fights and animals.'],
    by: ['Healing gene', 'Immunity and Resistance genes', 'Injury multipliers (Tough, Fragile, Pain tolerance, Bone density)'],
    live: function (c) { var a = c.sel; return a && a.hea !== undefined ? ['Health: ' + pc(a.hea), a.infection ? 'Infected: losing 0.3% per tick' : 'No infection'] : []; } });
  T(['warmth', 'cold', 'freezing', 'heat', 'overheated'], 'Warmth', 'Need', 'Distance from a comfortable body temperature. The bar drops when the agonist is too cold or too hot.', {
    r: ['Below -0.7 on the temperature scale health drains 0.4% per tick and stress climbs.', 'Comfort feeds happiness: comfort = 1 - 1.5 x |temperature offset|.'],
    by: ['Insulation gene', 'Clothing, armor and materials (Fur 5, Hide 2, Wood 1 insulation)', 'Fire, shelter and the season', 'Night and weather'],
    live: function (c) { var a = c.sel; return a && a.temp !== undefined ? ['Temperature offset: ' + (a.temp >= 0 ? '+' : '') + a.temp.toFixed(2) + (a.temp < -0.7 ? ' (freezing)' : a.temp > 0.7 ? ' (overheating)' : ' (ok)')] : []; } });
  T(['stress', 'panicked', 'calm'], 'Stress', 'Mind', 'Short-term alarm. Spikes from danger and pain and fades over a few days. Calm is its inverse.', {
    r: ['Falls 1% per tick (divided by the stress factor trait).', 'Raised by: damage +40%, fire +30%, starvation +5% per tick, fleeing +10%.', 'Lowered by company, parleys and trades (-5% to -10% each).'],
    by: ['Traits that change stress factor', 'Bravery and Empathy personality'],
    then: ['Above 50% it drains sanity.', 'Lowers the happiness target by 20% x stress.', 'Above 60% the face turns scared.'],
    live: function (c) { var a = c.sel; return a && a.stress !== undefined ? ['Stress: ' + pc(a.stress), 'Calm: ' + pc(1 - a.stress)] : []; } });
  T(['injury', 'wounded'], 'Injury', 'Need', 'Lingering wounds. Separate from health: you can be healthy and still carry injuries that heal slowly.', {
    r: ['Each blow adds injury (damage x 4). It heals only 0.1% per tick x Healing gene, so it lasts days.', 'Above 20% the agonist is marked Injured. Above 45% the face turns to pain. Above 50% it drains sanity.'],
    by: ['Tough, Fragile traits', 'Pain tolerance and Bone density genes', 'Armor'],
    live: function (c) { var a = c.sel; return a && a.inj !== undefined ? ['Injury: ' + pc(a.inj) + (a.inj > 0 ? '; clears in ~' + tkTime(a.inj / (0.001 * gv(a.g, 'healing'))) : '')] : []; } });
  T('sanity', 'Sanity', 'Mind', 'Long-term mental footing. Moves slowly, over days.', {
    r: ['Recovers: +0.15% per tick when stress is under 30%, +0.6% asleep, +0.2% when happy above 70%, +0.1% each for company, pets and shelter.', 'Drains: stress above 50%, grief for a death nearby, a night alone outside camp, injury or infection, going hungry or thirsty (under 10%).'],
    then: ['Below 30% it adds stress. Below 20% the face turns scared and the agonist may bolt.'], live: function (c) { var a = c.sel; return a && a.san !== undefined ? ['Sanity: ' + pc(a.san)] : []; } });
  T('happiness', 'Happiness', 'Mind', 'Long-run mood. It chases a target at 2% per tick rather than jumping.', {
    r: ['Target = 25% satiety + 15% hydration + 15% energy + 20% comfort + 15% (1 - injury)', '+ company (up to 12%, scaled by Sociability) + shelter 6% + a pet 8%', '- grief 15% - 20% x stress - infection 10%'],
    then: ['Above 70% it helps sanity. Below 50% it blocks the happy face.'], live: function (c) { var a = c.sel; return a && a.hap !== undefined ? ['Happiness: ' + pc(a.hap)] : []; } });
  T('infection', 'Infection', 'Need', 'A wound that has gone bad.', { r: ['Drains 0.3% health and adds a little stress every tick.', 'Clears with a 2% chance per tick x Resistance x Immunity, doubled while asleep.'], by: ['Resistance and Immunity genes', 'Sleep'] });
  T('mood', 'Mood', 'Mind', 'The facial expression, chosen by the most pressing state in this order:', { r: ['dead, asleep, hurt (injury above 45%), scared (stress above 60% or sanity under 20%), angry (mid-strike), sick, cold, thirsty (under 30%), hungry (under 30%), tired (under 30%), happy, else neutral.'], live: function (c) { return c.sel ? ['Right now: ' + expressionFor(c.sel)] : []; } });
  T('no bed', 'No bed', 'Info', 'This agonist has no assigned bed and sleeps rough.', { then: ['No shelter bonus to happiness while sleeping outside the camp floors.'] });
  T('injured', 'Injured', 'Info', 'Injury is above 20%. See Injury for how it heals.');
  T('dead', 'Dead', 'Info', 'Died. The body decays over time: flesh, then bones.', { r: ['Nearby agonists grieve for about 150 ticks (3 days): happiness -15% and sanity drain.'] });
  /* map labels */
  T('day', 'Day and time', 'World', 'The sun position drives light, darkness, temperature and who is awake. 50 ticks make a day; a tick is 600 ms at 1x.');
  T('tick', 'Tick', 'World', 'One simulation step (600 ms at 1x). 50 ticks = 1 day, 2000 ticks = 1 year (40 days).');
  /* profile */
  T('age', 'Age', 'Body', 'How old the agonist is in simulated years. 1 year is 2000 ticks (40 days).', { r: ['Percent of lifespan is shown next to it; the Lifespan gene sets the total. The base lifespan is 1000 simulated years.', 'Children are smaller and weaker until grown.'] });
  T('trained age', 'Trained age', 'Body', 'Experience gained by crafting, in years.', { r: ['Each craft adds 20 ticks of training, scaled by the learning traits.'], by: ['Fast learner / Slow learner', 'Intellect gene'] });
  T('summed age', 'Summed age', 'Body', 'Age plus trained age.');
  T('name', 'Name', 'Body', 'Names are given when an agonist is recognised as a person; founders start unnamed (ID only).');
  T('height', 'Height', 'Body', 'Standing height in centimetres.', { by: ['Frame size gene', 'Body scale and build', 'Age (children are short)'] });
  T('sex', 'Sex', 'Body', 'Biological sex. It only matters for births (Breed selected); it does not change stats.');
  T(['gender', 'sexuality'], 'Identity', 'Body', 'Hidden by default (sandbox). Use Show in the Sandbox section to reveal it. It has no mechanical effect on survival right now.');
  T('colony', 'Colony', 'Social', 'Which group the agonist belongs to. Home is yours; R01 to R04 are rival colonies that act on their own.');
  T('archetype', 'Archetype', 'Body', 'The two roles this body and mind are best suited to, ranked from a score:', { r: ['Hunter: strength + 0.5 speed + 2 bravery + 1.5 aggression', 'Gatherer: 0.6 metabolism + 2 patience + 1.5 diligence + greed', 'Crafter: intellect + 2.5 diligence + patience', 'Scout: 1.4 speed + 2 curiosity + bravery', 'Guardian: resistance + 2 bravery + 1.5 empathy', 'Healer: 0.8 intellect + 2.5 empathy + 1.2 patience'], then: ['Archetype sets the default work priorities.'] });
  T('strengths', 'Strengths', 'Body', 'Genes and temperament that are clearly above average.', { r: ['Strength, speed, resistance or insulation 1.3x or more. Efficient metabolism (0.8x or less). Long-lived. Intellect 1.4x or more. Diligence, bravery, patience or empathy at 75% or more.'] });
  T('flaws', 'Flaws', 'Body', 'Genes and temperament that are clearly below average.', { r: ['Strength or speed 0.75x or less, resistance 0.7x or less, ravenous metabolism, short-lived, low intellect, or diligence, bravery, patience, empathy at 25% or less.'] });
  T('job', 'Job', 'Work', 'What the agonist is doing next. An active order wins; otherwise the three highest work priorities are listed (1 is highest, 0 is never).');
  T('experience', 'Experience', 'Work', 'After 3 completed tasks it names the job with the most tasks done. Before that it shows the archetype default.');
  T('healthy', 'Healthy', 'Body', 'Yes only if health is above 80%, there is no infection and injury is under 20%.');
  T(['father', 'mother'], 'Parents', 'Body', 'Who the agonist was born to. Founders have none. Genes are mixed from both parents with a chance of mutation.');
  T('lineage', 'Lineage', 'Body', 'Generation count and the founder line. Generation 0 is a founder.');
  T('mutations', 'Mutations', 'Body', 'Random gene changes that arise in newborns and are inherited. They can help, hurt or do both.');
  T('traits', 'Traits', 'Mind', 'Fixed quirks that multiply specific rates (injury taken, hunger, learning speed and so on). Opposites cannot be held together.');
  /* orders and work */
  T('orders', 'Orders', 'Order', 'Tell this agonist what to do instead of following work priorities. An order lasts about 60 ticks (a day and a bit), then they go back to normal.');
  T('hunt', 'Hunt', 'Order', 'Chase and kill the nearest suitable animal. Meat gives +45% satiety and hide, bone and sinew. Fighting costs injury if the animal fights back.');
  T('gather', 'Gather', 'Order', 'Collect berries and fiber from bushes and grass into the pack and store.');
  T('haul', 'Haul', 'Order', 'Carry materials and food from the field to the camp store. Capacity grows with strength.');
  T('craft', 'Craft', 'Order', 'Make a weapon, armor or clothing from known schematics using stored materials. Quality improves with trained age.');
  T('tame', 'Tame', 'Order', 'Befriend a nearby animal so it follows and is owned. A pet adds +8% to the happiness target.');
  T('research', 'Research', 'Work', 'Contribute insight toward new technology (emergent mode only).');
  T('socialize', 'Socialize', 'Order', 'Walk to others and talk. Lowers stress 5% to 10%, trades food and builds relationship.');
  T('rest', 'Rest', 'Order', 'Stay put and recover energy (+1% per tick idling, +6% asleep).');
  T('go to...', 'Go to', 'Order', 'Arm it, then click the map. The agonist walks there and holds.');
  T('cancel', 'Cancel order', 'Order', 'Clear the current order and return to work priorities.');
  T('work priorities', 'Work priorities', 'Work', 'Per agonist and job: 1 (do first) to 4, or 0 for never. Defaults come from the two best archetypes.');
  /* toolbar */
  T('inspect', 'Inspect', 'Tool', 'Default view. Click an agonist to read and order it.');
  T('topology', 'Topology', 'Tool', 'Shows terrain height, water flow and walkable regions.');
  T('x-ray', 'X-ray', 'Tool', 'Shows the skeleton, organs, injuries and carried items beneath the skin.');
  T('follow', 'Follow', 'Tool', 'Keep the camera centred on this agonist as it moves.');
  T(['world size', 'small', 'medium', 'large', 'xl', 'xxl'], 'World size', 'World', 'Number of tiles. Bigger worlds hold more resources, animals and rival colonies and cost more to simulate.');
  T(['biome', 'temperate', 'forest', 'grassland', 'desert', 'tundra', 'highlands', 'wetland'], 'Biome', 'World', 'Sets terrain mix, water, food and temperature. Desert is dry and hot, tundra is cold, wetland has lots of water, highlands have stone and hills.');
  T('formation scale', 'Formation scale', 'World', 'Larger values make bigger, smoother landforms (lakes, forests, hills).');
  T(['zoom', 'fit'], 'Zoom', 'Tool', 'Mouse wheel zooms. WASD pans (and zooms in first if fit). Q and E zoom out and in. Any pan stops following.');
  T('planet', 'Planet', 'Tool', 'Open the planet view of this world.');
  T('border war', 'Border war', 'Tool', 'Start a hostile border scenario with a rival colony to test deliberate fights, raids, parleys and fleeing.');
  T('neighbours', 'Neighbours', 'Social', 'Rival colonies near yours. The bar is relationship score from -100 (war) to +100 (friends); clashes, trades and kills move it.');
  T('war chronicle', 'War chronicle', 'Social', 'Log of fights, raids, kills and parleys between colonies, with who started them.');
  T(['particles', 'text', 'icons'], 'Cues in game', 'Tool', 'How needs are shown on the map: particles around the agonist, floating text, and icons above the head.');
  /* thoughts */
  var TH = { food: 'Looking for or eating food.', water: 'Looking for or drinking water.', rest: 'Wants to sleep or sit down.', safety: 'Avoiding danger, fighting or fleeing.', warmth: 'Seeking fire, shelter or clothing.', social: 'Wants company or to talk.', work: 'Doing assigned work.', wander: 'Nothing pressing; exploring.' };
  THOUGHTS.forEach(function (t) {
    var lab = t.label.toLowerCase(), prev = TIPS[lab], note = TH[t.key] + ' Each tick the agonist weighs eight drives and acts on the strongest.';
    if (prev) { var cp = {}; for (var kk in prev) cp[kk] = prev[kk]; cp.b = prev.b + ' As a thought: ' + note; TIPS[lab] = cp; return; }
    T(lab, t.label, 'Mind', note, { live: function (c) { return c.sel && c.sel.thoughts && c.sel.thoughts[t.key] !== undefined ? [t.label + ' weight: ' + c.sel.thoughts[t.key].toFixed(2)] : []; } }); });
  T('current thoughts', 'Current thoughts', 'Mind', 'The weight of each drive right now. The tallest bar decides what the agonist does.');
  T(['live', 'history'], 'Live / History', 'Mind', 'Live shows this tick. History averages the recent past to show what dominates over time.');
  T(['bars', 'pie'], 'Chart style', 'Tool', 'Switch between bar and pie views of the same data.');
  T('inner voice', 'Inner voice', 'Mind', 'Short log of what the agonist decided and why, as icons, text or both.');
  T('decision engine', 'Decision engine', 'Tool', 'Rules uses the built-in utility rules. LLM agents lets a language model choose actions.');
  /* carry and gear */
  T('carrying', 'Carrying', 'Gear', 'Everything on the agonist: tool, berries, waterskin, materials. Weight is limited by strength.');
  T(['hands', 'belt', 'pack'], 'Carry slots', 'Gear', 'Hands hold a weapon or tool, the belt holds small items, the pack holds food and materials.');
  T('waterskin', 'Waterskin', 'Gear', 'Portable water. Refills at any water tile; each drink uses 35% and gives +40% hydration.');
  T('materials', 'Materials', 'Gear', 'Raw crafting stock carried by this agonist. See each material for hardness, sharpness and insulation.');
  T('equipment', 'Equipment', 'Gear', 'Worn and held gear. Each piece shows its stats, effects and remaining durability.');
  T('weapon', 'Weapon', 'Gear', 'Held item. Sets which fight moves are possible (club, spear, axe, knife, bow) and damage, reach and accuracy.');
  T('armor', 'Armor', 'Gear', 'Reduces damage from blows; a shield also unlocks guard moves.');
  T('clothing', 'Clothing', 'Gear', 'Adds insulation and rain cover; some pieces add carry or speed effects.');
  T('stats', 'Stats', 'Gene', 'The six main genes as a multiplier chart. 1.0x is average; the ring is log scale from 0.25x to 4x.');
  T(['hexagon', 'helix', 'needs'], 'Stat view', 'Tool', 'Hexagon shows six core genes, Helix lists all genes with modifiers, Needs shows current need levels.');
  T('personality', 'Personality', 'Mind', 'Eight temperament dials from 0 to 100%. They tilt decisions, fights and work choice.');
  /* personality */
  var PD = { Bravery: 'Willingness to face danger. Raises hunting and fighting odds, lowers fleeing.', Sociability: 'Wants company, trades and talks. Scales how much company lifts happiness.', Curiosity: 'Explores and tries new things. Drives scouting and insight.', Patience: 'Tolerance for waiting and being provoked. Lowers anger build-up.', Aggression: 'Anger build-up and readiness to attack. Pushes impulse fights.', Diligence: 'Sticks to assigned work. Higher means fewer idle ticks.', Empathy: 'Cares about others. Makes sparing, parleying and helping more likely.', Greed: 'Wants resources. Raises raid and hoarding utility.' };
  PERS.forEach(function (n, i) { T(n.toLowerCase(), n, 'Mind', PD[n], { live: function (c) { return c.sel && c.sel.pers ? [n + ': ' + pc(c.sel.pers[i])] : []; } }); });
  /* genes */
  var GD = { frame: 'Overall body size.', muscle: 'Muscle mass (latent: not yet affecting anything).', boneDensity: 'Dense bone resists injury.', fatStore: 'Body fat buys time between meals.', strength: 'Damage in fights, hauling and building.', speed: 'Movement speed on the map.', resist: 'Fights off infection and venom.', insulation: 'Keeps body heat in and out.', metabolism: 'Food burn rate. Higher burns faster.', lifespan: 'How many years until old age.', stamina: 'Slower energy drain.', lungs: 'Slower energy drain, helps endurance.', hydration: 'Slower thirst.', healing: 'Faster health and injury recovery.', painTol: 'Less injury from each blow.', immunity: 'Lower infection chance, faster recovery.', vision: 'Latent.', hearing: 'Latent.', smell: 'Latent.', intel: 'Faster schematic discovery and learning.', memory: 'Latent.', focus: 'Latent.', fertility: 'Latent.', growth: 'Latent.' };
  GENES.forEach(function (g) {
    var lab = g.label.toLowerCase();
    T(lab, g.label, 'Gene', (GD[g.key] || g.drives) + ' ' + (g.on ? '' : 'Latent: not yet wired to the sim.'), {
      r: ['Group: ' + g.group + '. Drives: ' + g.drives + '.', 'Genes are multipliers around 1.0x. Offspring inherit a mix of both parents, with a chance of mutation.'],
      live: function (c) { return c.sel && c.sel.g ? [g.label + ': x' + gv(c.sel.g, g.key).toFixed(2)] : []; } });
  });
  /* traits, mutations */
  TRAITS.forEach(function (t) { T(t.label.toLowerCase(), t.label, 'Trait', t.desc, { r: t.vs && t.vs.length ? ['Cannot be held with: ' + t.vs.join(', ')] : [] }); });
  MUTATIONS.forEach(function (m) { T(m.label.toLowerCase(), m.label, 'Gene', m.text, { r: ['Kind: ' + m.kind + '. Changes: ' + Object.keys(m.fx || {}).map(function (k) { return k + ' x' + m.fx[k]; }).join(', ')] }); });
  /* tech */
  TECH.forEach(function (t) {
    T(t.label.toLowerCase(), t.label, 'Tech', t.text, {
      r: ['Tier ' + t.tier + '. Cost ' + t.cost + ' insight. Requires: ' + (t.req.length ? t.req.join(', ') : 'nothing') + '.', t.schem.length ? 'Unlocks schematics: ' + t.schem.join(', ') : '', (t.machines || []).length ? 'Machines: ' + t.machines.join(', ') : '', (t.moves || []).length ? 'Combat primitives: ' + t.moves.join(', ') : ''].filter(Boolean),
      live: function (c) { return c.known ? [c.known.indexOf(t.key) >= 0 ? 'Known' : 'Not yet known'] : []; } });
  });
  T('technology', 'Technology', 'Tech', 'A tree of primitives. Preset mode knows everything; Emergent mode researches nodes by spending insight.');
  T(['schematic discovery', 'preset', 'emergent'], 'Schematic discovery', 'Tech', 'Preset: every schematic is known from the start. Emergent: agonists discover them from insight earned by crafting and exploring.');
  /* combat */
  T('combat moves (primitives > moves > styles)', 'Combat moves', 'Move', 'Primitives (step, dash, jab, swing) chain into moves; moves are chosen by style (brawler, duelist, skirmisher, guardian) from personality and gear.');
  Object.keys(MOVES).forEach(function (k) {
    var m = MOVES[k], q = m.req || {};
    T(m.label.toLowerCase(), m.label, 'Move', 'Sequence: ' + m.seq.map(function (p) { return PRIM[p] ? PRIM[p].label.toLowerCase() : p; }).join(' then ') + '.', {
      r: m.seq.map(function (p) { return PRIM[p] ? PRIM[p].label + ': ' + PRIM[p].text : p; }).concat([(q.weapon ? 'Needs weapon: ' + q.weapon.join(' or ') + '. ' : '') + (q.armor ? 'Needs armor: ' + q.armor.join(', ') + '. ' : '') + (q.tech ? 'Needs tech: ' + q.tech + '.' : '')].filter(function (s) { return s.trim(); })) });
  });
  Object.keys(PRIM).forEach(function (k) { var p = PRIM[k]; if (!TIPS[p.label.toLowerCase()]) T(p.label.toLowerCase(), p.label, 'Move', p.text, { r: ['Category: ' + p.cat + '. Energy cost ' + (p.cost * 100).toFixed(2) + '% per use.'] }); });
  STYLES && Object.keys(STYLES).forEach(function (k) { var s = STYLES[k]; T(s.label.toLowerCase(), s.label + ' style', 'Move', 'Favoured moves: ' + Object.keys(s.w).map(function (m) { return MOVES[m] ? MOVES[m].label : m; }).join(', ') + '.'); });
  /* materials */
  MAT_ORDER.forEach(function (k) { var m = MATS[k]; T(m.label.toLowerCase(), m.label, 'Material', 'Source: ' + m.src + '.', { r: ['Hardness ' + m.hard + ', sharpness ' + m.sharp + ', weight ' + m.weight + ', flex ' + m.flex, 'Insulation ' + m.ins + ', rain cover ' + m.rain + ', durability ' + m.dur] }); });
  /* sandbox */
  T('sandbox', 'Sandbox', 'Tool', 'Tools for testing: spawn agonists, animals, materials, insight, and force mutations or births.');
  T(['+ agent', '+ prey', '+ predator', '+ materials', '+ insight', 'equip selected', 'unlock next', '+ mutation', 'tame nearest', 'breed selected'], 'Sandbox action', 'Tool', 'Adds or changes things in the world right now for testing. It cannot be undone except by rewinding the timeline.');
  T(['intelligence scale', 'intelligence scale x1.00'], 'Intelligence scale', 'Tool', 'Multiplies how fast agonists learn and discover schematics.');
  T(['pause', 'resume'], 'Pause / Resume', 'Tool', 'Stops or restarts time. Space does the same.');
  T(['gone', 'latest'], 'Timeline', 'Tool', 'The last 300 ticks are recorded. Drag the slider to rewind; advancing from the past starts a new branch.');
  /* weather, seasons, fights, body parts, misc */
  [['spring', 'Spring', 'Season 1 of 4. Each season lasts 500 ticks (10 days); a year is 2000 ticks.'], ['summer', 'Summer', 'Season 2 of 4. Warmest part of the year (500 ticks).'], ['autumn', 'Autumn', 'Season 3 of 4. Temperatures fall (500 ticks).'], ['winter', 'Winter', 'Season 4 of 4. Coldest part of the year; insulation and fire matter most (500 ticks).']].forEach(function (a) { T(a[0], a[1], 'World', a[2]); });
  T('clear', 'Clear weather', 'World', 'Weather changes every 160 ticks (3.2 days) in a fixed cycle: clear, rain, clear, drought, storm, cold snap, clear, heat wave. Clear has no special effect.');
  T('rain', 'Rain', 'World', 'Wet weather. Fires burn out faster and spread at 30% of the normal chance.', { r: ['Cycle: clear, rain, clear, drought, storm, cold snap, clear, heat wave (160 ticks each).'] });
  T('storm', 'Storm', 'World', 'Wet weather that can also bring lightning, which can strike an agonist for 60% health and 50% injury.', { r: ['Fires still burn out faster, as in rain.'] });
  T('drought', 'Drought', 'World', 'Dry weather. Fire spreads 1.6x easier and random fires can start (3% per tick somewhere on the map).');
  T('heat wave', 'Heat wave', 'World', 'Dry and hot. Same fire risk as drought; heat raises the temperature offset, so insulation works against you.');
  T('cold snap', 'Cold snap', 'World', 'Sharp temperature drop. Below -0.7 offset agonists lose health to the cold, so shelter, fire and clothing matter.');
  T(['generator', 'character generator'], 'Generator', 'Tool', 'Open the character generator to design genes and appearance for founders.');
  [['attack', 'Attack', 'Decided to fight. The move shown after the dot is the current fight move. An exclamation mark means it was an impulse (anger above threshold), not a reasoned choice.'], ['raid', 'Raid', 'Going into a rival camp to take food or materials. Chosen when greed and need outweigh the risk.'], ['defend', 'Defend', 'Joining a fight to protect an ally or the camp.'], ['parley', 'Parley', 'Trying to talk or trade instead of fight. Success lowers stress and moves the relationship toward friendship; rebuffed overtures count against it.']].forEach(function (a) { T(a[0], a[1], 'Social', a[2], { r: ['Chosen by comparing utility of ignore, attack, raid, flee and parley, using anger, need, the power ratio of nearby allies and personality (bravery, aggression, patience, empathy, greed).'] }); });
  [['skin', 'Skin', 'Body', 'Outer covering. Shows skin tone and surface wounds; it is the first layer to take damage and carries insulation.'], ['torso', 'Torso', 'Body', 'Trunk and internal organs. Carries the body temperature and food and water status shown here.'], ['feet', 'Feet', 'Body', 'Footwear and walking. Boots and greaves cover them.'], ['left hand', 'Left hand', 'Body', 'Free hand or shield hand.'], ['right hand', 'Right hand', 'Body', 'Holds the weapon or tool.'], ['head', 'Head', 'Body', 'Skull and brain. Helms and hats go here.'], ['ears', 'Ears', 'Body', 'Hearing (latent gene, not yet used by the sim).'], ['hair', 'Hair', 'Body', 'Cosmetic: style, colour and clothing details come from the appearance genes.'], ['eyes', 'Eyes', 'Body', 'Vision (latent gene). Expressions show here.'], ['nose', 'Nose', 'Body', 'Smell (latent gene).'], ['mouth', 'Mouth', 'Body', 'Eating, drinking and expression.']].forEach(function (a) { T(a[0], a[1], a[2], a[3]); });
  T(['rules', 'rules engine'], 'Rules engine', 'Tool', 'Agonists choose actions with the built-in utility rules: each drive gets a weight from needs and personality and the strongest wins.');
  T(['llm agents', 'llm'], 'LLM agents', 'Tool', 'A language model picks actions instead of the rules. Needs a configured model and key.');
  T('show prompt', 'Show prompt', 'Tool', 'Reveal the exact prompt that would be sent to the language model for this agonist.');
  T(['both'], 'Both', 'Tool', 'Show icons and text.');
  T(['show', 'hide'], 'Show / hide', 'Tool', 'Toggle hidden details such as gender and sexuality.');
  T('births', 'Births', 'World', 'When on, agonists in good condition can have offspring that inherit mixed genes and mutations. Off by default.');
  T('gender and sexuality', 'Gender and sexuality', 'Tool', 'Hidden in sandbox by default. Show reveals each agonist\'s identity details; no mechanical effect on survival right now.');
  T('hunting', 'Hunting', 'Work', 'Currently hunting: seeking or chasing prey.');
  T(['gathering', 'crafting', 'hauling', 'taming', 'researching'], 'Working', 'Work', 'Currently doing this job from its work priorities.');
  T(['< event', 'event >'], 'Jump between events', 'Tool', 'Skip the timeline to the previous or next recorded event.');
  T('e.t', 'Event ticks', 'Tool', 'Ticks at which something notable happened, shown on the timeline.');
  T(['tier'], 'Tier', 'Tech', 'How deep in the tree a node is. Higher tiers need earlier ones.');
  T('hover anything to see what it is.', 'Map hover', 'Tool', 'Point at any agonist, animal, plant, building or tile on the map to read what it is.');
  /* activity chip */
  [['move', 'Moving', 'Walking toward a target.'], ['idle', 'Idle', 'Nothing to do; recovers +1% energy per tick.'], ['sleep', 'Sleeping', 'Asleep: energy +6% per tick, health +0.4% per tick, sanity +0.6% per tick.'], ['eat', 'Eating', 'Eating from pack, table or a bush.'], ['drink', 'Drinking', 'Drinking at water or from the waterskin.'], ['shelter', 'Sheltering', 'Waiting out rain or cold in a shelter. Energy +1% per tick.'], ['flee', 'Fleeing', 'Running from danger. Adds stress.'], ['strike', 'Fighting', 'Mid-blow in a fight. See Combat moves.']].forEach(function (a) { if (!TIPS[a[0]]) T(a[0], a[1], 'Info', a[2]); });
  /* lookup */
  function norm(s) { return String(s || '').toLowerCase().replace(/\s+/g, ' ').replace(/[:·]+$/, '').trim(); }
  var PREFIX = [['attack', 'attack'], ['raid', 'raid'], ['defend', 'defend'], ['parley', 'parley'], ['births:', 'births'], ['gender and sexuality:', 'gender and sexuality'], ['hunting', 'hunting'], ['dead', 'dead'], ['injured', 'injured'], ['mood', 'mood'], ['mood:', 'mood'], ['dead:', 'dead'], ['follow ', 'follow'], ['rewind to tick', 'latest'], ['tick ', 'tick'], ['day ', 'day'], ['intelligence scale', 'intelligence scale'], ['hunt ', null], ['friendly (', 'neighbours'], ['hostile (', 'neighbours'], ['tier ', null]];
  function tipLookup(text) {
    var n = norm(text); if (!n || n.length > 60) return null;
    if (TIPS[n]) return TIPS[n];
    for (var i = 0; i < PREFIX.length; i++) if (PREFIX[i][1] && n.indexOf(PREFIX[i][0]) === 0 && TIPS[PREFIX[i][1]]) return TIPS[PREFIX[i][1]];
    return null;
  }
  function esc(s) { return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;'); }
  function tipHTML(e, c) {
    var col = KINDCOL[e.k] || '#aea69a', live = [];
    try { live = e.live ? e.live(c || {}) : []; } catch (err) { live = []; }
    var h = '<div style="display:flex;gap:8px;align-items:center;margin-bottom:3px"><span style="font:700 13px/18px system-ui,sans-serif;color:#ede9e1">' + esc(e.t) + '</span><span style="font:600 10px/14px system-ui,sans-serif;padding:0 6px;border-radius:8px;background:' + col + ';color:#1c110b">' + esc(e.k) + '</span></div>';
    if (e.b) h += '<div style="font:12px/17px system-ui,sans-serif;color:#d8d0c4">' + esc(e.b) + '</div>';
    if (live && live.length) h += '<div style="margin-top:6px;padding:5px 7px;border-radius:4px;background:#30231b;border-left:3px solid ' + col + ';font:11px/16px ui-monospace,Menlo,Consolas,monospace;color:#ede9e1">' + live.map(esc).join('<br>') + '</div>';
    var sec = function (title, arr) { return arr && arr.length ? '<div style="margin-top:6px"><div style="font:700 10px/14px system-ui,sans-serif;letter-spacing:.06em;text-transform:uppercase;color:#aea69a">' + title + '</div>' + arr.map(function (l) { return '<div style="font:11px/16px system-ui,sans-serif;color:#cfc7ba">' + esc(l) + '</div>'; }).join('') + '</div>' : ''; };
    return h + sec('How it works', e.r) + sec('Affected by', e.by) + sec('What happens', e.then);
  }
  /* the hover engine: finds the nearest labelled element under the pointer, by data-tip, by title, or by its visible text */
  function attachTips(getCtx) {
    if (typeof document === 'undefined' || !document.body) return function () {};
    var box = document.createElement('div'), timer = null, curKey = null, mx = 0, my = 0, off = false;
    box.id = 'agon-tipx'; box.setAttribute('role', 'tooltip');
    box.style.cssText = 'position:fixed;left:0;top:0;z-index:2147483000;max-width:320px;min-width:180px;padding:9px 11px;border-radius:6px;background:rgba(28,17,11,0.97);border:1px solid #6b5545;box-shadow:0 6px 18px rgba(0,0,0,0.5);pointer-events:none;display:none;color:#ede9e1';
    document.body.appendChild(box);
    var txt = function (n) { return (n.textContent || '').replace(/\s+/g, ' ').trim(); };
    function find(el) {
      var depth = 0;
      for (var n = el; n && n !== document.body && depth < 7; n = n.parentElement, depth++) {
        if (n === box || (n.id && n.id === 'agon-tipx')) return null;
        if (n.getAttribute) {
          var dt = n.getAttribute('data-tip'); if (dt) return { key: 'd:' + dt, raw: dt };
          var ti = n.getAttribute('title'); if (ti) { n.setAttribute('data-title', ti); n.removeAttribute('title'); }
          var dd = n.getAttribute('data-title'); if (dd) return { key: 'd:' + dd, raw: dd };
        }
        var tx = n.childElementCount > 14 ? '' : txt(n);
        if (tx && tx.length <= 46) { var e = tipLookup(tx); if (e) return { key: 'k:' + tx, e: e }; }
        if (n.children && n.children.length >= 2 && n.children.length <= 4 && n.tagName !== 'BODY') {
          var f = txt(n.children[0]);
          if (f && f.length <= 30) { var e2 = tipLookup(f); if (e2) return { key: 'k:' + f, e: e2 }; }
        }
      }
      return null;
    }
    function place() {
      var w = window.innerWidth, h = window.innerHeight, r = box.getBoundingClientRect(), x = mx + 16, y = my + 18;
      if (x + r.width > w - 6) x = Math.max(6, mx - r.width - 12);
      if (y + r.height > h - 6) y = Math.max(6, my - r.height - 12);
      box.style.transform = 'translate(' + Math.round(x) + 'px,' + Math.round(y) + 'px)';
    }
    function show(f) {
      var e = f.e || { t: String(f.raw).split('\n')[0], k: 'Info', b: String(f.raw).split('\n').slice(1).join(' ') };
      box.innerHTML = tipHTML(e, getCtx ? getCtx() : {}); box.style.display = 'block'; place();
    }
    function hide() { clearTimeout(timer); timer = null; curKey = null; box.style.display = 'none'; }
    function onMove(ev) {
      if (off) return; mx = ev.clientX; my = ev.clientY;
      if (ev.buttons) { hide(); return; }
      var f = find(ev.target);
      if (!f) { hide(); return; }
      if (f.key === curKey) { if (box.style.display === 'block') place(); return; }
      curKey = f.key; clearTimeout(timer);
      timer = setTimeout(function () { show(f); }, 160);
    }
    var onOut = function (ev) { if (!ev.relatedTarget) hide(); };
    document.addEventListener('mousemove', onMove, true); document.addEventListener('mouseout', onOut, true);
    document.addEventListener('mousedown', hide, true); document.addEventListener('wheel', hide, true); document.addEventListener('keydown', hide, true);
    return function () { off = true; hide(); document.removeEventListener('mousemove', onMove, true); document.removeEventListener('mouseout', onOut, true); document.removeEventListener('mousedown', hide, true); document.removeEventListener('wheel', hide, true); document.removeEventListener('keydown', hide, true); if (box.parentNode) box.parentNode.removeChild(box); };
  }

  /* ---------- head rig ----------
     One 3D head (unit sphere scaled to the 9.6 unit pawn head) carries everything that sits on it: eyes, brows, mouth, nose, ears, hair, beard,
     headwear. Every facing is the same head turned by a yaw angle and projected with a slight downward camera tilt, so profile, three-quarter,
     front and back views can never disagree. Coordinates are head units (x right, y up, z toward the face), 1 unit = 9.6 pawn units. */
  var RIG = (function () {
    var R = 9.6, CX = 24, CY = 19.5, PHI = 0.2, cp = Math.cos(PHI), sp = Math.sin(PHI), D2R = Math.PI / 180, INK = '#1c110b';
    var YAW = { S: 0, SE: 46 * D2R, E: 90 * D2R, NE: 134 * D2R, N: Math.PI };
    var FACEK = 0.86;
    function sph(az, po, rho) { az *= D2R; po *= D2R; return [rho * Math.sin(po) * Math.sin(az), rho * Math.cos(po), rho * Math.sin(po) * Math.cos(az)]; }
    function prj(p, yaw) { var c = Math.cos(yaw), s = Math.sin(yaw), xp = p[0] * c + p[2] * s, zp = -p[0] * s + p[2] * c; return { X: CX + R * xp, Y: CY - R * (p[1] * cp - zp * sp), d: zp * cp + p[1] * sp, xp: xp, zp: zp }; }
    function f1(v) { return (Math.round(v * 100) / 100); }
    function Pi(d, fill, stroke, sw, op, tf, base) { op = op == null ? 1 : op; fill = fill || 'none'; stroke = stroke || 'none'; if (base && op < 1) { if (fill !== 'none') fill = mix(base, fill, op); if (stroke !== 'none') stroke = mix(base, stroke, op); op = 1; } return { d: d, fill: fill, stroke: stroke, sw: sw || 0, op: op, tf: tf || '' }; }
    function circ(x, y, r) { return 'M' + f1(x - r) + ' ' + f1(y) + ' a' + f1(r) + ' ' + f1(r) + ' 0 1 0 ' + f1(2 * r) + ' 0 a' + f1(r) + ' ' + f1(r) + ' 0 1 0 ' + f1(-2 * r) + ' 0Z'; }
    function ell(x, y, rx, ry) { return 'M' + f1(x - rx) + ' ' + f1(y) + ' a' + f1(rx) + ' ' + f1(ry) + ' 0 1 0 ' + f1(2 * rx) + ' 0 a' + f1(rx) + ' ' + f1(ry) + ' 0 1 0 ' + f1(-2 * rx) + ' 0Z'; }
    function poly(pts, close) { return 'M' + pts.map(function (q) { return f1(q[0]) + ' ' + f1(q[1]); }).join(' L') + (close ? 'Z' : ''); }
    function inside(pt, poly_) { var x = pt[0], y = pt[1], c = false, i, j; for (i = 0, j = poly_.length - 1; i < poly_.length; j = i++) { var a = poly_[i], b = poly_[j]; if ((a[1] > y) !== (b[1] > y) && x < (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]) c = !c; } return c; }
    function arcPts(a0, a1, rr, dir) { var out = [], d = a1 - a0; if (dir > 0) { while (d <= 0) d += 2 * Math.PI; } else { while (d >= 0) d -= 2 * Math.PI; } var n = Math.max(2, Math.ceil(Math.abs(d) / (4 * D2R))); for (var i = 1; i < n; i++) { var a = a0 + d * i / n; out.push([CX + rr * Math.cos(a), CY + rr * Math.sin(a)]); } return out; }
    /* region on the head: closed loop of 3D points; returns the visible outline (loop clipped to the near hemisphere and closed along the limb). mode cap closes over the crown, patch closes the short way. */
    function region(P3, yaw, mode) {
      var H = P3.map(function (p) { return prj(p, yaw); }), n = H.length, vis = H.map(function (h) { return h.d >= 0; }), i, k;
      var nv = vis.filter(Boolean).length; if (!nv) return null;
      var rho = P3.reduce(function (a, p) { return a + Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]); }, 0) / n;
      if (nv === n) return { pts: H.map(function (h) { return [h.X, h.Y]; }), d: poly(H.map(function (h) { return [h.X, h.Y]; }), true), run: H.map(function (h) { return [h.X, h.Y]; }), closed: true };
      var starts = []; for (i = 0; i < n; i++) if (vis[i] && !vis[(i - 1 + n) % n]) starts.push(i);
      var best = null;
      starts.forEach(function (s) { var len = 0; while (vis[(s + len) % n] && len < n) len++; if (!best || len > best.len) best = { s: s, len: len }; });
      var rr = R * rho, snap = function (x, y) { var u = x - CX, v = y - CY, L = Math.hypot(u, v) || 1; return [CX + u / L * rr, CY + v / L * rr]; };
      var h1 = H[best.s], h0 = H[(best.s - 1 + n) % n], t1 = h1.d / (h1.d - h0.d), cS = snap(h1.X + (h0.X - h1.X) * t1, h1.Y + (h0.Y - h1.Y) * t1);
      var e1 = H[(best.s + best.len - 1) % n], e0 = H[(best.s + best.len) % n], t2 = e1.d / (e1.d - e0.d), cE = snap(e1.X + (e0.X - e1.X) * t2, e1.Y + (e0.Y - e1.Y) * t2);
      var run = [cS]; for (k = 0; k < best.len; k++) { var hh = H[(best.s + k) % n]; run.push([hh.X, hh.Y]); } run.push(cE);
      var aE = Math.atan2(cE[1] - CY, cE[0] - CX), aS = Math.atan2(cS[1] - CY, cS[0] - CX), A1 = arcPts(aE, aS, rr, 1), A2 = arcPts(aE, aS, rr, -1), pick;
      if (mode === 'cap') { var pole = prj([0, rho, 0], yaw), pp = [pole.X, pole.Y]; pick = inside(pp, run.concat(A1)) ? A1 : A2; } else pick = A1.length <= A2.length ? A1 : A2;
      var all = run.concat(pick);
      return { pts: all, d: poly(all, true), run: run, closed: false };
    }
    /* patch that never needs a limb closure: points behind the silhouette are pushed onto it, so the outline stays continuous and keeps its own volume */
    function clampPatch(P3, yaw) {
      var pts = [], any = false;
      P3.forEach(function (p) { var h = prj(p, yaw), rho = Math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]), u = h.X - CX, v = h.Y - CY, L = Math.hypot(u, v) || 1, rr = R * rho; if (h.d >= 0) any = true; if (h.d < 0 || L > rr) { pts.push([CX + u / L * rr * (h.d < 0 ? 1 : 1), CY + v / L * rr]); } else pts.push([h.X, h.Y]); });
      return any ? { d: poly(pts, true) } : null;
    }
    /* smooth open curve through projected visible parts of a 3D polyline (for rings and strands); returns a list of path strings, one per visible run */
    function strands(P3, yaw, minD) {
      var out = [], cur = [], i; minD = minD == null ? 0.02 : minD;
      for (i = 0; i < P3.length; i++) { var h = prj(P3[i], yaw); if (h.d > minD) cur.push([h.X, h.Y]); else { if (cur.length > 1) out.push(poly(cur)); cur = []; } }
      if (cur.length > 1) out.push(poly(cur)); return out;
    }
    /* linear interpolation of a periodic key table [[az, value]] over az in degrees (-180..180), symmetric when sym */
    function key(tab, az, sym) { az = ((az + 540) % 360) - 180; if (sym) az = Math.abs(az); var i = 0; while (i < tab.length - 1 && az > tab[i + 1][0]) i++; var a = tab[i], b = tab[Math.min(tab.length - 1, i + 1)]; if (a === b || b[0] === a[0]) return a[1]; var t = (az - a[0]) / (b[0] - a[0]); t = t * t * (3 - 2 * t); return a[1] + (b[1] - a[1]) * t; }
    function loop(tab, rho, sym, rhoFn, n) { var out = [], i; n = n || 72; for (i = 0; i < n; i++) { var az = -180 + 360 * i / n; out.push(sph(az, key(tab, az, sym), rhoFn ? rhoFn(az) : rho)); } return out; }
    /* chains of spheres (strands of hair, buns, ponytails): outlines first, then fills, so overlapping balls read as one lump */
    function densify(B) { var out = []; for (var i = 0; i < B.length; i++) { out.push(B[i]); if (i < B.length - 1) { var A_ = B[i], C_ = B[i + 1], dist = Math.hypot(C_.p[0] - A_.p[0], C_.p[1] - A_.p[1], C_.p[2] - A_.p[2]), st = Math.max(1, Math.ceil(dist / (0.4 * Math.min(A_.r, C_.r)))); for (var k = 1; k < st; k++) { var t = k / st; out.push({ p: [A_.p[0] + (C_.p[0] - A_.p[0]) * t, A_.p[1] + (C_.p[1] - A_.p[1]) * t, A_.p[2] + (C_.p[2] - A_.p[2]) * t], r: A_.r + (C_.r - A_.r) * t }); } } } return out; }
    function chain(B, yaw, fill, hi, shd, opt) {
      if (B.length && Array.isArray(B[0])) { var parts = B.map(function (b) { return chain1(b, yaw, fill, hi, shd, opt, true); }), res = { items: [], d: 0 }; parts.forEach(function (q) { res.items.push.apply(res.items, q.out); res.d += q.d / parts.length; }); parts.forEach(function (q) { res.items.push.apply(res.items, q.fill); }); parts.forEach(function (q) { res.items.push.apply(res.items, q.lines); }); return res; }
      var r1 = chain1(B, yaw, fill, hi, shd, opt, false); if (r1.items) return r1; return { items: r1.out.concat(r1.fill, r1.lines), d: r1.d };
    }
    function chain1(B, yaw, fill, hi, shd, opt, split) {
      opt = opt || {}; var items = [], dsum = 0;
      var H0 = B.map(function (b) { var h = prj(b.p, yaw); return { h: h, r: b.r * R }; }); H0.forEach(function (q) { dsum += q.h.d; });
      var fr = []; if (opt.each) {
        var H = H0.slice().sort(function (u, v) { return u.h.d - v.h.d; }), bk = [];
        H.forEach(function (q) {
          var items = q.h.d < 0.05 ? bk : fr;
          items.push(Pi(circ(q.h.X, q.h.Y, q.r + 0.9), INK, 'none', 0), Pi(circ(q.h.X, q.h.Y, q.r), fill, 'none', 0));
          if (shd) items.push(Pi('M' + f1(q.h.X + q.r * 0.9) + ' ' + f1(q.h.Y - q.r * 0.2) + ' A' + f1(q.r) + ' ' + f1(q.r) + ' 0 0 1 ' + f1(q.h.X - q.r * 0.2) + ' ' + f1(q.h.Y + q.r * 0.9) + ' A' + f1(q.r * 1.1) + ' ' + f1(q.r * 1.1) + ' 0 0 0 ' + f1(q.h.X + q.r * 0.9) + ' ' + f1(q.h.Y - q.r * 0.2) + 'Z', shd, 'none', 0, 0.55, '', fill));
          if (hi && q.r > 1.2) items.push(Pi('M' + f1(q.h.X - q.r * 0.55) + ' ' + f1(q.h.Y - q.r * 0.1) + ' q' + f1(q.r * 0.1) + ' ' + f1(-q.r * 0.5) + ' ' + f1(q.r * 0.55) + ' ' + f1(-q.r * 0.5), 'none', hi, Math.min(1, q.r * 0.22), 0.75, '', fill));
        });
        return { items: fr, back: bk, d: dsum / B.length };
      }
      var Hd = densify(B).map(function (b) { var h = prj(b.p, yaw); return { h: h, r: b.r * R }; }), out = [], fl = [], ln = [];
      Hd.forEach(function (q) { out.push(Pi(circ(q.h.X, q.h.Y, q.r + 1.0), INK, 'none', 0)); });
      Hd.forEach(function (q) { fl.push(Pi(circ(q.h.X, q.h.Y, q.r), fill, 'none', 0)); });
      if (shd && !opt.nolines) { var ps = H0.map(function (q) { return [q.h.X + q.r * 0.38, q.h.Y + q.r * 0.1]; }); if (ps.length > 1) ln.push(Pi(poly(ps), 'none', shd, Math.max(0.8, H0[0].r * 0.5), 0.45, '', fill)); }
      if (hi && !opt.nolines) { var pl = H0.map(function (q) { return [q.h.X - q.r * 0.34, q.h.Y - q.r * 0.2]; }); if (pl.length > 1) ln.push(Pi(poly(pl), 'none', hi, Math.max(0.7, H0[0].r * 0.2), 0.7, '', fill)); }
      if (shd && !opt.nolines && H0.length > 1) { var pm = H0.map(function (q) { return [q.h.X, q.h.Y]; }); ln.push(Pi(poly(pm), 'none', shd, 0.7, 0.55, '', fill)); }
      return { out: out, fill: fl, lines: ln, d: dsum / B.length };
    }
    function ptsChain(pts, r0, r1) { return pts.map(function (p, i) { var t = pts.length > 1 ? i / (pts.length - 1) : 0; return { p: p, r: r0 + (r1 - r0) * t }; }); }
    /* ---- hair styles: hairline keys [[az, polar degrees from the crown]] and sphere radius, plus extra masses ---- */
    var HL = {
      crop: [[0, 55], [40, 61], [65, 71], [90, 78], [125, 100], [180, 134]],
      long: [[0, 53], [40, 60], [65, 80], [90, 108], [125, 128], [180, 135]],
      bob: [[0, 60], [40, 66], [65, 96], [90, 116], [125, 120], [180, 124]],
      curl: [[0, 52], [40, 58], [65, 74], [90, 88], [125, 106], [180, 138]],
      short: [[0, 56], [40, 62], [65, 72], [90, 80], [125, 102], [180, 136]]
    };
    function hairPieces(L, yaw, c) {
      // c: hair, hs, hl colours. Returns { cap: {loop, rho, fill}, masses: [chain objects], front: [items], strokes }
      var st = L.hairStyle, o = { cap: null, ch: [], spikes: [], extraFront: [], hairline: null, part: true, thin: false };
      var capDef = function (tab, rho, sym) { return { tab: tab, rho: rho, sym: sym !== false }; };
      if (st === 0) { o.cap = capDef(HL.crop, 1.06);  }
      else if (st === 1) { o.cap = capDef(HL.long, 1.1); o.backs = [-0.35, -0.48, -0.61, -0.74, -0.87, -1.0, -1.13, -1.26, -1.4, -1.52].map(function (y, j) { var q = [], w = Math.max(0.2, 0.66 - j * j * 0.0045 - j * 0.012), wv = 0.05 * Math.sin(j * 0.9); for (var x = -w; x <= w + 0.001; x += 0.2) q.push({ p: [x + wv, y, -Math.sqrt(Math.max(0.12, 1 - x * x * 0.5)) * 0.84 - 0.02 * j], r: 0.27 }); return q; }); o.nolines = true; o.locks = [-1, 1].map(function (s) { return [[s * 0.95, -0.25, -0.12], [s * 0.97, -0.55, -0.2], [s * 0.96, -0.85, -0.26], [s * 0.95, -1.15, -0.3], [s * 0.94, -1.4, -0.3]].map(function (q, i) { return { p: q, r: 0.2 - i * 0.012 }; }); }); }
      else if (st === 2) { o.cap = capDef(HL.crop, 1.06); o.ch.push(ptsChain([[0, 1.06, -0.18], [0, 1.3, -0.12]], 0.3, 0.38)); }
      else if (st === 3) { o.cap = null; }
      else if (st === 4) { o.cap = capDef(HL.short, 1.015); o.stub = true; var cr = []; for (var b = 38; b >= -112; b -= 6) { var br = b * D2R, t = (b + 112) / 150, sn = Math.sin(Math.PI * Math.min(1, Math.max(0, t))), h = 1.03 + 0.3 * sn; cr.push({ p: [0, Math.cos(br) * h, Math.sin(br) * h], r: 0.12 + 0.11 * sn }); } o.ch.push(cr); }
      else if (st === 5) { o.cap = capDef(HL.crop, 1.06);  o.ch.push(ptsChain([[0, 0.86, -0.42], [0, 1.02, -0.52]], 0.34, 0.42)); }
      else if (st === 6) { o.cap = capDef(HL.crop, 1.07);  o.backs = [[{ p: [0.06, 0.5, -0.97], r: 0.2 }, { p: [0.12, 0.28, -1.12], r: 0.3 }, { p: [0.18, -0.1, -1.2], r: 0.34 }, { p: [0.2, -0.55, -1.2], r: 0.3 }, { p: [0.18, -0.92, -1.12], r: 0.22 }]]; o.tie = [0.07, 0.46, -0.99]; }
      else if (st === 7) { o.cap = capDef(HL.curl, 1.2); o.curls = true; }
      else if (st === 8) { o.cap = capDef(HL.bob, 1.12); o.ch.push(ptsChain([[-0.95, -0.34, 0.06], [-0.92, -0.58, -0.12], [-0.8, -0.74, -0.4]], 0.32, 0.3)); o.ch.push(ptsChain([[0.95, -0.34, 0.06], [0.92, -0.58, -0.12], [0.8, -0.74, -0.4]], 0.32, 0.3)); o.backs = [[-0.6, -0.3, 0, 0.3, 0.6].map(function (x) { return { p: [x, -0.5 - 0.05 * Math.abs(x), -Math.sqrt(1 - x * x * 0.55) * 0.88], r: 0.4 }; }), [-0.5, -0.25, 0, 0.25, 0.5].map(function (x) { return { p: [x, -0.74, -Math.sqrt(1 - x * x * 0.55) * 0.84], r: 0.34 }; })]; }
      else if (st === 9) { o.cap = capDef(HL.short, 1.05); o.spiky = true; }
      else if (st === 10) { o.cap = capDef(HL.crop, 1.06);  var br2 = []; for (var k = 0; k < 7; k++) br2.push({ p: [(k % 2 ? 0.1 : -0.1), -0.05 - k * 0.22, -1.0 - 0.02 * k], r: 0.18 - k * 0.006 }); o.backs = [br2]; o.braid = true; }
      else if (st === 11) { o.cap = capDef(HL.short, 1.025); o.thin = true; }
      return o;
    }
    function side(L, s) { return ptsChain([[-0.93, -0.16, 0.34], [-0.95, -0.34, 0.28]], 0.1, 0.1); }
    /* ---- hair clumps: a projection of small solids standing on the head sphere.
       Every clump is placed by (azimuth, polar angle) on the scalp and built in 3D from the local tangent frame, so it follows the curvature of the skull instead of being drawn flat.
       tuft: a triangular pyramid. Its base is a spherical triangle sitting on the scalp, its apex rises along the surface normal and leans along a sweep direction.
       blob: a dome (curl, puff, lump) whose rim lies on the scalp and whose crown bulges out.
       Faces are culled and shaded by their 3D normal against the view and a screen-space light, then drawn far to near, so a clump reads right at any head turn. */
    function v3sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
    function v3dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
    function v3cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
    function v3norm(a) { var l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; }
    function scalpFrame(az, po) { var a = az * D2R, p = po * D2R; return { n: [Math.sin(p) * Math.sin(a), Math.cos(p), Math.sin(p) * Math.cos(a)], e1: [Math.cos(a), 0, -Math.sin(a)], e2: [Math.cos(p) * Math.sin(a), -Math.sin(p), Math.cos(p) * Math.cos(a)] }; }
    function hull2(pts) {
      var P = pts.slice().sort(function (a, b) { return a[0] - b[0] || a[1] - b[1]; }), cr = function (o, a, b) { return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]); }, lo = [], up = [], i;
      for (i = 0; i < P.length; i++) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], P[i]) <= 0) lo.pop(); lo.push(P[i]); }
      for (i = P.length - 1; i >= 0; i--) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], P[i]) <= 0) up.pop(); up.push(P[i]); }
      lo.pop(); up.pop(); return lo.concat(up);
    }
    /* camera and light for one facing, in model space */
    function viewRig(yaw) {
      var c = Math.cos(yaw), s = Math.sin(yaw), lp = v3norm([-0.45, 0.75, 0.55]);
      return { v: [-cp * s, sp, cp * c], l: v3norm([lp[0] * c - lp[2] * s, lp[1], lp[0] * s + lp[2] * c]) };
    }
    function shadeOf(lam, hair, hs, hl) { return lam >= 0.5 ? mix(hair, hl, Math.min(1, (lam - 0.5) / 0.5)) : mix(hs, hair, Math.max(0, lam) / 0.5); }
    /* returns { d: depth of the clump base, items: [Pi] } for one clump */
    function clumpItems(cl, yaw, hair, hs, hl, rig) {
      var f = scalpFrame(cl.az, cl.po), rho = cl.rho || 1.04, ctr = [f.n[0] * rho, f.n[1] * rho, f.n[2] * rho], items = [], base = prj(ctr, yaw), k, ring = [], th;
      var onScalp = function (ang, rad, r2) { var q = [f.n[0] + rad * (Math.cos(ang) * f.e1[0] + Math.sin(ang) * f.e2[0]), f.n[1] + rad * (Math.cos(ang) * f.e1[1] + Math.sin(ang) * f.e2[1]), f.n[2] + rad * (Math.cos(ang) * f.e1[2] + Math.sin(ang) * f.e2[2])]; q = v3norm(q); return [q[0] * r2, q[1] * r2, q[2] * r2]; };
      var rad = Math.tan((cl.size || 16) * D2R);
      if (cl.kind === 'blob') {
        var N = 12, h = cl.height || 0.3, top = [f.n[0] * (rho + h), f.n[1] * (rho + h), f.n[2] * (rho + h)], sw = cl.sweep || [0, 0, 0], tp = v3sub(sw, [f.n[0] * v3dot(sw, f.n), f.n[1] * v3dot(sw, f.n), f.n[2] * v3dot(sw, f.n)]);
        top = [top[0] + tp[0] * h * (cl.lean || 0), top[1] + tp[1] * h * (cl.lean || 0), top[2] + tp[2] * h * (cl.lean || 0)];
        for (k = 0; k < N; k++) { th = k / N * Math.PI * 2; ring.push(onScalp(th, rad, rho)); }
        var pr = ring.map(function (q) { var p = prj(q, yaw); return [p.X, p.Y]; }), pt = prj(top, yaw), hl2 = hull2(pr.concat([[pt.X, pt.Y]]));
        if (hl2.length < 3) return null;
        var path = poly(hl2.map(function (q) { return q; }), true), cx = 0, cy = 0; hl2.forEach(function (q) { cx += q[0]; cy += q[1]; }); cx /= hl2.length; cy /= hl2.length;
        var lx = rig.l, lp2 = prj([lx[0], lx[1], lx[2]], yaw), dx = (lp2.X - CX) * 0.14, dy = (lp2.Y - CY) * 0.14;
        var inner = poly(hl2.map(function (q) { return [cx + (q[0] - cx) * 0.82 + dx, cy + (q[1] - cy) * 0.82 + dy]; }), true), spec = poly(hl2.map(function (q) { return [cx + (q[0] - cx) * 0.34 + dx * 2.2, cy + (q[1] - cy) * 0.34 + dy * 2.2]; }), true);
        items.push(Pi(path, 'none', INK, 2.4, 1), Pi(path, hs, 'none', 0, 1), Pi(inner, hair, 'none', 0, 1), Pi(spec, hl, 'none', 0, 0.9));
        return { d: base.d, out: [], items: items };
      }
      // tuft
      var hgt = cl.height || 0.5, rot = (cl.rot || 0) * D2R, tg = cl.radial ? f.e2 : (cl.sweep || [0, 0.3, -0.9]);
      var P = [0, 1, 2].map(function (q) { return onScalp(rot + q * 2 * Math.PI / 3, rad, rho); });
      var up = cl.up == null ? 0.5 : cl.up, ln = cl.lean == null ? 0.5 : cl.lean, dir = v3norm([f.n[0] * (1 - up) + tg[0] * ln, f.n[1] * (1 - up) + up + tg[1] * ln, f.n[2] * (1 - up) + tg[2] * ln]);
      var apex = [ctr[0] + dir[0] * hgt, ctr[1] + dir[1] * hgt, ctr[2] + dir[2] * hgt];
      var pp = P.map(function (q) { var p = prj(q, yaw); return [p.X, p.Y]; }), pa = prj(apex, yaw), apx = [pa.X, pa.Y], bc = [(pp[0][0] + pp[1][0] + pp[2][0]) / 3, (pp[0][1] + pp[1][1] + pp[2][1]) / 3];
      /* a lock, not a pyramid: the tip is bent sideways the same way on every tuft, so the spikes swirl like hair that has been pushed up */
      var ax0 = apx[0] - bc[0], ay0 = apx[1] - bc[1], al = Math.hypot(ax0, ay0) || 1, bnd = (cl.bend == null ? 0.3 : cl.bend) * al, bvx = -ay0 / al * bnd, bvy = ax0 / al * bnd;
      apx = [apx[0] + bvx * 0.6, apx[1] + bvy * 0.6];
      /* the two sides that run to the apex are drawn concave, so a tuft is a flame-like point and not a gem; adjacent faces share the same curve, so they tile exactly */
      var cur = pp.map(function (q) { var mx = (q[0] + apx[0]) / 2, my = (q[1] + apx[1]) / 2; return [mx + (bc[0] - mx) * 0.3 + bvx * 0.25, my + (bc[1] - my) * 0.3 + bvy * 0.25]; });
      var faces = [];
      for (k = 0; k < 3; k++) {
        var A = P[k], B = P[(k + 1) % 3], nrm = v3norm(v3cross(v3sub(B, apex), v3sub(A, apex))), cen = [(A[0] + B[0] + apex[0]) / 3, (A[1] + B[1] + apex[1]) / 3, (A[2] + B[2] + apex[2]) / 3];
        if (v3dot(nrm, cen) < 0) nrm = [-nrm[0], -nrm[1], -nrm[2]];
        if (v3dot(nrm, rig.v) <= 0.02) continue;
        var cc = prj(cen, yaw), a2 = pp[k], b2 = pp[(k + 1) % 3];
        var d = 'M' + f1(a2[0]) + ' ' + f1(a2[1]) + ' Q' + f1(cur[k][0]) + ' ' + f1(cur[k][1]) + ' ' + f1(apx[0]) + ' ' + f1(apx[1]) + ' Q' + f1(cur[(k + 1) % 3][0]) + ' ' + f1(cur[(k + 1) % 3][1]) + ' ' + f1(b2[0]) + ' ' + f1(b2[1]) + 'Z';
        faces.push({ d: cc.d, path: d, lam: v3dot(nrm, rig.l) * 0.5 + 0.5, crease: 'M' + f1(apx[0]) + ' ' + f1(apx[1]) + ' Q' + f1(cur[(k + 1) % 3][0]) + ' ' + f1(cur[(k + 1) % 3][1]) + ' ' + f1(b2[0]) + ' ' + f1(b2[1]) });
      }
      faces.sort(function (a, b) { return a.d - b.d; });
      /* one outline per lock: only the tip (left base corner, apex, right base corner), never the base, so a lock grows out of the hair instead of sitting on it */
      var ex0 = 0, ex1 = 0, cmin = 1e9, cmax = -1e9, kk;
      for (kk = 0; kk < 3; kk++) { var cz = ax0 * (pp[kk][1] - bc[1]) - ay0 * (pp[kk][0] - bc[0]); if (cz < cmin) { cmin = cz; ex0 = kk; } if (cz > cmax) { cmax = cz; ex1 = kk; } }
      var tip = 'M' + f1(pp[ex0][0]) + ' ' + f1(pp[ex0][1]) + ' Q' + f1(cur[ex0][0]) + ' ' + f1(cur[ex0][1]) + ' ' + f1(apx[0]) + ' ' + f1(apx[1]) + ' Q' + f1(cur[ex1][0]) + ' ' + f1(cur[ex1][1]) + ' ' + f1(pp[ex1][0]) + ' ' + f1(pp[ex1][1]);
      var fl = [];
      faces.forEach(function (fc) { fl.push(Pi(fc.path, (fc.lam < 0.52 ? hs : hair), 'none', 0, 1)); });
      fl.push(Pi('M' + f1(bc[0]) + ' ' + f1(bc[1]) + ' Q' + f1((bc[0] + apx[0]) / 2 + bvx * 0.3) + ' ' + f1((bc[1] + apx[1]) / 2 + bvy * 0.3) + ' ' + f1(apx[0]) + ' ' + f1(apx[1]), 'none', hs, 0.7, 0.55));
      return { d: base.d, out: [Pi(tip, 'none', INK, 1.9, 1)], items: fl };
    }
    /* crown of tufts over the scalp: equal-area spiral so the spacing is even, each tuft clipped to the hairline, taller toward the top, swept back */
    function crownClumps(hairTab) {
      /* three rings of big tufts: a fringe ring at the hairline, a mid ring, and one on the crown. Few and large, so the silhouette reads as bold spikes and not as noise. */
      var out = [], rings = [[9, 57, 0.46, 23, 0.62, 14], [5, 34, 0.5, 22, 0.55, 30], [2, 12, 0.5, 20, 0.5, 10]];
      rings.forEach(function (rg) {
        for (var i = 0; i < rg[0]; i++) {
          var az = rg[0] === 1 ? 0 : -180 + (i + 0.5) * 360 / rg[0] + rg[5], hlp = key(hairTab, az, true);
          if (rg[1] + rg[3] * 0.55 > hlp - 3) continue;
          out.push({ kind: 'tuft', az: az, po: rg[1], size: rg[3], height: rg[2], rot: (i * 47) % 120, lean: rg[4], up: 0.5, radial: true, rho: 1.05 });
        }
      });
      return out;
    }
    /* build everything for one look and facing */
    function build(L, c, view, opt) {
      opt = opt || {};
      var yaw = opt.yaw != null ? opt.yaw : (YAW[view] != null ? YAW[view] : 0), fy = yaw * (view === 'E' ? 0.52 : FACEK);
      var res = { back: [], headBack: [], headSkin: [], face: [], headFront: [], faceOn: false, capOn: false };
      var hair = c.hair, hs = c.hairShade, hl = c.hairHi, skin = c.skin, sks = c.skinShade, st = L.hairStyle, i, items;
      var hp = hairPieces(L, yaw, c);
      var capFill = hair; if (hp.thin) capFill = hs;
      var beanie = L.acc === 2, band = L.acc === 1;
      // ---- ears (3D blobs on the sides of the head); the near ear is drawn after the beard so sideburns tuck behind it
      var earTop = [];
      [-1, 1].forEach(function (s) {
        var p = sph(s * (view === 'E' ? 112 : 92), view === 'E' ? 99 : 97, 1.0), h = prj(p, yaw), front = h.d > 0.3, r = (view === 'E' ? 0.155 : 0.27) * R;
        var it = [Pi(ell(h.X, h.Y, r * 0.85, r * 1.15), skin, INK, 1.0)];
        if (h.d > 0.45) { var rad = Math.abs(h.xp) < 0.98 ? 1 : 0; it.push(Pi('M' + f1(h.X - r * 0.1) + ' ' + f1(h.Y - r * 0.5) + ' q' + f1(r * 0.7) + ' ' + f1(r * 0.5) + ' ' + f1(0) + ' ' + f1(r * 1.0), 'none', sks, 0.8, 0.9)); }
        if (front) earTop.push.apply(earTop, it); else res.headBack.push.apply(res.headBack, it);
      });
      // ---- nose: a small ball in front of the face, behind the head outline so it breaks the silhouette in profile
      var noseTop = [], nT = prj(sph(0, 101, 1.13), fy);
      if (nT.d > -0.15) {
        var nb = Pi(circ(nT.X, nT.Y, (0.125 * R + 0.1) * (L.face ? L.face.nose : 1)), sks, INK, 1.5); res.headBack.push(nb);
        var nf = Pi(circ(nT.X, nT.Y, 0.125 * R - 0.55), skin, 'none', 0); if (nT.d > 0.2) noseTop.push(Pi(circ(nT.X, nT.Y, 0.125 * R - 0.35), skin, 'none', 0), Pi('M' + f1(nT.X - 0.6) + ' ' + f1(nT.Y + 0.9) + ' q0.9 0.7 1.8 0', 'none', sks, 0.7, 1));
      }
      // ---- beard (follows the face yaw)
      var beard = [];
      /* facial hair is male-pattern: women only grow it with the rare hirsute gene (hormonal / mutation) */
      if (L.beard && !opt.child && (opt.sex !== 'F' || L.hirsute)) {
        var bt = L.beard, stub = bt === 1, loops = [];
        /* sideburns run up to the ear; the near ear is drawn over them */
        var JAW = [[-86, 98, 1.03], [-82, 110, 1.05], [-72, 122, 1.07], [-50, 135, 1.09], [-38, 142, 1.12], [-20, 150, 1.16], [0, 153, 1.18], [20, 150, 1.16], [38, 142, 1.12], [50, 133, 1.09], [72, 122, 1.07], [82, 110, 1.05], [86, 98, 1.03]];
        var CHEEK = [[78, 97, 1.03], [62, 101, 1.035], [46, 106, 1.04], [30, 111, 1.05], [14, 114, 1.06], [0, 114, 1.07], [-14, 114, 1.06], [-30, 111, 1.05], [-46, 106, 1.04], [-62, 101, 1.035], [-78, 97, 1.03]];
        var MOUS = [[-28, 107, 1.03], [-14, 104, 1.05], [0, 105, 1.05], [14, 104, 1.05], [28, 107, 1.03], [20, 114, 1.04], [0, 111, 1.05], [-20, 114, 1.04]];
        var GOAT = [[-14, 125, 1.04], [0, 123, 1.05], [14, 125, 1.04], [17, 139, 1.07], [0, 152, 1.11], [-17, 139, 1.07]];
        if (bt === 1 || bt === 2) {
          var out = JAW, inn = CHEEK;
          if (stub) { out = out.map(function (q) { return [q[0], q[1] - 5, 1.012]; }); inn = inn.map(function (q) { return [q[0], q[1] + 1, 1.012]; }); }
          loops.push(out.concat(inn));
        } else if (bt === 3) loops.push(MOUS);
        else if (bt === 4) loops.push(GOAT);
        else if (bt === 5) { /* long beard: the jaw line drops into a hanging point below the chin */
          loops.push([[-86, 98, 1.03], [-76, 124, 1.08], [-60, 146, 1.16], [-40, 160, 1.3], [-20, 170, 1.42], [0, 174, 1.55], [20, 170, 1.42], [40, 160, 1.3], [60, 146, 1.16], [76, 124, 1.08], [86, 98, 1.03]].concat(CHEEK));
        } else if (bt === 6) { /* chinstrap: a thin band along the jaw, clean cheeks */
          loops.push(JAW.concat(JAW.map(function (q) { return [q[0] * 0.92, Math.max(100, q[1] - 14), q[2]]; }).reverse()));
        } else if (bt === 7) { loops.push(MOUS); loops.push(GOAT); } /* van dyke */
        beard = [];
        loops.forEach(function (loopB) {
          var P3b = []; loopB.forEach(function (q, idx) { var nxt = loopB[(idx + 1) % loopB.length]; for (var t = 0; t < 4; t++) { var u = t / 4; P3b.push(sph(q[0] + (nxt[0] - q[0]) * u, q[1] + (nxt[1] - q[1]) * u, q[2] + (nxt[2] - q[2]) * u)); } });
          var rg = region(P3b, fy, 'patch');
          if (!rg) return; /* shading lies on the beard's own segments: the whole shape is drawn in the shade colour, then the same shape with its lower edge lifted is drawn in the beard colour on top, so the shade is exactly the underside of each segment (jaw, moustache, chin) and can never spill outside it */
          var pm = 0; P3b.forEach(function (q) { pm += Math.acos(Math.max(-1, Math.min(1, q[1] / (Math.hypot(q[0], q[1], q[2]) || 1)))) / D2R; }); pm /= P3b.length;
          var lit = P3b.map(function (q) { var rr = Math.hypot(q[0], q[1], q[2]) || 1, po = Math.acos(Math.max(-1, Math.min(1, q[1] / rr))) / D2R, az = Math.atan2(q[0], q[2]) / D2R; return po > pm ? sph(az, po - 7, rr) : q; }), rgL = stub ? null : region(lit, fy, 'patch');
          if (stub) beard = beard.concat([Pi(rg.d, hs, 'none', 0, 0.42, '', skin)]);
          else if (rg.closed) beard = beard.concat([Pi(rg.d, hs, INK, 1.0, 1)], rgL && rgL.closed ? [Pi(rgL.d, hair, 'none', 0, 1)] : []);
          else beard = beard.concat([Pi(rg.d, hs, 'none', 0, 1)], rgL ? [Pi(rgL.d, hair, 'none', 0, 1)] : [], [Pi(poly(rg.run), 'none', INK, 1.0, 1)]);
        });
        if (!stub && beard.length) { /* hair texture: a few short strokes sweeping down from the jaw, plus a lit crest */
          var tcol = shade(hair, 0.62), tk = bt === 5 ? 1.5 : 1;
          (bt === 3 ? [] : bt === 4 || bt === 7 ? [-8, 0, 8] : bt === 6 ? [] : [-46, -24, 0, 24, 46]).forEach(function (az) {
            var sp = []; for (var u = 0; u <= 4; u++) sp.push(sph(az * (1 + u * 0.05), 130 + u * (bt === 5 ? 10 : 5), 1.16 + (bt === 5 ? u * 0.07 : u * 0.012)));
            strands(sp, fy, 0.12).forEach(function (d) { beard.push(Pi(d, 'none', tcol, 0.7, 0.75)); });
          });
        }
      }
      // ---- face plate
      var feat = function (az, po, rho) { return prj(sph(az, po, rho), fy); };
      var F = L.face || { az: 24, po: 92, es: 1, bw: 1.3, bpo: 72, bl: 1.75, nose: 1, mw: 1 }, isF = opt.sex === 'F';
      var eyeKind = opt.eyes || 'dot', ln = { happy: ['M-1.6 0.8 q1.6 -2.8 3.2 0', 'M-1.6 0.8 q1.6 -2.8 3.2 0'], closed: ['M-1.6 0.2 q1.6 1.4 3.2 0', 'M-1.6 0.2 q1.6 1.4 3.2 0'], pain: ['M-1.4 -1.6 l2.6 1.6 l-2.6 1.6', 'M1.4 -1.6 l-2.6 1.6 l2.6 1.6'], x: ['M-1.5 -1.2 l3 2.4 M1.5 -1.2 l-3 2.4', 'M-1.5 -1.2 l3 2.4 M1.5 -1.2 l-3 2.4'] };
      var tilt = (opt.tilt || 0) * 0.3, lift = (opt.lift || 0) * 0.3;
      [-1, 1].forEach(function (s, si) {
        var e = feat(s * F.az, F.po, 1.0), op = Math.max(0, Math.min(1, (e.d - 0.04) / 0.2)); op = op >= 0.5 ? 1 : 0; if (e.zp < (view === 'E' ? 0.5 : 0.34)) op = 0; if (!op) return;
        var sx = Math.max(0.22, e.zp), tf = 'translate(' + f1(e.X) + ' ' + f1(e.Y) + ') scale(' + f1(sx * F.es) + ' ' + f1(F.es) + ')';
        if (isF && op) res.face.push(Pi('M' + f1(s * 1.3) + ' -1.0 l' + f1(s * 0.9) + ' -0.9', 'none', INK, 0.9, op, tf));
        if (eyeKind === 'dot') { res.face.push(Pi(ell(0, 0, 1.3, 1.7), INK, 'none', 0, op, tf), Pi(circ(0.5, -0.6, 0.5), '#ffffff', 'none', 0, op, tf)); }
        else if (eyeKind === 'wide') { res.face.push(Pi(ell(0, -0.2, 1.9, 2.4), '#f3ece3', INK, 0.8, op, tf), Pi(circ(0, -0.1, 0.9), INK, 'none', 0, op, tf)); }
        else if (eyeKind === 'half') { res.face.push(Pi(ell(0, 0.6, 1.3, 1), INK, 'none', 0, op, tf), Pi('M-1.7 -0.4 h3.4', 'none', INK, 1.2, op, tf)); }
        else if (ln[eyeKind]) res.face.push(Pi(ln[eyeKind][si], 'none', INK, 1.3, op, tf));
        { var b = feat(s * F.az, F.bpo, 1.0), bop = Math.max(0, Math.min(1, (b.d - 0.04) / 0.2)); bop = bop >= 0.5 ? 1 : 0; if (bop > 0) { var bsx = Math.max(0.22, b.zp), btf = 'translate(' + f1(b.X) + ' ' + f1(b.Y - lift) + ') scale(' + f1(bsx) + ' 1)', bl = F.bl, bwd = F.bw * (isF ? 0.75 : 1.15), tt = opt.brow ? tilt : 0, ar = isF ? 0.65 : 0.4;
          res.face.push(Pi(s < 0 ? 'M' + f1(-bl) + ' ' + f1(-tt + 0.3) + ' Q0 ' + f1(-ar) + ' ' + f1(bl) + ' ' + f1(tt + 0.3) : 'M' + f1(bl) + ' ' + f1(-tt + 0.3) + ' Q0 ' + f1(-ar) + ' ' + f1(-bl) + ' ' + f1(tt + 0.3), 'none', hs, f1(bwd), bop, btf)); } }
        if (opt.blush) { var bl = feat(s * 44, 106, 1.0), blo = Math.max(0, Math.min(1, (bl.d - 0.1) / 0.25)); blo = blo >= 0.5 ? 1 : 0; if (blo > 0) res.face.push(Pi(ell(0, 0, 1.7, 1), '#e08a78', 'none', 0, 0.7, 'translate(' + f1(bl.X) + ' ' + f1(bl.Y) + ') scale(' + f1(Math.max(0.2, bl.zp)) + ' 1)', skin)); }
      });
      var m = feat(0, 118, 1.0), mop = Math.max(0, Math.min(1, (m.d - 0.04) / 0.2));
      mop = mop >= 0.5 ? 1 : 0; if (mop > 0 && opt.mouth) res.face.push(Pi(opt.mouth[0], opt.mouth[1], INK, 1.1, mop, 'translate(' + f1(m.X) + ' ' + f1(m.Y) + ') scale(' + f1(Math.max(0.22, m.zp) * F.mw) + ' 1) translate(-24 -25.5)'));
      res.faceOn = true;
      var mouthItem = (mop > 0 && opt.mouth) ? res.face[res.face.length - 1] : null;
      // ---- hair
      var shell = hp.cap ? hp.cap.rho : 1.06;
      if (hp.cap) {
        var P3 = loop(hp.cap.tab, 0, hp.cap.sym, function () { return hp.cap.rho; }, 72), rg2 = region(P3, yaw, 'cap');
        if (rg2) {
          var capItems = [];
          // forehead shadow under the hairline
          var hlP = strands(P3, yaw, 0.1);
          
          capItems.push(hp.stub ? Pi(rg2.d, hs, 'none', 0, 0.28, '', skin) : Pi(rg2.d, capFill, INK, hp.thin ? 1 : 1.5, 1));
          // flow strokes: meridians that carry the light and shade across the head
          if (!hp.stub) [[-38, hl, 0.55, 1.1], [36, hs, 0.5, 1], [-12, hs, 0.4, 0.8], [150, hs, 0.45, 1], [-150, hl, 0.45, 1]].forEach(function (mq) {
            var pts = []; for (var po = 10; po < 160; po += 5) { var lim = key(hp.cap.tab, mq[0], hp.cap.sym) - 6; if (po > lim) break; pts.push(sph(mq[0], po, hp.cap.rho + 0.004)); }
            strands(pts, yaw, 0.18).forEach(function (d) { capItems.push(Pi(d, 'none', mq[1], mq[3], mq[2], '', hair)); });
          });
          if (!hp.thin && !hp.stub && hp.part && !hp.spiky) { var pp = []; for (var po2 = 4; po2 < 58; po2 += 4) pp.push(sph(-24, po2, hp.cap.rho + 0.006)); strands(pp, yaw, 0.2).forEach(function (d) { capItems.push(Pi(d, 'none', hs, 0.9, 0.7, '', hair)); }); }
          res.capItems = capItems;
        }
      }
      // masses
      var layerOf = function (obj) { return obj.d > 0.3 ? 'front' : 'back'; };
      var chains = [];
      (hp.backs ? [hp.backs] : []).forEach(function (bk) { var o = chain(bk, yaw, hair, hl, hs, { nolines: hp.nolines }); if (hp.nolines) { [-0.55, -0.2, 0.2, 0.55].forEach(function (x) { var st_ = []; for (var y = -0.4; y > -1.5; y -= 0.1) st_.push(sph(0, 0, 1) && [x * (1 + (-0.4 - y) * 0.05), y, -Math.sqrt(1 - x * x * 0.5) * 0.84 - 0.1]); strands(st_, yaw, 0.0).forEach(function (d) { o.items.push(Pi(d, 'none', hs, 0.9, 0.55, '', hair)); }); }); } chains.push({ o: o, lay: o.d > 0.12 ? 'front' : 'back' }); });
      (hp.ch || []).forEach(function (cc) { var o = chain(cc, yaw, hair, hl, hs); chains.push({ o: o, lay: layerOf(o) }); });
      (hp.locks || []).forEach(function (cc) { var o = chain(cc, yaw, hair, hl, hs); chains.push({ o: o, lay: o.d > 0.12 ? 'front' : 'back' }); });
      if (hp.tie) { var tp = prj(hp.tie, yaw); res._tie = tp; }
      // curls: ring of small balls along the hairline, plus crown clumps
      if (hp.curls) {
        var cb = []; for (i = 0; i < 20; i++) { var az = -180 + i * 18 + (i % 2 ? 4 : -4), po = key(HL.curl, az, true) - 1 + (i % 3) * 3; cb.push({ p: sph(az, po, 1.14), r: 0.27 }); }
        for (i = 0; i < 26; i++) { var az3 = i * 137.5, po3 = 6 + 62 * Math.sqrt((i + 0.5) / 26); cb.push({ p: sph(az3, po3, 1.12 + 0.05 * (i % 3)), r: 0.28 - 0.04 * (po3 / 70) }); }
        var oc = chain(cb, yaw, hair, hl, hs, { each: true }); chains.push({ o: oc, lay: 'cap', cap: true });
      }
      // spikes
      var spikes = [];
      if (hp.spiky) { var rig = viewRig(yaw); crownClumps(HL.short).forEach(function (cl) { var ci = clumpItems(cl, yaw, hair, hs, hl, rig); if (ci) spikes.push(ci); }); }
      spikes.sort(function (p, q) { return p.d - q.d; });
      // headwear
      var accItems = [];
      if (band) { var ring = []; for (i = 0; i <= 72; i++) ring.push(sph(-180 + i * 5, 62 + 6 * Math.cos((-180 + i * 5) * D2R * 1) * 0 + 4 * Math.sin((-180 + i * 5) * D2R) * 0, 1.11)); strands(ring, yaw, 0.0).forEach(function (d) { accItems.push(Pi(d, 'none', INK, 4.0, 1)); }); strands(ring, yaw, 0.0).forEach(function (d) { accItems.push(Pi(d, 'none', L.trim, 2.4, 1)); }); }
      if (beanie) {
        var bl2 = []; for (i = 0; i < 72; i++) { var az2 = -180 + i * 5; bl2.push(sph(az2, 64 + 8 * Math.sin((az2 - 0) * D2R * 0.5) * 0 + 6 * (1 - Math.cos(az2 * D2R)) / 2, 1.16)); }
        var rgB = region(bl2, yaw, 'cap');
        if (rgB) { accItems.push(Pi(rgB.d, shade(L.cloth, 0.85), INK, 1.7, 1)); var ring2 = bl2.concat([bl2[0]]); strands(ring2, yaw, 0.0).forEach(function (d) { accItems.push(Pi(d, 'none', shade(L.cloth, 0.62), 3.0, 1)); accItems.push(Pi(d, 'none', INK, 0.8, 1, 'translate(0 -1.3)')); }); }
      }
      // ---- assemble layers
      var backCh = [], frontCh = [];
      chains.forEach(function (q) { if (q.cap) return; (q.lay === 'front' ? frontCh : backCh).push(q.o.items); });
      backCh.forEach(function (it) { res.back.push.apply(res.back, it); });
      (function (g) { g.forEach(function (s) { res.back.push.apply(res.back, s.out); }); g.forEach(function (s) { res.back.push.apply(res.back, s.items); }); })(spikes.filter(function (s) { return s.d < 0.2; }));
      res.headFront = [];
      /* lock outlines go under the hair cap: they show only where a lock stands out past the head against skin or sky, and are hidden wherever it overlaps other hair, so there are no stray lines running across the hair */
      spikes.filter(function (s) { return s.d >= 0.2; }).forEach(function (s) { res.headFront.push.apply(res.headFront, s.out); });
      res.headFront.push.apply(res.headFront, beard);
      res.headFront.push.apply(res.headFront, noseTop); /* the nose sits over the moustache and beard outline, never under it */
      res.headFront.push.apply(res.headFront, earTop);
      if (mouthItem && L.beard >= 2 && beard.length) res.headFront.push(mouthItem);
      if (res.capItems) res.headFront.push.apply(res.headFront, res.capItems);
      chains.filter(function (q) { return q.cap; }).forEach(function (q) { res.headFront.push.apply(res.headFront, q.o.items); res.back.push.apply(res.back, q.o.back || []); });
      frontCh.forEach(function (it) { res.headFront.push.apply(res.headFront, it); });
      spikes.filter(function (s) { return s.d >= 0.2; }).forEach(function (s) { res.headFront.push.apply(res.headFront, s.items); });
      if (res._tie) { var tt = res._tie; if (tt.d > -0.2) res.headFront.push(Pi(circ(tt.X, tt.Y, 1.2), L.trim, INK, 1.1, 1)); }
      res.headFront.push.apply(res.headFront, accItems);
      res.capOn = !!res.capItems;
      return res;
    }
    return { build: build, YAW: YAW, prj: prj, sph: sph, region: region };
  })();

  window.AgonArt = {
    shAnim: shAnim,
    ANISH: { small: [28, 54, 21, 3.6], large: [32, 54, 28, 4], pred: [32, 54, 27, 3.6], deer: [30, 55, 25, 3.4], goat: [30, 55, 22, 3.4], grouse: [28, 56, 17, 3], fox: [30, 55, 24, 3.2], bear: [32, 55, 30, 4], snake: [28, 56, 26, 2.8] },
    TIPS: TIPS, tipLookup: tipLookup, tipHTML: tipHTML, attachTips: attachTips,
    INK: INK,
    SKIN: SKIN, HAIR: HAIR, CLOTH: CLOTH,
    SOIL: ['#4a382b', '#463429', '#4e3c2f', '#44311f'],
    GRASS: ['#555f3a', '#4f5a37', '#5a6540'],
    FLOOR: '#5d4838', WALL: '#7a614c', FOREST: '#2f4128', ROCK: '#7d7c73',
    WATER: '#2a4a6a', WATER_EDGE: '#3a6a94',
    TROUSER: '#4a3828', BOOT: '#2a1d15', LEATHER: '#9a8a24', FUR: '#8a7a68',
    SHADE: { step: 0.74, deep: 0.6, light: 1.2 },
    EXPR: EXPR, EXPR_ORDER: EXPR_ORDER, expressionFor: expressionFor,
    MOTION: MOTION, CMX: CMX, cmAnim: cmAnim, anim: anim, ANIMALS: ANIMALS, ANIMAL_VARS: ANIMAL_VARS, VAR_OF: VAR_OF, animalPal: animalPal,
    RIG: RIG, NECK: NECK, TICK_MS: TICK_MS, armJoints: armJoints, shade: shade, tint: tint, mix: mix, lg: lg, pick: pick, look: look, founderAlleles: founderAlleles, applyDis: applyDis, founderDis: founderDis, breed2: breed2, linChild: linChild, inbreeding: inbreeding, skinOf: skinOf, disRows: disRows, disAffected: disAffected, DISORDERS: DISORDERS, coatLum: coatLum, visibility: visibility, founderGe: founderGe, breedGe: breedGe, geStr: geStr, parseGe: parseGe, phenotype: phenotype, geRows: geRows, founderCoat: founderCoat, breedCoat: breedCoat, coatKey: coatKey, coatPheno: coatPheno, coatPal: coatPal, clamp: clamp, hairKit: hairKit, detailKit: detailKit, beardKit: beardKit, outfitKit: outfitKit, HAIRNAMES: HAIRNAMES, OUTFITNAMES: OUTFITNAMES, BEARDNAMES: BEARDNAMES, ACCNAMES: ACCNAMES, TRIM: TRIM,
    BUILDS: BUILDS, BUILD_ORDER: BUILD_ORDER, pawnDims: pawnDims, physique: physique, solid: solid, bodyKit: bodyKit, armKit: armKit, scaleTf: scaleTf, DECAY_TICKS: DECAY_TICKS, DAY_TICKS: DAY_TICKS, PERISH: PERISH, PRESERVE: PRESERVE, spoilRate: spoilRate, spoilRisk: spoilRisk, lotsStep: lotsStep, DECAY_STAGES: DECAY_STAGES, decayOf: decayOf, rotItems: rotItems, skeleton: skeleton, LIGHT: LIGHT, lightModel: lightModel,
    GLYPH: GLYPH, MSG: MSG, glyphPaths: glyphPaths, msgSyms: msgSyms, msgSymNames: msgSymNames, msgText: msgText, msgWants: msgWants, lexSeed: lexSeed, glyphGloss: glyphGloss, PERS: PERS, THOUGHTS: THOUGHTS,
    wornParts: wornParts, legStance: legStance, wornKey: wornKey, MATS: MATS, MAT_ORDER: MAT_ORDER, TPL: TPL, QUALITY: QUALITY, EFFECTS: EFFECTS, qualityOf: qualityOf, craftStats: craftStats, pickMats: pickMats, itemShapes: itemShapes,
    NEEDCOL: NEEDCOL, NAMES: NAMES, genName: genName, genHouse: genHouse, ARCH: ARCH, TRAITS: TRAITS, traitFx: traitFx, archetypesOf: archetypesOf, strengthsFlaws: strengthsFlaws, SCHEM: SCHEM, discoverable: discoverable, discoverChance: discoverChance, MACHINES: MACHINES, LLM_GOALS: LLM_GOALS, llmSystem: llmSystem, parseDecision: parseDecision, mockDecide: mockDecide, MUTATIONS: MUTATIONS, MUT_BY: MUT_BY, applyMutation: applyMutation, rollMutations: rollMutations, breed: breed, mutationLines: mutationLines, GENES: GENES, GENE_GROUPS: GENE_GROUPS, gv: gv,
    THINGS: THINGS, VERBS: VERBS, COMP: COMP, interact: interact, evalNode: evalNode, affordances: affordances, matrix: matrix, TECH: TECH, PRIM: PRIM, MOVES: MOVES, STYLES: STYLES, moveOK: moveOK, styleOf: styleOf, pickMove: pickMove, moveStats: moveStats, techOf: techOf, schemOf: schemOf, machinesOf: machinesOf, techAvail: techAvail, nextTech: nextTech, techFx: techFx, lore: { cls: expClass, LORE: LORE, MAT2T: MAT2T, cands: expCands, pick: expPick, odds: expOdds, derive: loreDerive, have: loreHave, front: loreFront, teach: loreTeach, label: sigLabel, info: sigInfo, yields: sigYields, hit: patHit }, mind: MIND, thoughtLine: thoughtLine, geneDetail: geneDetail, thoughtIcons: thoughtIcons, glyphNames: function () { return Object.keys(GLYPH); }
  };
})();
// Agon macro (region) map: one planet cell refined into a MW x MH heightfield. 1 region cell = CELL tiles.
// Everything is driven by the planet grid P (elev, sea, tmp, wet) sampled bilinearly, and by noise in WORLD coordinates (planet cell * MW + cell),
// so neighbouring regions share their borders and the region's mean climate/altitude matches its planet cell.
(function (root) {
  const CELL = 256;                       // tiles per region cell
  const MW = 320, MH = 320;               // region grid (one planet cell), 164 km square at 2 m per tile
  const HREAL = 6500;                     // metres for normalised height 1.0
  const hash = (x, y, k) => { let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(k | 0, 2147483647)) | 0; h = Math.imul(h ^ (h >>> 13), 1274126177); h ^= h >>> 16; return (h >>> 0) / 4294967296; };
  const mulberry = (a) => () => { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const smooth = (t) => t * t * (3 - 2 * t), clamp01 = (v) => Math.max(0, Math.min(1, v));
  // planet field sampled at planet coordinates (u, v) in cells; cell centres sit at +0.5
  const psample = (P, f, u, v) => { const x = Math.max(0, Math.min(P.PW - 1.0001, u - 0.5)), y = Math.max(0, Math.min(P.PH - 1.0001, v - 0.5)), x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0, a = P[f][y0 * P.PW + x0], b = P[f][y0 * P.PW + x0 + 1], c = P[f][(y0 + 1) * P.PW + x0], d = P[f][(y0 + 1) * P.PW + x0 + 1]; return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy; };

  function genMacro(P, px, py, pseed) {
    const N = MW * MH, seed = pseed | 0, ox = px * MW, oy = py * MH;
    const vn = (x, y, k) => { const ix = Math.floor(x), iy = Math.floor(y), fx = smooth(x - ix), fy = smooth(y - iy), kk = k + seed * 7; const a = hash(ix, iy, kk), b = hash(ix + 1, iy, kk), c = hash(ix, iy + 1, kk), d = hash(ix + 1, iy + 1, kk); return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy; };
    const fbm = (x, y, k, o = 5) => { let s = 0, a = 0.5, f = 1, t = 0; for (let i = 0; i < o; i++) { s += a * vn(x * f, y * f, k + i); t += a; a *= 0.5; f *= 2.03; } return s / t; };
    const ridgeN = (x, y, k) => 1 - Math.abs(2 * fbm(x, y, k, 4) - 1);
    const Hb = new Float32Array(N), Uup = new Float32Array(N), Hh = new Float32Array(N), fixd = new Uint8Array(N), Tp = new Float32Array(N), Wp = new Float32Array(N), volc = new Float32Array(N), relA = new Float32Array(N);
    for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) {
      const k = y * MW + x, u = px + (x + 0.5) / MW, v = py + (y + 0.5) / MH, wx = ox + x, wy = oy + y;
      const d = psample(P, 'elev', u, v) - P.sea, base = d > 0 ? d * 2600 : d * 6000;      // metres relative to sea level
      Hb[k] = base; Tp[k] = -12 + 45 * psample(P, 'tmp', u, v); Wp[k] = psample(P, 'wet', u, v);
      const mk = smooth(clamp01((d - 0.05) / 0.2));
      // cordilleras: ridged noise lines, domain-warped, in world coordinates; strength follows the planet's highland mask
      const wx2 = wx + 45 * (fbm(wx * 0.009, wy * 0.009, 3) - 0.5), wy2 = wy + 45 * (fbm(wx * 0.009 + 7, wy * 0.009 + 3, 4) - 0.5);
      const band = ridgeN(wx2 * 0.0105, wy2 * 0.0105, 11), crest = Math.pow(band, 1.8) * (0.55 + 0.7 * ridgeN(wx2 * 0.05, wy2 * 0.05, 12));
      Uup[k] = d <= 0 ? 0 : mk * (0.2 + 0.95 * crest) + Math.min(0.05, d * 0.2);
      Hh[k] = Hb[k] / HREAL + Uup[k] * 0.3 + (fbm(wx * 0.05, wy * 0.05, 9) - 0.5) * 0.12 * (0.3 + Uup[k]) + (hash(wx, wy, 91) - 0.5) * 0.02 * (d > 0 ? 1 : 0);
      if (d > 0) {                                                                       // relief provinces and formations, world-coordinate functions so regions agree at their borders
        const rg = smooth(clamp01((fbm(wx * 0.011 + 90, wy * 0.011, 21, 3) - 0.3) / 0.4));               // ruggedness province: flat plains next to broken hill country
        let rel = (fbm(wx * 0.035, wy * 0.035, 22, 4) - 0.5) * 0.2 * (0.2 + rg);                         // rolling hills and low ranges
        const mp = smooth(clamp01((fbm(wx * 0.008 + 30, wy * 0.008 + 70, 23, 3) - 0.5) / 0.12));          // mesa / plateau country: terraced steps with cliffed rims
        if (mp > 0) { const tt = fbm(wx * 0.02, wy * 0.02, 24, 3) * 6, fl = Math.floor(tt); rel += mp * 0.02 * (fl + smooth(clamp01((tt - fl - 0.8) / 0.2))); }
        const gx = Math.floor(wx / 70), gy = Math.floor(wy / 70);                                          // volcanic cones, roughly one in four lattice cells
        for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const lx = gx + i, ly = gy + j; if (hash(lx, ly, 25 + seed) < 0.74) continue; const vcx = (lx + 0.2 + 0.6 * hash(lx, ly, 26 + seed)) * 70, vcy = (ly + 0.2 + 0.6 * hash(lx, ly, 27 + seed)) * 70, RR = 9 + 13 * hash(lx, ly, 28 + seed), r = Math.hypot(wx - vcx, wy - vcy); if (r < RR) { const q = 1 - r / RR; rel += (0.05 + 0.17 * hash(lx, ly, 29 + seed)) * Math.pow(q, 1.5) * (r < RR * 0.09 ? 0.75 : 1); volc[k] = Math.max(volc[k], q); } }
        relA[k] = rel;                                                                                    // added after erosion: a pure function of world position, so region borders stay seamless
      }
      if (d <= 0) Hh[k] = Hb[k] / HREAL;                                                  // ocean floor
      fixd[k] = d <= 0 || x === 0 || y === 0 || x === MW - 1 || y === MH - 1 ? 1 : 0;
      if (fixd[k] && d > 0) Hh[k] -= (hash(wx, wy, 91) - 0.5) * 0.02;                      // border cells stay smooth: white-noise jitter would not erode there and would show as a seam
    }
    // --- stream-power erosion (implicit) on a priority-flood drainage tree; sea and map edges are fixed base levels
    const heap = new Int32Array(N + 8), hk = new Float32Array(N + 8), key = new Float32Array(N); let hs = 0;   // hk[p] = key[heap[p]], kept beside the heap so comparisons skip an indirection (same order, same ties)
    const hpush = (i) => { let p = hs++; const ki = key[i]; while (p > 0) { const q = (p - 1) >> 1; if (hk[q] <= ki) break; heap[p] = heap[q]; hk[p] = hk[q]; p = q; } heap[p] = i; hk[p] = ki; };
    const hpop = () => { const top = heap[0], last = heap[--hs], kl = hk[hs]; let p = 0; for (;;) { let c = p * 2 + 1; if (c >= hs) break; if (c + 1 < hs && hk[c + 1] < hk[c]) c++; if (hk[c] >= kl) break; heap[p] = heap[c]; hk[p] = hk[c]; p = c; } heap[p] = last; hk[p] = kl; return top; };
    const recv = new Int32Array(N), order = new Int32Array(N), F = new Float32Array(N), area = new Float32Array(N), done = new Uint8Array(N);
    const NBX = [1, -1, 0, 0, 1, -1, 1, -1], NBY = [0, 0, 1, -1, 1, 1, -1, -1];
    const drain = () => {
      done.fill(0); hs = 0; let no = 0;
      for (let k = 0; k < N; k++) if (fixd[k]) { F[k] = Hh[k]; key[k] = F[k]; recv[k] = -1; done[k] = 1; hpush(k); }
      while (hs) { const c = hpop(); order[no++] = c; const cx = c % MW, cy = (c / MW) | 0;
        const Fc = F[c] + 1e-4;
        for (let d = 0; d < 8; d++) { const nx = cx + NBX[d], ny = cy + NBY[d]; if (nx < 0 || ny < 0 || nx >= MW || ny >= MH) continue; const n = ny * MW + nx; if (done[n]) continue; done[n] = 1; recv[n] = c; F[n] = Math.max(Hh[n], Fc); key[n] = F[n]; hpush(n); } }
      area.fill(1); for (let i = N - 1; i >= 0; i--) { const c = order[i]; if (recv[c] >= 0) area[recv[c]] += area[c]; }
    };
    const K = 0.07, ITER = 28;
    for (let k = 0; k < N; k++) if (fixd[k] && Hb[k] > 0) Hh[k] += Uup[k] * 0.018 * ITER * 0.6;   // borders sit part-way up the uplift so they do not form a cliff
    for (let it = 0; it < ITER; it++) {
      for (let k = 0; k < N; k++) if (Hb[k] > 0 && !fixd[k]) Hh[k] += Uup[k] * 0.018;                 // rock uplift; borders keep their initial value (a pure function of world coords) so they drain and match the neighbour
      drain();
      for (let i = 0; i < N; i++) { const c = order[i], r = recv[c]; if (r < 0) continue; const ex = Math.hypot((c % MW) - (r % MW), ((c / MW) | 0) - ((r / MW) | 0)), f = K * Math.sqrt(area[c]) / ex; Hh[c] = (Hh[c] + f * Hh[r]) / (1 + f); }
      if (it % 2) { const t = Float32Array.from(Hh); for (let y = 1; y < MH - 1; y++) for (let x = 1; x < MW - 1; x++) { const k = y * MW + x; if (!fixd[k]) Hh[k] = t[k] + 0.12 * ((t[k - 1] + t[k + 1] + t[k - MW] + t[k + MW]) / 4 - t[k]); } }
    }
    drain();
    const Hm = new Float32Array(N), lake = new Float32Array(N);
    for (let k = 0; k < N; k++) { Hm[k] = Hh[k] * HREAL; if (Hb[k] > 0) { const rm = relA[k] * HREAL, h0 = Math.max(Hm[k], 0); Hm[k] = h0 + (rm > 0 ? rm * Math.min(1, 0.25 + h0 / 60) : Math.max(rm, -0.45 * h0)); Hm[k] = Math.max(Hm[k], 2 + Math.min(1, Hb[k] / 150) * 10); } }
    for (let k = 0; k < N; k++) Hh[k] = Hm[k] / HREAL;                       // drain again over the final surface, so rivers and lakes follow the hills and basins that were just added
    drain(); for (let k = 0; k < N; k++) lake[k] = Hb[k] > 0 ? Math.max(0, F[k] - Hh[k] - 0.0012) * HREAL : 0;   // basins shallower than ~8 m are marsh, not lakes
    // --- climate from the planet: mean temperature and wetness come from the planet cell; altitude and rain shadow vary inside it
    const T = new Float32Array(N), Rn = new Float32Array(N);
    for (let y = 0; y < MH; y++) { let moist = 0.35 + 0.6 * Wp[y * MW]; for (let x = 0; x < MW; x++) { const k = y * MW + x, sea = Hb[k] <= 0; T[k] = Tp[k] - (sea ? 0 : 6.5 * (Hm[k] - Hb[k]) / 1000) + (fbm((ox + x) * 0.03, (oy + y) * 0.03, 4) - 0.5) * 3; const nx = Math.min(MW - 1, x + 1), lift = Math.max(0, Hm[y * MW + nx] - Hm[k]) / 300, p = Math.min(moist, lift * 0.12) + 0.04 * moist; moist = Math.min(1, moist - p + (sea ? 0.05 : Hm[k] < 900 ? 0.02 * (0.4 + Wp[k]) : 0.004)); Rn[k] = Math.max(0.02, Math.min(1, Wp[k] * 0.7 + p * 2.6)); } }
    // --- geology (world coordinates): 0 granite 1 limestone 2 sandstone 3 basalt 4 schist
    const G = new Uint8Array(N), cave = new Float32Array(N);
    for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { const k = y * MW + x, z = fbm((ox + x) * 0.02, (oy + y) * 0.02, 60), core = Uup[k]; G[k] = volc[k] > 0.12 ? 3 : core > 0.7 ? (z < 0.5 ? 0 : 4) : z < 0.3 ? 2 : z < 0.62 ? 1 : z < 0.78 ? 4 : 3; const g = G[k]; cave[k] = (g === 1 ? 0.9 : g === 3 ? 0.45 : g === 2 ? 0.3 : 0.1) * (0.4 + 0.6 * Rn[k]) * Math.min(1, Math.max(0, Hm[k]) / 600) * (0.6 + 0.8 * fbm((ox + x) * 0.1, (oy + y) * 0.1, 70)); }
    // --- camp sites: flat-ish, watered valleys at moderate altitude with big relief nearby; best first, kept apart
    const slope = (x, y) => { const k = y * MW + x; return Math.hypot(Hm[k + 1] - Hm[k - 1], Hm[k + MW] - Hm[k - MW]) / 2; };
    const cand = [];
    for (let y = 20; y < MH - 20; y += 2) for (let x = 20; x < MW - 20; x += 2) { const k = y * MW + x; if (Hm[k] < 8 || Hm[k] > 2600 || lake[k] > 0 || Hb[k] <= 0) continue; let relief = 0, wet = 0; for (let a = 0; a < 8; a++) { const q = (a * Math.PI) / 4, hx = Math.max(0, Math.min(MW - 1, Math.round(x + Math.cos(q) * 14))), hy = Math.max(0, Math.min(MH - 1, Math.round(y + Math.sin(q) * 14))); relief = Math.max(relief, Hm[hy * MW + hx] - Hm[k]); if (Hm[hy * MW + hx] <= 0) wet = 0.4; } cand.push([Math.min(relief, 2200) / 1000 + Math.min(3, Math.log(area[k])) * 0.5 - slope(x, y) / 60 + hash(x, y, 33) * 0.2 + (T[k] > 3 && T[k] < 24 ? 1 : -3) + wet, x, y]); }
    cand.sort((a, b) => b[0] - a[0]); const sites = [];
    for (const [, x, y] of cand) { if (sites.length >= 40) break; if (sites.every((q) => Math.hypot(q[0] - x, q[1] - y) > 12)) sites.push([x, y]); }
    // high camps (alpine / highlands wants): steep-sided shelves at altitude
    const hi = []; for (let y = 20; y < MH - 20; y += 3) for (let x = 20; x < MW - 20; x += 3) { const k = y * MW + x; if (Hm[k] < 2000 || Hm[k] > 5400 || lake[k] > 0) continue; hi.push([Hm[k] / 1000 - slope(x, y) / 40 + hash(x, y, 34) * 0.5, x, y]); }
    hi.sort((a, b) => b[0] - a[0]); const nHi = sites.length; for (const [, x, y] of hi) { if (sites.length >= nHi + 24) break; if (sites.every((q) => Math.hypot(q[0] - x, q[1] - y) > 10)) sites.push([x, y]); }
    if (!sites.length) sites.push([MW >> 1, MH >> 1]);
    // regional climate belts, added after site and cave choice so those stay as they were: a planet cell is 164 km, and real temperature and rain also drift over about 100 km, so a
    // zoomed-out map shows steppe, forest and dry zones side by side instead of one tone per planet cell (world coordinates, so neighbouring regions agree)
    for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { const k = y * MW + x, wx = ox + x, wy = oy + y; T[k] += (fbm(wx * 0.0042, wy * 0.0042, 31, 3) - 0.5) * 16 + (fbm(wx * 0.017 + 11, wy * 0.017, 34, 3) - 0.5) * 9; Rn[k] = Math.max(0.02, Math.min(1, Rn[k] * (1 + (fbm(wx * 0.0037 + 50, wy * 0.0037, 32, 3) - 0.5) * 2.6) * (1 + (fbm(wx * 0.017 + 70, wy * 0.017 + 5, 33, 3) - 0.5) * 2.4))); }
    // rain shadow: prevailing wind blows west to east. A running upwind crest height (it decays 25 m per cell, so a ridge shades the ~40 cells behind it) is compared with the ground here:
    // lee-side plains dry out, windward slopes catch extra rain. It only looks a short way upwind, so regions agree at their borders.
    for (let y = 0; y < MH; y++) { let crest = Hm[y * MW]; for (let x = 0; x < MW; x++) { const k = y * MW + x; crest = Math.max(Hm[k], crest - 25); if (Hb[k] <= 0) { crest = Hm[k]; continue; } const dry = Math.max(0, Math.min(1, (crest - Hm[k] - 40) / 520)), lift = x > 3 ? Math.max(0, Math.min(1, (Hm[k] - Hm[k - 4]) / 140)) : 0; Rn[k] = Math.max(0.02, Math.min(1, Rn[k] * (1 - 0.6 * dry) * (1 + 0.55 * lift) * (1 + 0.12 * Math.min(2, Math.max(0, Hm[k]) / 1000)))); } }
    return { MW, MH, CELL, HREAL, H: Hm, U: Uup, area, lake, T, R: Rn, G, cave, sites, site: sites[0], px, py, seed, recv, sea: Hb };
  }
  // region cell -> value, bilinear in tile coordinates
  function sample(M, f, tx, ty) { const x = tx / CELL - 0.5, y = ty / CELL - 0.5, x0 = Math.max(0, Math.min(M.MW - 2, Math.floor(x))), y0 = Math.max(0, Math.min(M.MH - 2, Math.floor(y))), fx = Math.max(0, Math.min(1, x - x0)), fy = Math.max(0, Math.min(1, y - y0)); const a = M[f][y0 * M.MW + x0], b = M[f][y0 * M.MW + x0 + 1], c = M[f][(y0 + 1) * M.MW + x0], d = M[f][(y0 + 1) * M.MW + x0 + 1]; return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy; }
  const GEO = ['Granite', 'Limestone', 'Sandstone', 'Basalt', 'Schist'];
  // Catmull-Rom bicubic: C1-continuous, so slope (and hillshade) has no creases along cell edges when zoomed in
  function sampleC(M, f, wx, wy) {
    const A = M[f], x = wx - 0.5, y = wy - 0.5, ix = Math.floor(x), iy = Math.floor(y), tx = x - ix, ty = y - iy, MW = M.MW, MH1 = M.MH - 1, MW1 = MW - 1;
    const tx2 = tx * tx, tx3 = tx2 * tx, ty2 = ty * ty, ty3 = ty2 * ty;
    const a0 = -0.5 * tx3 + tx2 - 0.5 * tx, a1 = 1.5 * tx3 - 2.5 * tx2 + 1, a2 = -1.5 * tx3 + 2 * tx2 + 0.5 * tx, a3 = 0.5 * tx3 - 0.5 * tx2;
    const b0 = -0.5 * ty3 + ty2 - 0.5 * ty, b1 = 1.5 * ty3 - 2.5 * ty2 + 1, b2 = -1.5 * ty3 + 2 * ty2 + 0.5 * ty, b3 = 0.5 * ty3 - 0.5 * ty2;
    const c0 = Math.max(0, Math.min(MW1, ix - 1)), c1 = Math.max(0, Math.min(MW1, ix)), c2 = Math.max(0, Math.min(MW1, ix + 1)), c3 = Math.max(0, Math.min(MW1, ix + 2));
    let r = 0;
    for (let j = 0; j < 4; j++) {
      const o = Math.max(0, Math.min(MH1, iy - 1 + j)) * MW; let row = 0;
      row += a0 * A[o + c0]; row += a1 * A[o + c1]; row += a2 * A[o + c2]; row += a3 * A[o + c3];
      r += (j === 0 ? b0 : j === 1 ? b1 : j === 2 ? b2 : b3) * row;
    }
    return r;
  }
  function cw(t) { const t2 = t * t, t3 = t2 * t; return [-0.5 * t3 + t2 - 0.5 * t, 1.5 * t3 - 2.5 * t2 + 1, -1.5 * t3 + 2 * t2 + 0.5 * t, 0.5 * t3 - 0.5 * t2]; }
  // the world surface in metres at region coordinates (cells): bicubic macro heights plus detail that depends only on world position,
  // so every tier (local engine, satellite layers, area view) reads the same ground
  const vnS = (x, y) => { const a = Math.floor(x), b = Math.floor(y), fx = x - a, fy = y - b, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), h = (i, j) => hash(a + i, b + j, 77); return (h(0, 0) * (1 - u) + h(1, 0) * u) * (1 - v) + (h(0, 1) * (1 - u) + h(1, 1) * u) * v; };
  // gradient (Perlin) noise: band-limited around 1/wavelength, so summed octaves keep the intended fBm spectrum better than value noise
  const gnS = (x, y, k) => {
    const a = Math.floor(x), b = Math.floor(y), fx = x - a, fy = y - b, u = fx * fx * fx * (fx * (fx * 6 - 15) + 10), v = fy * fy * fy * (fy * (fy * 6 - 15) + 10), kk = 91 + k, fx1 = fx - 1, fy1 = fy - 1;
    let t = hash(a, b, kk) * 6.2831853; const d00 = Math.cos(t) * (fx - 0) + Math.sin(t) * (fy - 0);
    t = hash(a + 1, b, kk) * 6.2831853; const d10 = Math.cos(t) * fx1 + Math.sin(t) * (fy - 0);
    t = hash(a, b + 1, kk) * 6.2831853; const d01 = Math.cos(t) * (fx - 0) + Math.sin(t) * fy1;
    t = hash(a + 1, b + 1, kk) * 6.2831853; const d11 = Math.cos(t) * fx1 + Math.sin(t) * fy1;
    return (d00 * (1 - u) + d10 * u) * (1 - v) + (d01 * (1 - u) + d11 * u) * v;
  };
  // Roughness anchor: the macro grid resolves wavelengths down to 2 cells (1024 m). Its curvature there (4-neighbour Laplacian, 3x3 smoothed) gives the amplitude of
  // the shortest wave it can hold; finer octaves continue the fBm power law A(l) = A0 (l / l0)^H, H = 0.85 (profile PSD ~ k^-(2H+1) = k^-2.7), down to 4 m.
  const HURST = 0.92, L0 = 2 * CELL * 2;
  function roughOf(M) {
    if (M.rough) return M.rough;
    const { MW, MH, H } = M, Lp = new Float32Array(MW * MH), R = new Float32Array(MW * MH);
    for (let y = 1; y < MH - 1; y++) for (let x = 1; x < MW - 1; x++) { const k = y * MW + x; Lp[k] = Math.abs(H[k - 1] + H[k + 1] + H[k - MW] + H[k + MW] - 4 * H[k]) * 0.25; }
    for (let y = 2; y < MH - 2; y++) for (let x = 2; x < MW - 2; x++) { let a = 0; for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) a += Lp[(y + j) * MW + x + i]; R[y * MW + x] = a / 25 * 1.1 + 2.5; }
    for (let x = 0; x < MW; x++) for (let y = 0; y < MH; y++) if (x < 2 || y < 2 || x >= MW - 2 || y >= MH - 2) R[y * MW + x] = 8;
    return (M.rough = R);
  }
  // octave wavelengths in metres; value noise has std ~0.2 so (n - 0.5) * 2.4 is ~unit std
  function surface(M, wx, wy, fine) {
    roughOf(M); const R0 = sampleC(M, 'rough', wx, wy); let h = sampleC(M, 'H', wx, wy); const dmp = Math.max(0.1, Math.min(1, Math.abs(h) / 30));   // fine detail fades toward the shoreline so coasts stay clean inlets, not ragged flooded flats
    const kmax = fine ? 8 : 4;
    for (let k = 1; k <= kmax; k++) { const lam = L0 / Math.pow(2, k), a = R0 * Math.pow(lam / L0, HURST), sc = lam / (CELL * 2); h += a * dmp * gnS(wx / sc + k * 3.7, wy / sc + k * 1.9, k) * 2.6; }
    return h;
  }
  // Ground cover from climate: the ONE rule every tier uses. R rainfall 0..1, T mean temperature C, hm altitude m. g = fraction of land under grass or woods
  // (everything else is bare soil), f = fraction under forest (a subset of g). The local engine thresholds its terrain to these fractions and the map layers
  // colour themselves with the same mix, so the region around a camp looks like the camp.
  const sm01 = (t) => { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); };
  function cover(R, T, hm) {
    const warm = sm01((T + 9) / 10), alt = Math.max(0, Math.min(1, (4300 - hm) / 2600)), veg = sm01((R - 0.1) / 0.38) * warm * (0.35 + 0.65 * alt);
    const g = 0.05 + 0.9 * veg, f = g * 0.95 * sm01((R - 0.3) / 0.22) * warm * alt * sm01((T + 2.5) / 4.5);
    const sav = sm01((T - 17) / 4) * sm01((R - 0.25) / 0.1) * (1 - sm01((R - 0.5) / 0.1)) * 0.07;   // scattered savanna trees
    return { g: +Math.min(0.97, g).toFixed(3), f: +Math.min(g * 0.92, Math.max(f, sav)).toFixed(3) };
  }
  const snowOf = (T, hm) => Math.max(sm01((-4 - T) / 5), sm01((hm - 3300) / 700));   // snow cover fraction: polar/cold ground and high summits
  const COV_SOIL = [84, 64, 48], COV_GRASS = [92, 108, 62], COV_FOREST = [50, 84, 46], COV_SAND = [205, 176, 120];
  // Climate ground palette: the soil, grass, forest-floor and sand colours a place has, from its rainfall R, temperature T and rock. Laterite red in the wet tropics, ochre and pale sand in
  // dry heat, savanna straw where it is warm and semi-dry, grey-brown cold soil and moss in the cold, ash over basalt. Every tier reads this, so a desert looks like one at every zoom.
  function ground(R, T, hm, geo) {
    const lerp = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
    const warm = sm01((T - 14) / 10), cold = sm01((6 - T) / 10), dry = 1 - sm01((R - 0.2) / 0.25), wetw = sm01((R - 0.45) / 0.2), sv = warm * (1 - sm01((R - 0.32) / 0.22));
    let soil = lerp(COV_SOIL, [126, 72, 46], warm * wetw); const bad = geo === 2 && R < 0.29 && T > 10 ? 1 : 0; soil = lerp(soil, [176, 138, 94], Math.max(warm * dry, dry * 0.45)); soil = lerp(soil, [146, 98, 62], sv * 0.6); soil = lerp(soil, [98, 90, 80], cold); if (geo === 3) soil = lerp(soil, [62, 54, 54], 0.65); soil = lerp(soil, [139, 84, 64], bad);
    const ste = (1 - cold) * (1 - sm01((R - 0.3) / 0.15)); let grass = lerp(COV_GRASS, [190, 174, 88], Math.max(sv, ste * 0.8)); grass = lerp(grass, [74, 114, 50], wetw * (1 - cold) * (1 - sv)); grass = lerp(grass, R > 0.3 ? [104, 128, 80] : [112, 112, 88], cold * 0.8);
    let forest = lerp(COV_FOREST, [28, 98, 52], warm * wetw); forest = lerp(forest, [36, 68, 62], cold);
    return { soil, grass, forest, sand: lerp(COV_SAND, [200, 138, 90], warm * dry * 0.6), rich: soil.map((v) => v * 0.74), grav: lerp(soil, [128, 120, 108], 0.55), marsh: [58, 70, 52] };
  }

  // Terrain-derived ecology fields on the macro grid: slope, valley/ridge curvature, north-facing index and a wetness index from drainage area and lakes. They steer the
  // effective rainfall a spot gets (valley floors and north slopes are wetter, ridges and steep faces drier), rock exposure, wetlands and dunes, so vegetation follows the ground.
  function ecoFields(M) {
    if (M.eSl) return M;
    const { MW, MH, H } = M, N = MW * MH, sl = new Float32Array(N), cv = new Float32Array(N), asp = new Float32Array(N), wet = new Float32Array(N), w0 = new Float32Array(N), D = CELL * 2;
    const hx = (x, y) => H[Math.max(0, Math.min(MH - 1, y)) * MW + Math.max(0, Math.min(MW - 1, x))];
    for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) {
      const k = y * MW + x, gx = hx(x + 1, y) - hx(x - 1, y), gy = hx(x, y + 1) - hx(x, y - 1);
      sl[k] = Math.hypot(gx, gy) / (2 * D); asp[k] = Math.max(-1, Math.min(1, gy / (2 * D) / 0.12));
      cv[k] = Math.max(-1, Math.min(1, ((hx(x + 2, y) + hx(x - 2, y) + hx(x, y + 2) + hx(x, y - 2)) / 4 - hx(x, y)) / 45));
      w0[k] = Math.max(M.lake[k] > 0 ? 1 : 0, Math.max(0, Math.min(1, Math.log10(Math.max(1, M.area[k]) / 25) / 2.2)));
    }
    for (let pass = 0; pass < 2; pass++) { const src = pass ? wet : w0, dst = pass ? w0 : wet; for (let y = 0; y < MH; y++) for (let x = 0; x < MW; x++) { let m = 0, a = 0; for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) { const q = Math.max(0, Math.min(MH - 1, y + j)) * MW + Math.max(0, Math.min(MW - 1, x + i)); m = Math.max(m, src[q]); a += src[q]; } dst[y * MW + x] = 0.55 * m * 0.9 + 0.45 * a / 9; } }
    M.eSl = sl; M.eCv = cv; M.eAsp = asp; M.eWet = w0; return M;
  }
  function ecoAt(M, wx, wy) { return { sl: sample(M, 'eSl', wx * CELL, wy * CELL), cv: sample(M, 'eCv', wx * CELL, wy * CELL), asp: sample(M, 'eAsp', wx * CELL, wy * CELL), wet: sample(M, 'eWet', wx * CELL, wy * CELL) }; }
  // T, R and (over land) the four ecology fields all sample the macro grid at the same point: share the cell lookup and weights (same arithmetic as sample())
  const _bf = { T: 0, R: 0, sl: 0, cv: 0, asp: 0, wet: 0 };
  function sampleAt(M, wx, wy, eco) {
    const tx = wx * CELL, ty = wy * CELL, x = tx / CELL - 0.5, y = ty / CELL - 0.5, x0 = Math.max(0, Math.min(M.MW - 2, Math.floor(x))), y0 = Math.max(0, Math.min(M.MH - 2, Math.floor(y))), fx = Math.max(0, Math.min(1, x - x0)), fy = Math.max(0, Math.min(1, y - y0));
    const o0 = y0 * M.MW + x0, o1 = o0 + 1, o2 = (y0 + 1) * M.MW + x0, o3 = o2 + 1, fxy = fx * fy;
    let A = M.T, a = A[o0], b = A[o1], c = A[o2], d = A[o3]; _bf.T = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
    A = M.R; a = A[o0]; b = A[o1]; c = A[o2]; d = A[o3]; _bf.R = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
    if (eco) {
      A = M.eSl; a = A[o0]; b = A[o1]; c = A[o2]; d = A[o3]; _bf.sl = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
      A = M.eCv; a = A[o0]; b = A[o1]; c = A[o2]; d = A[o3]; _bf.cv = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
      A = M.eAsp; a = A[o0]; b = A[o1]; c = A[o2]; d = A[o3]; _bf.asp = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
      A = M.eWet; a = A[o0]; b = A[o1]; c = A[o2]; d = A[o3]; _bf.wet = a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fxy;
    }
    return _bf;
  }
  // Incremental painter: step(n) paints the next n rows (heights first, one row ahead for the slope lookups) so callers can spread a big image over many short tasks.
  // paintRect() below runs it to the end, so both give identical pixels.
  function paintRectInc(M, x0, y0, wc, hc, S, ci, opts) {
    const OPT = opts || {}; roughOf(M); ecoFields(M); const { MW, MH } = M, W = Math.round(wc * S), Hh = Math.round(hc * S), px = new Uint8ClampedArray(W * Hh * 4), GC = [[150, 146, 140], [214, 206, 176], [206, 156, 98], [86, 82, 90], [126, 120, 136]];
    const det = S >= 6, fine = S >= 48, vn = (x, y) => { const a = Math.floor(x), b = Math.floor(y), fx = x - a, fy = y - b, u = fx * fx * (3 - 2 * fx), v = fy * fy * (3 - 2 * fy), h = (i, j) => hash(a + i, b + j, 77); return (h(0, 0) * (1 - u) + h(1, 0) * u) * (1 - v) + (h(0, 1) * (1 - u) + h(1, 1) * u) * v; };
    // heights on a padded grid (one extra ring) so slopes and contours are cheap lookups. Detail noise is in world units, so it looks the same at every zoom.
    const G1 = W + 2, HG = new Float32Array(G1 * (Hh + 2));
    let hgRow = -1, py = 0;
    const fillHG = (upto) => { for (; hgRow <= upto; hgRow++) { const y = hgRow; for (let x = -1; x <= W; x++) { const wx = x0 + x / S, wy = y0 + y / S; HG[(y + 1) * G1 + x + 1] = det ? surface(M, wx, wy, fine) : sample(M, 'H', wx * CELL, wy * CELL); } } };
    const hv = (x, y) => HG[(y + 1) * G1 + x + 1], mpp = CELL * 2 / S, K = S <= 3 ? 3.2 : 2.3;
    const step = (n) => {
    const end = Math.min(Hh, py + n); fillHG(Math.min(Hh, end));
    for (let y = py; y < end; y++) for (let x = 0; x < W; x++) {
      const wx = x0 + x / S, wy = y0 + y / S, h = hv(x, y), edge = x < 1 || y < 1 || x >= W - 1 || y >= Hh - 1, hx = hv(x + 1, y) - hv(x - 1, y), hy = hv(x, y + 1) - hv(x, y - 1), sh = S >= 64 ? Math.max(0.72, Math.min(1.28, 0.95 + (hx + hy) / (2 * mpp) * 1.1)) : Math.max(0.45, Math.min(1.4, 0.88 + (hx + hy) / (2 * mpp) * K));
      const jx = det ? (vn(wx / 0.5 + 40, wy / 0.5) - 0.5) * 0.7 + (vn(wx / 0.09 + 11, wy / 0.09) - 0.5) * 0.3 : 0, jy = det ? (vn(wx / 0.5, wy / 0.5 + 40) - 0.5) * 0.7 + (vn(wx / 0.09, wy / 0.09 + 11) - 0.5) * 0.3 : 0, mx = Math.max(0, Math.min(MW - 1, Math.floor(wx + jx))), my = Math.max(0, Math.min(MH - 1, Math.floor(wy + jy))), k = my * MW + mx, BF = sampleAt(M, wx, wy, h > 0), T = BF.T, R = BF.R, EF = h > 0 ? BF : null, Re = EF ? Math.max(0.02, Math.min(1, R * (1 + 0.5 * Math.max(0, EF.cv) + 0.55 * EF.wet + 0.18 * EF.asp - 0.38 * Math.max(0, -EF.cv)) * (1 + (vn(wx / 6 + 3, wy / 6) - 0.5) * 0.55 + (vn(wx / 1.8 + 8, wy / 1.8 + 2) - 0.5) * 0.35 + (S >= 6 && S < 64 ? (vn(wx / 0.8 + 21, wy / 0.8 + 4) - 0.5) * 0.5 + (vn(wx / 0.3 + 5, wy / 0.3 + 13) - 0.5) * 0.3 : 0)))) : R;
      let c, shx = 1;
      if (h <= 0) { const dp = Math.min(1, -h / 1500); c = [48 - 20 * dp, 96 - 30 * dp, 160 - 40 * dp]; if (OPT.mask) OPT.mask[y * W + x] = 3; }
      else if (T < -9 || h > 3900) c = [240, 243, 248];
      else if (!(S >= 64 && OPT.snow) && snowOf(T, h) > 0.01) { const gR = [96, 100, 80], sn = snowOf(T, h) * 0.85; c = [gR[0] + (240 - gR[0]) * sn, gR[1] + (243 - gR[1]) * sn, gR[2] + (248 - gR[2]) * sn]; }
      else { let g = GC[M.G[k]]; if (fine) { const cx = wx + jx - 0.5, cy = wy + jy - 0.5, ix = Math.max(0, Math.min(MW - 2, Math.floor(cx))), iy = Math.max(0, Math.min(MH - 2, Math.floor(cy))), fx = Math.max(0, Math.min(1, cx - ix)), fy = Math.max(0, Math.min(1, cy - iy)), q = iy * MW + ix, g0 = GC[M.G[q]], g1 = GC[M.G[q + 1]], g2 = GC[M.G[q + MW]], g3 = GC[M.G[q + MW + 1]]; g = [0, 1, 2].map((i) => (g0[i] * (1 - fx) + g1[i] * fx) * (1 - fy) + (g2[i] * (1 - fx) + g3[i] * fx) * fy); } const cvr = cover(Re, T, h), gd = ground(Re, T, h, M.G[k]), gr = (() => { let wg = cvr.g, wf = cvr.f, sc = gd.soil; if (OPT.cov && (S >= 64 || OPT.covAny)) { const b = OPT.cov.b(wx, wy) + 3.5 * Math.max(-0.6, Math.min(0.6, (Re - R) / Math.max(R, 0.12))); const sf = S >= 300 ? 0.06 : 0.35; wg = sm01((b - OPT.cov.gth) / sf + 0.5); wf = OPT.cov.fth > 50 ? 0 : Math.min(wg, sm01((b - OPT.cov.fth) / sf + 0.5)); } if (OPT.clear && S >= 64 && OPT.clear(wx, wy)) { wg = 0; wf = 0; } else if (OPT.fert && S >= 64) { const v = OPT.fert.f(wx, wy); if (v > OPT.fert.rich) sc = R > 0.62 && h < 300 ? gd.marsh : gd.rich; else if (v < OPT.fert.grav) sc = gd.grav; } return [0, 1, 2].map((i) => sc[i] * (1 - wg) + gd.grass[i] * (wg - wf) + gd.forest[i] * wf); })(), f = Math.max(0, Math.min(1, (h - 1800) / 2600)); c = [gr[0] + (g[0] - gr[0]) * f, gr[1] + (g[1] - gr[1]) * f, gr[2] + (g[2] - gr[2]) * f];
        if (S < 64) {                                                                                       // ecology steered by the ground: exposed rock on steep faces, wetlands in flat wet bottoms, dune fields in sandy deserts
          const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];
          const rk = sm01((EF.sl - 0.38) / 0.3); if (rk > 0) c = mix(c, [g[0] * 0.9 + 14, g[1] * 0.9 + 12, g[2] * 0.9 + 10], rk * 0.88);
          const mar = sm01((EF.wet - 0.5) / 0.25) * sm01((0.14 - EF.sl) / 0.1) * sm01((Re - 0.3) / 0.2) * (h < 700 && T > 2 ? 1 : 0); if (mar > 0) c = mix(c, gd.marsh, mar * 0.85);
          if (S >= 5 && Re < 0.24 && T > 6 && EF.sl < 0.18) { const dn = 0.5 + 0.5 * Math.sin((wx * 1.7 - wy * 0.8) * 2.1 + vn(wx / 2.2, wy / 2.2) * 8), dm = sm01((0.24 - Re) / 0.12); c = mix(c, gd.sand, dm * 0.55); shx *= 1 + (dn - 0.5) * 0.26 * dm; }
          shx *= 1 - 0.1 * EF.cv + Math.max(0, Math.min(1, h / 4000)) * 0.1;
        } if (S >= 64 && OPT.snow && OPT.snow(wx, wy)) { c = [234, 241, 247]; } else if (S >= 64 && OPT.sand && OPT.sand(wx, wy) && !(OPT.clear && OPT.clear(wx, wy)) && (R < 0.3 || T > 14)) { c = [gd.sand[0], gd.sand[1], gd.sand[2]]; } { const sa = (S >= 64 && OPT.sand ? 0 : (R < 0.25 && T > 10 ? 1 : 0)) * Math.max(0, 1 - cvr.g * 4) * (OPT.clear && S >= 64 && OPT.clear(wx, wy) ? 0 : 1); if (sa > 0) c = [c[0] * (1 - sa) + gd.sand[0] * sa, c[1] * (1 - sa) + gd.sand[1] * sa, c[2] * (1 - sa) + gd.sand[2] * sa]; } if (fine) { const n = vn(wx / 0.006, wy / 0.006) - 0.5, m2 = vn(wx / 0.0015, wy / 0.0015) - 0.5, t = 1 + n * 0.14 + m2 * 0.1; c = [c[0] * t, c[1] * t, c[2] * t]; } }
      if (S >= 64 && h > 0 && OPT.clear && OPT.clear(wx, wy)) { const sg = ground(R, T, h, M.G[k]).soil; c = [sg[0] * 0.9, sg[1] * 0.9, sg[2] * 0.9]; }
      if (h > 0) {
        let a, lkv, river;
        { const cx = wx - 0.5, cy = wy - 0.5, ix = Math.max(0, Math.min(MW - 2, Math.floor(cx))), iy = Math.max(0, Math.min(MH - 2, Math.floor(cy))), fx = Math.max(0, Math.min(1, cx - ix)), fy = Math.max(0, Math.min(1, cy - iy)), i0 = iy * MW + ix, A0 = M.area[i0], A1 = M.area[i0 + 1], A2 = M.area[i0 + MW], A3 = M.area[i0 + MW + 1]; a = (A0 * (1 - fx) + A1 * fx) * (1 - fy) + (A2 * (1 - fx) + A3 * fx) * fy; lkv = sample(M, 'lake', wx * CELL, wy * CELL); river = a > 60; if (river) { river = false; const mx2 = Math.floor(wx), my2 = Math.floor(wy), qx = wx + (vn(wx / 0.3 + 5, wy / 0.3) - 0.5) * 0.12 + (vn(wx / 0.07, wy / 0.07 + 9) - 0.5) * 0.02, qy = wy + (vn(wx / 0.3, wy / 0.3 + 5) - 0.5) * 0.12 + (vn(wx / 0.07 + 9, wy / 0.07) - 0.5) * 0.02; for (let dy = -1; dy <= 1 && !river; dy++) for (let dx = -1; dx <= 1; dx++) { const ci2 = (my2 + dy) * MW + mx2 + dx; if (mx2 + dx < 0 || my2 + dy < 0 || mx2 + dx >= MW || my2 + dy >= MH || M.area[ci2] <= 60) continue; const rc = M.recv[ci2]; if (rc < 0) continue; const ax = (ci2 % MW) + 0.5, ay = Math.floor(ci2 / MW) + 0.5, bx = (rc % MW) + 0.5, by = Math.floor(rc / MW) + 0.5, ex = bx - ax, ey = by - ay, tt = Math.max(0, Math.min(1, ((qx - ax) * ex + (qy - ay) * ey) / (ex * ex + ey * ey || 1))), dd = Math.hypot(qx - ax - ex * tt, qy - ay - ey * tt), hw = Math.max(4 + 9 * Math.log10(M.area[ci2] / 60 + 1), 0.45 * mpp) / (CELL * 2); if (dd < hw) { river = true; a = M.area[ci2]; break; } } } }
        if (lkv > 0.5) { c = [60, 120, 190]; if (OPT.mask) OPT.mask[y * W + x] = 1; } else if (river) { const w = S >= 64 ? 1 : Math.min(1, Math.log(a / 60) / 3); c = [c[0] * (1 - w) + 50 * w, c[1] * (1 - w) + 110 * w, c[2] * (1 - w) + 190 * w]; if (OPT.mask) OPT.mask[y * W + x] = 2; }
      }
      const lv = Math.floor(h / ci); if (h > 0 && !edge && S < 64 && (lv !== Math.floor(hv(x + 1, y) / ci) || lv !== Math.floor(hv(x, y + 1) / ci))) c = [c[0] * 0.72, c[1] * 0.72, c[2] * 0.72];
      const o = (y * W + x) * 4; px[o] = c[0] * sh * shx; px[o + 1] = c[1] * sh * shx; px[o + 2] = c[2] * sh * shx; px[o + 3] = 255;
    }
    py = end; return py >= Hh;
    };
    const res = { w: W, h: Hh, data: px };
    return { res, step, rows: Hh, get at() { return py; } };
  }
  function paintRect(M, x0, y0, wc, hc, S, ci, opts) { const j = paintRectInc(M, x0, y0, wc, hc, S, ci, opts); if (j.rows > 0) j.step(j.rows); else j.step(0); return j.res; }
  function paint(M, S) { return paintRect(M, 0, 0, M.MW, M.MH, S, 250); }
  root.AgonMacro = { ground, cover, snowOf, COV: { soil: COV_SOIL, grass: COV_GRASS, forest: COV_FOREST, sand: COV_SAND }, paint, paintRect, paintRectInc, surface, roughOf, HURST, GEO, CELL, MW, MH, HREAL, genMacro, sample, psample, hash, mulberry };
  if (typeof module !== 'undefined') module.exports = root.AgonMacro;
})(typeof globalThis !== 'undefined' ? globalThis : this);
