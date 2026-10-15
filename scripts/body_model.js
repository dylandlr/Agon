/* Agon body model: one scientific, composable figure for every person, from which the map sprite, the
 * inspector's figure and the character creator are all drawn, so they always agree.
 *
 *   1. measures  what the simulation knows about a body (height, weight, build, strength, age, sex,
 *                pregnancy, injury, clothing, skin and hair, ...)
 *   2. build()   a rig: segment lengths and widths in cm from anthropometric shares of stature
 *                (adult shares after Drillis & Contini, as tabulated in Winter, "Biomechanics and Motor
 *                Control of Human Movement"), with the head and leg shares following growth from birth
 *                to adulthood; breadths from frame, body mass index and sex; all bounded to human ranges
 *   3. pose()    joint angles (trunk lean, shoulders, elbows, hips, knees) for an action and a phase
 *                (walk, eat, drink, reach, touch, strike, rest), so any action is a pose and any pose
 *                animates by its phase
 *   4. render()  the rig in a pose, projected to a view (front, side, back), laid out as body parts
 *                (capsules for limbs, a tapered torso, an ellipse head), clothed by coverage, rasterised
 *                into palette slots with a part id per pixel (the inspector uses the ids to name parts)
 *
 * The one stylisation is DISPLAY.headScale: the head is drawn larger (about its chin) so faces read at
 * map size, the same for everyone; with headScale = 1 the figure is true to its measures.
 */
(function (root) {
  "use strict";
  const lerp = (tab, v) => {
    if (v <= tab[0][0]) return tab[0][1];
    for (let i = 1; i < tab.length; i++) if (v <= tab[i][0]) {
      const [x0, y0] = tab[i - 1], [x1, y1] = tab[i]; return y0 + (y1 - y0) * (v - x0) / (x1 - x0);
    }
    return tab[tab.length - 1][1];
  };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // shares of stature (adult): vertex to chin, chin to shoulder line, shoulder line to hip joint, hip
  // joint to floor (thigh + shank + foot height), and the arm and breadth shares
  const ADULT = { head: 0.130, neck: 0.052, upperArm: 0.186, foreArm: 0.146, hand: 0.108,
                  thigh: 0.245, shank: 0.246, footH: 0.039, footL: 0.152, shoulderW: 0.259, hipW: 0.191 };
  // growth: head and leg (hip-joint height) shares of stature by age in years
  const HEAD = [[0, 0.25], [2, 0.2], [6, 0.167], [12, 0.143], [18, 0.13]];
  const LEG = [[0, 0.36], [2, 0.42], [6, 0.48], [12, 0.52], [18, 0.53]];
  const ARM = [[0, 0.38], [6, 0.42], [18, 0.44]];          // shoulder to fingertip
  const BOUNDS = { height: [45, 215], bmi: [13, 40], frame: [0.8, 1.25] };
  const DISPLAY = { pxPerCm: 34 / 175, headScale: 1.55, minLimbR: 1.0 };   // minLimbR: no limb under 2 px wide

  function measures(m) {                                  // fill and bound what is given
    const age = clamp(m.age ?? 25, 0, 100);
    const height = clamp(m.height ?? 170, ...BOUNDS.height);
    const bmi = clamp(m.weight ? m.weight / (height / 100) ** 2 : m.bmi ?? 22, ...BOUNDS.bmi);
    return { age, height, bmi, frame: clamp(m.frame ?? 1, ...BOUNDS.frame), strength: clamp(m.strength ?? 1, 0.4, 2),
             sex: m.sex === "female" ? "female" : "male", pregnant: !!m.pregnant, injury: clamp(m.injury ?? 0, 0, 1),
             scars: clamp(m.scars ?? 0, 0, 1), relWeight: clamp(m.relWeight ?? 1, 0.6, 1.4), clothing: clamp(m.clothing ?? 0, 0, 1),
             skin: m.skin || "#c98d63", hair: m.hair || "#3b2718", hairTexture: m.hairTexture ?? 0, hairLength: m.hairLength ?? 10,
             beard: m.beard ?? 0, grey: clamp(m.grey ?? 0, 0, 1), eyes: m.eyes || "#2a2622", colour: m.colour || "#c04848",
             tint: m.tint || "", held: m.held || "", hide: m.hide || "#8a6a4a" };
  }

  function build(m) {                                     // segment lengths and widths, cm
    const H = m.height, mature = clamp(m.age / 18, 0, 1), adult = m.age >= 13;
    const head = lerp(HEAD, m.age) * H, legs = lerp(LEG, m.age) * H, neck = ADULT.neck * (0.75 + 0.25 * mature) * H;
    const trunk = H - head - neck - legs;
    const legK = legs / ((ADULT.thigh + ADULT.shank + ADULT.footH) * H);
    const armK = lerp(ARM, m.age) / (ADULT.upperArm + ADULT.foreArm + ADULT.hand);
    const bulk = clamp(1 + 0.55 * (m.bmi / 22 - 1), 0.7, 1.6), sexW = m.sex === "male" && adult ? 1.06 : 0.95;
    const shoulderW = clamp(ADULT.shoulderW * Math.sqrt(m.frame) * sexW * (0.85 + 0.15 * mature) * (1 + 0.06 * (m.strength - 1)), 0.2, 0.31) * H;
    const hipW = clamp(ADULT.hipW * Math.sqrt(m.frame) * (m.sex === "female" && adult ? 1.12 : 0.97) * bulk ** 0.5, 0.15, 0.27) * H;
    const limb = (share) => share * H * bulk ** 0.6;
    return {
      H, head, headW: head * 0.8, neck, neckW: 0.06 * H * bulk ** 0.4, trunk,
      shoulderW: shoulderW * bulk ** 0.35, waistW: Math.min(shoulderW, hipW) * clamp(0.82 + 0.3 * (m.bmi / 22 - 1), 0.75, 1.15),
      hipW, belly: m.pregnant ? 0.06 * H : Math.max(0, (m.bmi - 26) * 0.004 * H),
      upperArm: ADULT.upperArm * armK * H, foreArm: ADULT.foreArm * armK * H, hand: ADULT.hand * armK * H * 0.55,
      thigh: ADULT.thigh * legK * H, shank: ADULT.shank * legK * H, footH: ADULT.footH * legK * H, footL: ADULT.footL * legK * H,
      upperArmW: limb(0.052) * (1 + 0.15 * (m.strength - 1)), foreArmW: limb(0.042), handW: 0.045 * H,
      thighW: limb(0.085), shankW: limb(0.058), footW: 0.05 * H,
    };
  }

  // poses: joint angles in degrees. Sagittal angles are positive forward (shoulder flexion raises the arm
  // forward, hip flexion lifts the thigh forward, knee and elbow flexion bend them); abduction moves a
  // limb out to the side. lean tilts the trunk forward about the hips. Shoulder angles are relative to
  // the trunk (they lean with it); hip angles are the thigh's to the vertical (the legs stand on the
  // ground whatever the trunk does). phase in [0, 1) animates; index 0 is the left limb, 1 the right.
  const NEUTRAL = { lean: 0, neck: 0, sh: [[0, 6], [0, 6]], el: [8, 8], hip: [[0, 2], [0, 2]], kn: [0, 0] };
  function pose(action, phase = 0) {
    const p = JSON.parse(JSON.stringify(NEUTRAL)), s = Math.sin(phase * 2 * Math.PI), c = Math.cos(phase * 2 * Math.PI);
    switch (action) {
      case "walk": {                                       // sagittal gait curves (after Winter's normal-walking
        // averages): hip flexion +30 at heel strike to -10 at toe-off; knee 15 in loading, 60 in swing
        const leg = (u) => { u = ((u % 1) + 1) % 1;
          return [10 + 20 * Math.cos(2 * Math.PI * u), 4 + 14 * Math.exp(-(((u - 0.15) / 0.08) ** 2)) + 56 * Math.exp(-(((u - 0.72) / 0.11) ** 2))]; };
        const [h0, k0] = leg(phase), [h1, k1] = leg(phase + 0.5);
        p.hip[0][0] = h0; p.kn[0] = k0; p.hip[1][0] = h1; p.kn[1] = k1;
        p.sh[0][0] = 0.6 * (h1 - 10); p.sh[1][0] = 0.6 * (h0 - 10); p.el = [12 + 0.3 * Math.max(0, h1 - 10), 12 + 0.3 * Math.max(0, h0 - 10)];
        p.lean = 3; break; }
      case "eat":                                          // hand to the mouth and down again
        p.sh[1] = [55 + 20 * Math.max(0, s), 12]; p.el[1] = 120 + 25 * Math.max(0, s); p.sh[0] = [10, 8]; p.el[0] = 30; break;
      case "drink":                                        // a crouch, both hands down to the water
        p.hip = [[95, 8], [95, 8]]; p.kn = [120, 120]; p.lean = 35 + 6 * Math.max(0, s); p.sh = [[55 + 6 * Math.max(0, s), 10], [55 + 6 * Math.max(0, s), 10]]; p.el = [10, 10]; break;
      case "reach":                                        // a stoop, a hand to the ground
        p.lean = 55 + 10 * Math.max(0, s); p.hip = [[20, 4], [20, 4]]; p.kn = [25, 25]; p.sh = [[45, 8], [70 + 10 * Math.max(0, s), 8]]; p.el = [25, 5]; break;
      case "touch":                                        // an arm out toward someone
        p.sh[1] = [70 + 15 * Math.max(0, s), 10]; p.el[1] = 10; break;
      case "strike":                                       // wind up and blow
        p.sh[1] = phase < 0.4 ? [150, 15] : [85, 5]; p.el[1] = phase < 0.4 ? 60 : 0; p.lean = phase < 0.4 ? -5 : 12; break;
      default:                                             // rest: breathing
        p.lean = 1 + c; break;
    }
    return p;
  }

  // project and lay out the parts; returns {parts, bounds}. Units: sprite pixels, y down, origin at
  // the ground under the body's centre. view: "s" front, "n" back, "e" profile facing right.
  function layout(rig, m, P, view, opts = {}) {
    const k = DISPLAY.pxPerCm * (opts.scale || 1), hs = opts.headScale ?? DISPLAY.headScale, side = view === "e";
    const rad = (d) => (d * Math.PI) / 180;
    const parts = [];
    // pelvis height from the stance (the lower foot rests on the ground)
    const legDrop = (i) => { const t = rad(P.hip[i][0]), sh = t - rad(P.kn[i]);
      return (rig.thigh * Math.cos(t) * Math.cos(rad(P.hip[i][1])) + rig.shank * Math.cos(sh) + rig.footH) * k; };
    const pelvisY = -Math.max(legDrop(0), legDrop(1));
    const L = rad(P.lean);
    const hipY = pelvisY, shY = hipY - rig.trunk * k * Math.cos(L), shX = (side ? rig.trunk * k * Math.sin(L) : 0);
    const torsoDepth = 0.62;                                // profile breadth relative to front breadth
    const W = (w) => (side ? w * torsoDepth : w) * k;
    // limbs: a chain of capsules from a joint, by absolute sagittal angle (and abduction in front view)
    const chain = (x0, y0, segs, lr) => {
      let x = x0, y = y0; const out = [];
      segs.forEach(([len, w, ang, abd, id]) => {
        const l = len * k;
        let dx, dy;
        if (side) { dx = l * Math.sin(ang); dy = l * Math.cos(ang); }
        else { const out_ = lr * (view === "n" ? -1 : 1); dx = out_ * l * Math.sin(abd); dy = l * Math.cos(abd) * Math.cos(ang); }
        out.push({ kind: "cap", id, x0: x, y0: y, x1: x + dx, y1: y + dy, r: Math.max(DISPLAY.minLimbR, (w * k) / 2), fwd: Math.sin(ang) });
        x += dx; y += dy;
      });
      return out;
    };
    // legs (index 0 left, 1 right; in front view left is on the viewer's right... we keep the
    // anatomical side and mirror by view)
    const chestW = W(rig.shoulderW) - (side ? 0 : 1.1 * Math.max(2 * DISPLAY.minLimbR, rig.upperArmW * k));
    const legs = [0, 1].map((i) => {
      const lr = i === 0 ? 1 : -1, hx = side ? 0 : lr * rig.hipW * k * 0.28 * (view === "n" ? -1 : 1);
      const th = rad(P.hip[i][0]), sh = th - rad(P.kn[i]), ab = rad(P.hip[i][1]);
      const segs = chain(hx, hipY, [[rig.thigh, rig.thighW, th, ab, "legs"], [rig.shank, rig.shankW, sh, ab * 0.5, "legs"]], lr);
      const end = segs[1];
      // the foot: a capsule whose underside is the sole, rig.footH below the ankle
      const fr = Math.max(0.6, rig.footH * k * 0.6), sole = end.y1 + rig.footH * k, fy = sole - fr;
      const foot = side
        ? { kind: "cap", id: "feet", x0: end.x1 - rig.footL * k * 0.25 + fr, y0: fy, x1: end.x1 + rig.footL * k * 0.75 - fr, y1: fy, r: fr }
        : { kind: "cap", id: "feet", x0: end.x1 - rig.footW * k * 0.5 + fr, y0: fy, x1: end.x1 + rig.footW * k * 0.5 - fr, y1: fy, r: fr };
      return { i, segs: [...segs, foot], depth: side ? (i === 1 ? 1 : -1) : Math.sin(th) };
    });
    // arms
    const arms = [0, 1].map((i) => {
      const lr = i === 0 ? 1 : -1, ar = Math.max(DISPLAY.minLimbR, rig.upperArmW * k / 2);
      const sx = side ? shX : lr * (chestW / 2 + ar * 0.55) * (view === "n" ? -1 : 1);
      const ua = rad(P.sh[i][0]) - L, fa = ua + rad(P.el[i]), ab = rad(P.sh[i][1]);
      const segs = chain(sx, shY + rig.upperArmW * k * 0.4, [[rig.upperArm, rig.upperArmW, ua, ab, "arms"], [rig.foreArm, rig.foreArmW, fa, ab, "arms"], [rig.hand, rig.handW, fa, ab, "hands"]], lr);
      return { i, segs, depth: side ? (i === 1 ? 2 : -2) : Math.sin(ua) * 0.5 };
    });
    // torso: a tapered body from shoulders to hips (leans in profile), chest above belly
    const torso = { kind: "torso", id: "torso", top: shY, bot: hipY, topX: shX, botX: 0, wTop: chestW, wWaist: W(rig.waistW), wHip: W(rig.hipW),
                    belly: rig.belly * k, side };
    const neckTopY = shY - rig.neck * k * Math.cos(L) * 0.9, neckX = shX + (side ? rig.neck * k * Math.sin(L) : 0);
    const neck = { kind: "cap", id: "neck", x0: shX, y0: shY + 0.5, x1: neckX, y1: neckTopY, r: Math.max(0.8, W(rig.neckW) / 2) };
    const hh = rig.head * k * hs, hw = rig.headW * k * hs * (side ? 1.08 : 1);
    const head = { kind: "head", id: "head", cx: neckX + (side ? hh * 0.08 : 0), cy: neckTopY - hh / 2 + 0.5, rx: hw / 2, ry: hh / 2 };   // drawn larger about the chin
    // draw order: far limbs, torso, neck, near limbs, head (front view: by forward-ness)
    const far = [], near = [];
    [...legs, ...arms].forEach((g) => (g.depth < 0 ? far : near).push(g));
    far.sort((a, b) => a.depth - b.depth); near.sort((a, b) => a.depth - b.depth);
    far.forEach((g) => parts.push(...g.segs));
    const legsFirst = near.filter((g) => g.segs[0].id === "legs");
    legsFirst.forEach((g) => parts.push(...g.segs));
    parts.push(torso, neck);
    near.filter((g) => g.segs[0].id !== "legs").forEach((g) => parts.push(...g.segs));
    parts.push(head);
    return { parts, head, torso, hipY, shY, legs, arms };
  }

  // rasterise: slots (skin s/S/d, hair, hide, ...) and a part id per pixel
  const SLOTS = ".sSdFhHkELlmxGBbIJRCK";
  function render(m0, opts = {}) {
    const m = measures(m0), rig = build(m), view = opts.view || "s", P = opts.pose || pose(opts.action || "rest", opts.phase || 0);
    const lay = layout(rig, m, P, view, opts);
    const sc = opts.scale || 1, W = opts.width || Math.round(40 * sc), Hh = opts.height || Math.round(50 * sc), ox = W / 2, oy = Hh;   // the ground is the canvas's bottom edge
    const slot = new Uint8Array(W * Hh), part = new Array(W * Hh).fill("");
    const put = (x, y, ch, id) => { x = Math.round(x); y = Math.round(y); if (x < 0 || y < 0 || x >= W || y >= Hh) return; slot[y * W + x] = SLOTS.indexOf(ch); if (id) part[y * W + x] = id; };
    const get = (x, y) => (x < 0 || y < 0 || x >= W || y >= Hh ? 0 : slot[y * W + x]);
    const pid = (x, y) => (x < 0 || y < 0 || x >= W || y >= Hh ? "" : part[y * W + x]);
    const dseg = (px, py, s) => { const vx = s.x1 - s.x0, vy = s.y1 - s.y0, L2 = vx * vx + vy * vy || 1;
      const t = Math.max(0, Math.min(1, ((px - s.x0) * vx + (py - s.y0) * vy) / L2)); return Math.hypot(px - (s.x0 + t * vx), py - (s.y0 + t * vy)); };
    let n = 0;
    const tag = (id) => `${id}#${n++}`;                    // each part instance its own id, for edges
    lay.parts.forEach((p) => {
      const id = tag(p.id);
      if (p.kind === "cap") {
        const x0 = Math.floor(Math.min(p.x0, p.x1) - p.r - 1 + ox), x1 = Math.ceil(Math.max(p.x0, p.x1) + p.r + 1 + ox);
        const y0 = Math.floor(Math.min(p.y0, p.y1) - p.r - 1 + oy), y1 = Math.ceil(Math.max(p.y0, p.y1) + p.r + 1 + oy);
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (dseg(x + 0.5 - ox, y + 0.5 - oy, p) <= p.r) put(x, y, "S", id);
      } else if (p.kind === "torso") {
        for (let y = Math.floor(p.top + oy); y <= Math.ceil(p.bot + oy); y++) {
          const t = clamp((y - oy - p.top) / Math.max(1, p.bot - p.top), 0, 1);
          let w = t < 0.5 ? p.wTop + (p.wWaist - p.wTop) * (t / 0.5) : p.wWaist + (p.wHip - p.wWaist) * ((t - 0.5) / 0.5);
          const cxT = p.topX + (p.botX - p.topX) * t + ox, bulge = p.belly * Math.max(0, Math.sin(Math.PI * clamp((t - 0.35) / 0.6, 0, 1)));
          const round = t < 0.08 ? 1.5 : 0;
          const xa = p.side ? cxT - w / 2 + round : cxT - w / 2 - bulge * 0.4 + round, xb = p.side ? cxT + w / 2 + bulge - round : cxT + w / 2 + bulge * 0.4 - round;
          for (let x = Math.floor(xa); x < Math.ceil(xb); x++) put(x, y, "S", id);
        }
      } else if (p.kind === "head") {
        for (let y = Math.floor(p.cy - p.ry - 2 + oy); y <= Math.ceil(p.cy + p.ry + 2 + oy); y++) for (let x = Math.floor(p.cx - p.rx - 2 + ox); x <= Math.ceil(p.cx + p.rx + 2 + ox); x++)
          if (((x + 0.5 - ox - p.cx) / p.rx) ** 2 + ((y + 0.5 - oy - p.cy) / p.ry) ** 2 <= 1) put(x, y, "S", id);
      }
    });
    // shading per part: light on the left edge of each part, dark on the right, a dark line where a
    // part lies over another (so crossing limbs read)
    const base = slot.slice();
    for (let y = 0; y < Hh; y++) for (let x = 0; x < W; x++) {
      if (!base[y * W + x]) continue;
      const me = pid(x, y), L_ = pid(x - 1, y), R_ = pid(x + 1, y);
      const LIMB = /^(arms|hands|legs|feet)#/;
      if (R_ !== me) put(x, y, R_ && R_.split("#")[0] !== me.split("#")[0] && (LIMB.test(R_) || LIMB.test(me)) ? "K" : "d");
      else if (L_ !== me) put(x, y, "s");
    }
    const kind = (x, y) => pid(x, y).split("#")[0];
    // clothing by coverage: loin wrap (hips), tunic (torso and hips, to mid-thigh), robe (all but hands,
    // head and the lower shins; foot wraps). Hide has a grain and a ragged hem.
    const cl = m.clothing, hipYpx = lay.hipY + oy;
    const covered = (x, y) => {
      const kd = kind(x, y); if (!kd || kd === "head" || kd === "hands") return false;
      if (cl >= 0.7) return kd === "feet" || kd === "neck" ? kd === "feet" : !(kd === "legs" && y > oy - rig.shank * DISPLAY.pxPerCm * sc * 0.25);
      if (cl >= 0.35) return kd === "torso" || (kd === "legs" && y < hipYpx + rig.thigh * DISPLAY.pxPerCm * sc * 0.45);
      if (cl > 0.05) return (kd === "torso" && y > hipYpx - 2.5) || (kd === "legs" && y < hipYpx + 2);
      return false;
    };
    const grain = (x, y) => ((x * 73 + y * 151 + (opts.seed || 0) * 17) % 11) < 2;
    for (let y = 0; y < Hh; y++) for (let x = 0; x < W; x++) {
      if (!base[y * W + x] || !covered(x, y)) continue;
      const cur = SLOTS[slot[y * W + x]], hem = !covered(x, y + 1) && kind(x, y + 1) !== "";
      if (hem && ((x * 7 + (opts.seed || 0)) % 3 === 0)) continue;                        // ragged
      put(x, y, cur === "K" ? "K" : kind(x, y) === "feet" ? "G" : cur === "s" ? "l" : cur === "d" || hem ? "m" : grain(x, y) ? "x" : "L");
    }
    if (cl >= 0.35 && view !== "e") {                                                  // a sinew cord at the waist
      const cy = Math.round(lay.shY + (lay.hipY - lay.shY) * 0.72 + oy);
      for (let x = 0; x < W; x++) if (kind(x, cy) === "torso" && "lLxm".includes(SLOTS[slot[cy * W + x]])) put(x, cy, x % 3 ? "B" : "b");
    }
    // condition: wounds on bare skin, ribs when thin
    const nW = Math.min(3, Math.round(m.injury * 4));
    const tTop = Math.round(lay.shY + oy), tMid = Math.round(ox + (lay.torso.topX || 0));
    for (let i = 0; i < nW; i++) { const x = tMid - 1 + (i % 2 ? 2 : -1), y = tTop + 3 + i * 2; if ("sSd".includes(SLOTS[get(x, y)])) put(x, y, "R"); }
    if (m.relWeight < 0.88 && view !== "n") for (let i = 0; i < 2; i++) { const y = tTop + 3 + i * 2; for (let x = tMid - 2; x <= tMid + 2; x++) if (SLOTS[get(x, y)] === "S") put(x, y, "d"); }
    // the head: shading, hair, face
    const h = lay.head, hx = h.cx + ox, hy = h.cy + oy, side = view === "e";
    const inHead = (x, y, gx = 0, gy = 0) => ((x + 0.5 - hx) / (h.rx + gx)) ** 2 + ((y + 0.5 - hy) / (h.ry + gy)) ** 2 <= 1;
    for (let y = Math.floor(hy - h.ry - 1); y <= Math.ceil(hy + h.ry); y++) for (let x = Math.floor(hx - h.rx - 1); x <= Math.ceil(hx + h.rx); x++) if (inHead(x, y)) {
      const t = 0.55 * (x + 0.5 - hx) / h.rx + 0.45 * (y + 0.5 - hy) / h.ry; put(x, y, t < -0.5 ? "s" : t > 0.45 ? "d" : "S", "head#h"); }
    if (side) put(hx + h.rx, hy + 1, "S", "head#h");                                       // the nose
    const curly = m.hairTexture >= 2, long = m.hairLength > 8 && !curly;
    for (let y = Math.floor(hy - h.ry - 3); y <= Math.ceil(hy + h.ry); y++) for (let x = Math.floor(hx - h.rx - 3); x <= Math.ceil(hx + h.rx + 1); x++) {
      const dx = (x + 0.5 - hx) / h.rx, dy = (y + 0.5 - hy) / h.ry, g = curly ? 1.2 + 0.5 * ((x + y) % 2) : 0.6;
      if (!inHead(x, y, g, g)) continue;
      let hair;
      if (view === "n") hair = dy < 0.55 || (long && Math.abs(dx) < 0.95);
      else if (side) hair = dy < -0.2 || (dx < -0.15 && dy < 0.5) || (long && dx < -0.35);
      else hair = dy < -0.25 + (x % 3 === 0 ? 0.12 : 0) || (long && Math.abs(dx) > 0.78 && dy < 0.85);
      if (!hair) continue;
      const t = 0.55 * dx + 0.45 * dy; put(x, y, curly && (x * 7 + y * 3) % 5 === 0 ? "k" : t < -0.5 ? "h" : t > 0.35 ? "k" : "H", "head#hair");
    }
    if (long && view === "n") for (let y = Math.round(hy + h.ry); y < Math.round(lay.shY + (lay.hipY - lay.shY) * 0.35 + oy); y++)
      for (let x = Math.round(hx - h.rx * 0.7); x <= Math.round(hx + h.rx * 0.6); x++) if (get(x, y)) put(x, y, x % 3 === 1 ? "k" : "H", "head#hair");
    const ey = Math.floor(hy + h.ry * 0.2), eyesOpen = !opts.eyesClosed, eyeCh = opts.dead ? "K" : "E";
    const eye = (x) => { if (eyesOpen) { put(x, ey, eyeCh, "eyes"); if (h.ry * 2 >= 9) put(x, ey + 1, eyeCh, "eyes"); } else put(x, ey + 1, "d", "eyes"); };
    if (view === "s") { const ex = Math.max(1.5, h.rx * 0.38); eye(hx - ex); eye(hx + ex - 1); }
    else if (side) eye(hx + h.rx * 0.5);
    if (view !== "n") for (let x = Math.round(hx - 1); x <= Math.round(hx + (side ? 2 : 1)); x++) { const my = Math.floor(hy + h.ry * 0.62); if (SLOTS[get(x, my)] === "S" || SLOTS[get(x, my)] === "d") part[my * W + x] = "mouth"; }
    if (m.beard && view !== "n") for (let y = Math.floor(hy); y <= Math.ceil(hy + h.ry + 3); y++) for (let x = Math.floor(hx - h.rx); x <= Math.ceil(hx + h.rx); x++) {
      const dx = (x + 0.5 - hx) / h.rx, dy = (y + 0.5 - hy) / h.ry, full = m.beard > 1;
      if (!inHead(x, y, 0, full ? 1.0 : 0.2) || dy < (full ? 0.5 : 0.66) || (side ? dx < 0 : Math.abs(dx) > (full ? 0.72 : 0.6))) continue;
      put(x, y, inHead(x, y + 1, 0, full ? 1.0 : 0.2) ? "H" : "k", "mouth");
    }
    // what is in hand: at the end of the right hand
    const rh = lay.arms.find((a) => a.i === 1).segs[2];
    if (m.held) { const x = Math.round(rh.x1 + ox), y = Math.round(rh.y1 + oy);
      const q = Math.max(1, Math.round(sc));
      if (m.held === "wood") { for (let i = 0; i < 7 * q; i++) for (let j = 0; j < q; j++) put(x + q + j, y - i, i >= 6 * q ? "J" : "I", "hands"); }
      else for (let i = 0; i < 2 * q; i++) for (let j = 0; j < 2 * q; j++) put(x + j, y + i, i < q ? "I" : "J", "hands"); }
    return { slot, part: part.map((p) => p.split("#")[0]), width: W, height: Hh, rig, m, layout: lay };
  }

  // colour ramps (skin shades along its own hue toward brown; hair and hide in their own hue)
  const hexToHsl = (hex) => { const n = parseInt(hex.slice(1), 16), r = (n >> 16) / 255, g = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), l = (mx + mn) / 2, d = mx - mn; let h = 0, s = 0;
    if (d) { s = d / (1 - Math.abs(2 * l - 1)); h = mx === r ? ((g - b) / d) % 6 : mx === g ? (b - r) / d + 2 : (r - g) / d + 4; h *= 60; if (h < 0) h += 360; } return [h, s, l]; };
  const hslToHex = (h, s, l) => { s = clamp(s, 0, 1); l = clamp(l, 0, 1); const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - c / 2;
    const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return "#" + [r, g, b].map((v) => Math.round((v + m) * 255).toString(16).padStart(2, "0")).join(""); };
  const mix = (a, b, t) => { const p = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)), A = p(a), B = p(b);
    return "#" + A.map((v, i) => Math.round(v + (B[i] - v) * t).toString(16).padStart(2, "0")).join(""); };
  const toHex = (c) => { if (c[0] === "#") return c; const mm = c.match(/hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%/); return mm ? hslToHex(+mm[1], +mm[2] / 100, +mm[3] / 100) : "#888888"; };
  const ramp = (hex, hueTo = null) => { const [h, s, l] = hexToHsl(hex), hh = hueTo === null ? h : h + (((hueTo - h + 540) % 360) - 180) * 0.3;
    return [hslToHex(hh, s * 0.9, l + 0.08), hex, hslToHex(hh, s * 0.85, l - 0.09), hslToHex(hh, s * 0.7, l - 0.2)]; };
  function palette(m0, ink) {
    const m = measures(m0), tintTo = { sick: "#a8b070", cold: "#8ea4cc", hot: "#e0705a", dead: "#9a9a9a" }[m.tint];
    const SK = ramp(tintTo ? mix(toHex(m.skin), tintTo, { sick: 0.2, cold: 0.15, hot: 0.18, dead: 0.7 }[m.tint]) : toHex(m.skin), 28);
    const dead = (c) => (m.tint === "dead" ? mix(c, "#9a9a9a", 0.6) : c);
    const HR = ramp(dead(mix(toHex(m.hair), "#cfcfcf", m.grey))), HD = ramp(dead(toHex(m.hide))), BD = ramp(dead(toHex(m.colour)));
    const ITEM = { wood: ["#8a5a36", "#5a3b26"], stone: ["#9a94a4", "#6c6678"], plant: ["#6fae50", "#3f7f3e"], raw_meat: ["#d0484a", "#8a2a2a"],
      cooked_meat: ["#b07040", "#6a4020"], honey: ["#f0b030", "#b07a16"], grubs: ["#f0e6c8", "#c8b890"] }[m.held] || ["#8aa0b8", "#5a6a80"];
    return { s: SK[0], S: SK[1], d: SK[2], F: SK[3], h: HR[0], H: HR[1], k: HR[3], E: "#1c2229", R: "#8a2a2a", C: mix(SK[0], "#ffffff", 0.35),
             l: HD[0], L: HD[1], m: HD[2], x: mix(HD[1], HD[2], 0.6), G: HD[3], B: BD[1], b: BD[0], I: ITEM[0], J: ITEM[1], K: mix(SK[3], ink || "#203036", 0.5) };
  }

  // the person's measures from a playback record (the simulation's own numbers) and its metadata
  const HIDES = ["#8a6a4a", "#7a5a3e", "#9a7a56", "#6e5440", "#a08868", "#5e4a3a"];
  function fromAgent(a, look, k) {
    return measures({ age: look.years, height: a.height, weight: a.weight, frame: look.frame ** 2, strength: a.strength ?? 1,
      sex: a.sex, pregnant: a.pregnant, injury: a.injury, scars: a.scars, relWeight: (a.weight || 60) / (a.ideal_kg || a.weight || 60),
      clothing: a.clothing || 0, skin: look.skin, hair: look.hair, hairTexture: look.tex, hairLength: look.hairLen, beard: look.beard,
      grey: clamp(((look.years || 0) - 40) / 37.5, 0, 0.8), colour: look.colour, held: look.held, hide: HIDES[k % HIDES.length],
      tint: !a.alive ? "dead" : a.sick ? "sick" : (a.temp ?? 0) < -0.5 ? "cold" : (a.temp ?? 0) > 0.6 ? "hot" : "" });
  }

  const api = { measures, build, pose, layout, render, palette, fromAgent, SLOTS, ADULT, HEAD, LEG, ARM, BOUNDS, DISPLAY, HIDES, toHex };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.AgonBody = api;
})(typeof window !== "undefined" ? window : globalThis);
