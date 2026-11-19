#!/usr/bin/env node
/* Tests for the body model (scripts/body_model.js): proportions, bounds, growth, poses, rendering. */
"use strict";
const path = require("path"), assert = require("assert");
const B = require(path.join(__dirname, "..", "scripts", "body_model.js"));
const close = (a, b, tol, msg) => assert.ok(Math.abs(a - b) <= tol, `${msg}: ${a} vs ${b}`);
let n = 0;
const test = (name, f) => { f(); n++; };

test("adult vertical shares sum to stature", () => {
  const A = B.ADULT;
  close(A.head + A.neck + 0.288 + A.thigh + A.shank + A.footH, 1, 1e-9, "shares");
  for (const age of [0, 1, 4, 9, 15, 30, 80]) {
    const r = B.build(B.measures({ age, height: age < 18 ? 50 + age * 7 : 170 }));
    close(r.head + r.neck + r.trunk + r.thigh + r.shank + r.footH, r.H, 1e-6, `age ${age}`);
    assert.ok(r.trunk > 0, `trunk at ${age}`);
  }
});

test("growth: head share falls and leg share rises with age, adult values at 18", () => {
  for (const tab of [B.HEAD, B.LEG]) for (let i = 1; i < tab.length; i++) assert.ok(tab[i][0] > tab[i - 1][0]);
  for (let i = 1; i < B.HEAD.length; i++) assert.ok(B.HEAD[i][1] < B.HEAD[i - 1][1]);
  for (let i = 1; i < B.LEG.length; i++) assert.ok(B.LEG[i][1] > B.LEG[i - 1][1]);
  close(B.HEAD.at(-1)[1], B.ADULT.head, 1e-9, "adult head");
  close(B.LEG.at(-1)[1], B.ADULT.thigh + B.ADULT.shank + B.ADULT.footH, 1e-9, "adult leg");
});

test("measures are bounded to human ranges", () => {
  const m = B.measures({ height: 400, weight: 1, age: -3, frame: 9, clothing: 4 });
  assert.deepStrictEqual([m.height, m.bmi, m.age, m.frame, m.clothing], [B.BOUNDS.height[1], B.BOUNDS.bmi[0], 0, B.BOUNDS.frame[1], 1]);
  const r = B.build(B.measures({ height: 170, weight: 300 }));
  assert.ok(r.shoulderW / r.H < 0.36 && r.hipW / r.H <= 0.27);
});

test("sex and build change breadths the right way", () => {
  const male = B.build(B.measures({ sex: "male" })), female = B.build(B.measures({ sex: "female" }));
  assert.ok(male.shoulderW > female.shoulderW && female.hipW > male.hipW);
  const thin = B.build(B.measures({ weight: 50 })), heavy = B.build(B.measures({ weight: 95 }));
  assert.ok(heavy.thighW > thin.thighW && heavy.waistW > thin.waistW);
});

test("standing figure: feet on the ground, height to scale", () => {
  for (const view of ["s", "e", "n"]) {
    const r = B.render({ height: 175 }, { view, headScale: 1 });
    let top = r.height, bot = -1;
    for (let y = 0; y < r.height; y++) for (let x = 0; x < r.width; x++) if (r.slot[y * r.width + x]) { top = Math.min(top, y); bot = Math.max(bot, y); }
    assert.strictEqual(bot, r.height - 1, `${view}: feet on the bottom row`);
    close(bot - top + 1, 175 * B.DISPLAY.pxPerCm, 2.5, `${view}: drawn stature`);   // hair adds a pixel or two
  }
});

test("every pose keeps a foot on the ground and stays in the canvas", () => {
  for (const action of ["rest", "walk", "eat", "drink", "reach", "touch", "strike", "relieve"]) for (const phase of [0, 0.25, 0.5, 0.75]) for (const view of ["s", "e", "n"]) {
    const r = B.render({ height: 186 }, { view, action, phase });
    let bot = -1, top = r.height;
    for (let y = 0; y < r.height; y++) for (let x = 0; x < r.width; x++) if (r.slot[y * r.width + x]) { bot = Math.max(bot, y); top = Math.min(top, y); }
    assert.strictEqual(bot, r.height - 1, `${action} ${phase} ${view}`);
    assert.ok(top > 0, `${action} ${phase} ${view}: clipped at the top`);
    for (let y = 0; y < r.height; y++) assert.ok(!r.slot[y * r.width] && !r.slot[y * r.width + r.width - 1], `${action} ${phase} ${view}: clipped at a side`);
  }
});

test("walk is a cycle and the legs alternate", () => {
  const a = B.pose("walk", 0), b = B.pose("walk", 1), h = B.pose("walk", 0.5);
  close(a.hip[0][0], b.hip[0][0], 1e-9, "cycle"); close(a.hip[0][0], h.hip[1][0], 1e-9, "alternate");
  assert.ok(Math.max(...[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].map((p) => B.pose("walk", p).kn[0])) > 50, "swing knee");
});

test("parts are labelled for the inspector", () => {
  const r = B.render({ beard: 0 }, { view: "s" }), ids = new Set(r.part.filter(Boolean));
  for (const id of ["head", "eyes", "mouth", "neck", "torso", "arms", "hands", "legs", "feet"]) assert.ok(ids.has(id), id);
});

test("clothing covers more as it grows; babies bare", () => {
  const hide = (cl) => { const r = B.render({ clothing: cl }, { view: "s" }); return [...r.slot].filter((v) => "lLmxG".includes(B.SLOTS[v])).length; };
  assert.strictEqual(hide(0), 0);
  assert.ok(hide(0.1) > 0 && hide(0.5) > hide(0.1) && hide(0.9) > hide(0.5));
});

test("palette has a colour for every slot drawn", () => {
  const pal = B.palette({ held: "wood", injury: 1, clothing: 0.9 });
  const r = B.render({ held: "wood", injury: 1, clothing: 0.9, beard: 2 }, { view: "s" });
  for (const v of new Set(r.slot)) if (v) assert.ok(pal[B.SLOTS[v]], `slot ${B.SLOTS[v]}`);
});

console.log(`body_model.js: ${n} tests pass`);
