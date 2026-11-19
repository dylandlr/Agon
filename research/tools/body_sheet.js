#!/usr/bin/env node
/* Draw a sheet of people from the body model to a PNG (no dependencies): rows of characters, columns of
 * views and poses, each with the playback's ink outline, scaled up so pixels can be checked.
 *   node tools/body_sheet.js out.png [scale]
 */
"use strict";
const fs = require("fs"), zlib = require("zlib"), path = require("path");
const B = require(path.join(__dirname, "..", "scripts", "body_model.js"));

const INK = "#203036";
const hexRGB = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function sprite(m, opts) {                       // RGBA pixels of one figure with a one-pixel ink outline
  const r = B.render(m, opts), pal = B.palette(m, INK), W = r.width + 2, H = r.height + 2, px = new Uint8Array(W * H * 4);
  const on = (x, y) => x >= 0 && y >= 0 && x < r.width && y < r.height && r.slot[y * r.width + x] > 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const sx = x - 1, sy = y - 1, i = (y * W + x) * 4;
    let c = null;
    if (on(sx, sy)) c = pal[B.SLOTS[r.slot[sy * r.width + sx]]] || "#ff00ff";
    else if (on(sx - 1, sy) || on(sx + 1, sy) || on(sx, sy - 1) || on(sx, sy + 1)) c = INK;
    if (c) { const [R, G, Bb] = hexRGB(c); px.set([R, G, Bb, 255], i); }
  }
  return { W, H, px };
}

function png(W, H, rgba) {
  const crcT = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
  const crc = (b) => { let c = 0xffffffff; for (const v of b) c = crcT[(c ^ v) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
  const chunk = (t, d) => { const len = Buffer.alloc(4); len.writeUInt32BE(d.length); const td = Buffer.concat([Buffer.from(t), d]);
    const c = Buffer.alloc(4); c.writeUInt32BE(crc(td)); return Buffer.concat([len, td, c]); };
  const ihdr = Buffer.alloc(13); ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4); ihdr[8] = 8; ihdr[9] = 6;
  const raw = Buffer.alloc((W * 4 + 1) * H);
  for (let y = 0; y < H; y++) { raw[y * (W * 4 + 1)] = 0; Buffer.from(rgba.buffer, y * W * 4, W * 4).copy(raw, y * (W * 4 + 1) + 1); }
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", zlib.deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

const PEOPLE = [
  { sex: "male", age: 30, height: 178, weight: 74, clothing: 0.5, skin: "#c98d63", hair: "#2a1a10", beard: 1 },
  { sex: "female", age: 26, height: 164, weight: 58, clothing: 0.8, skin: "#8d5a3b", hair: "#1a1410", hairTexture: 2, hairLength: 6 },
  { sex: "male", age: 55, height: 170, weight: 92, clothing: 0.2, skin: "#e2b996", hair: "#6a5038", beard: 2, grey: 0.6, injury: 0.5 },
  { sex: "female", age: 29, height: 168, weight: 66, pregnant: true, clothing: 0.4, skin: "#b07850", hair: "#a06a30", hairLength: 14 },
  { sex: "male", age: 8, height: 128, weight: 26, clothing: 0.1, skin: "#d4a07a", hair: "#3a2a1a" },
  { sex: "female", age: 1, height: 75, weight: 9.5, clothing: 0, skin: "#c08860", hair: "#2a1a10", hairLength: 2 },
  { sex: "male", age: 22, height: 186, weight: 60, relWeight: 0.82, clothing: 0, skin: "#5e3a26", hair: "#101010", hairTexture: 2 },
];
const COLS = [{ view: "s" }, { view: "e" }, { view: "n" }, { view: "e", action: "walk", phase: 0 }, { view: "e", action: "walk", phase: 0.25 },
  { view: "e", action: "walk", phase: 0.5 }, { view: "s", action: "walk", phase: 0.25 }, { view: "e", action: "eat", phase: 0.25 },
  { view: "e", action: "drink" }, { view: "e", action: "reach" }, { view: "e", action: "touch" }, { view: "e", action: "strike", phase: 0.2 },
  { view: "e", action: "strike", phase: 0.6 }, { view: "s", action: "eat", phase: 0.25 }];

const out = process.argv[2] || "body_sheet.png", S = +(process.argv[3] || 4);
const cells = PEOPLE.map((m, i) => COLS.map((o) => sprite({ ...m, colour: ["#c04848", "#3a7ac0", "#c0a030", "#8a48c0", "#40a070", "#c07030", "#30a0b0"][i] }, { ...o, seed: i })));
const cw = Math.max(...cells.flat().map((c) => c.W)) + 2, ch = Math.max(...cells.flat().map((c) => c.H)) + 2;
const W = cw * COLS.length * S, H = ch * PEOPLE.length * S, img = new Uint8Array(W * H * 4);
for (let i = 0; i < W * H; i++) img.set([86, 110, 70, 255], i * 4);                 // grass
cells.forEach((row, r) => row.forEach((c, k) => {
  for (let y = 0; y < c.H; y++) for (let x = 0; x < c.W; x++) { const a = c.px[(y * c.W + x) * 4 + 3]; if (!a) continue;
    for (let dy = 0; dy < S; dy++) for (let dx = 0; dx < S; dx++) { const X = (k * cw + 1 + x) * S + dx, Y = (r * ch + 1 + y) * S + dy;
      img.set(c.px.subarray((y * c.W + x) * 4, (y * c.W + x) * 4 + 4), (Y * W + X) * 4); } }
}));
fs.writeFileSync(out, png(W, H, img));
console.log(`${out}: ${PEOPLE.length} people x ${COLS.length} views/poses`);
