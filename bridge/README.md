# Eos Unity bridge

The Python process owns the simulation. The Unity client owns its display. The
current diagnostic scene is a transport/rendering scaffold, not the finished game.

From the repository root on Windows:

```powershell
uv venv --python 3.13 node_modules/.cache/eos-venv
uv pip install --python node_modules/.cache/eos-venv/Scripts/python.exe -r research/requirements-ci.txt
node_modules/.cache/eos-venv/Scripts/python.exe -m pytest bridge -q
unity test unity-game --editor-version 6000.6.5f1 --mode PlayMode --output unity-game/Logs/playmode.xml
```

Open `unity-game/Assets/Eos.unity` in the Unity Editor. It starts paused in
explicitly labelled random diagnostic mode. The UI Toolkit HUD provides run/pause,
single-step, 1/4/16-tick speed, save/load, camera pan/zoom and an agent inspector.
No Unity pawn jobs run in this scene. The HUD has been inspected in a live Editor;
Step and colonist selection have been exercised through UI Toolkit submit events.
Physical mouse/keyboard coverage and presentation polish remain pending.

The client defaults to the isolated Python path above and `bridge/eos_server.py`;
override these component fields for another environment. Packaged distribution
and a portable Python runtime remain pending.

For a trained run, set TrainedRun and disable DiagnosticRandom. Set EngineRoot
to an isolated matching historical source checkout when needed. The run needs
`history.json`, its population/weights or checkpoint, and a `bridge.json` with
`code_version`, **all** exact `env_config` fields, `hidden`, `encoder` and
`world_model`. The manifest must come from the matching training code; do not
fill missing settings with current defaults. Both history and manifest must
match the engine fingerprint before population loading is attempted. Imports
come from the selected engine before any population is loaded; no schema coercion
or fallback is used. Historical source and private weights belong in ignored local
storage, not the public repository.

With a trusted matching source checkout and the run's saved history, export the
configuration using that checkout's own training configuration statements:

```powershell
python bridge/export_run_config.py --engine-root PATH_TO_ENGINE --run PATH_TO_RUN
python bridge/verify_trained_run.py --engine-root PATH_TO_ENGINE --run PATH_TO_RUN
```

Export refuses to overwrite an existing manifest. Verification compares visible
state against independent direct policy inference and Eos stepping, and tests
trained recurrent-policy continuation after save/load. It is opt-in because the
private weights are not distributed with the repository.

Protocol: one JSON request/response per line, version 1 and an echoed id. Commands
currently supported: hello, reset, step, snapshot, save, load, close. Random mode is uniform
masked actions with silent messages, a diagnostic baseline only. Manual mode is
for parity tests. Snapshots identify the engine fingerprint, fixed-policy inference
and whether each population slot has ever been occupied. Empty slots are hidden.
Full event/body/fauna rendering, online learning and newborn brain inheritance
are still pending. Loading trained weights does not implement those mechanisms.

Verified on 2026-11-19: direct-state bridge parity and protocol checks; Unity
PlayMode spawned the actual Python process, received a 24x24 map and four agents,
created renderers, remained paused, and advanced exactly one Eos tick. This was
a headless test; subsequent live-Editor checks verified layout and advancement.

Save/load uses named slots in `node_modules/.cache/eos-saves` by default. Override
with `--save-dir` or the client's SaveDirectory field. Saves use atomic replacement
and array-only npz content (no pickle); they retain the full Eos state, PRNG key,
policy carry and parameter arrays. Loading requires a reset session with the same
engine/configuration/policy architecture. Invalid or incompatible loads leave the
current session intact. They restore inference state, not PPO optimizer/training
state; online learning is not implemented in this bridge yet.

Latest verification: six Python tests pass, including continuation equality after
save/load and bfloat16 policy-array storage. The extended actual Unity PlayMode
test also passes: HUD loads, client saves, steps, and restores an identical
snapshot. Test saves use an isolated directory, separate from gameplay saves.

Historical verification on 2026-11-19: a user-owned baseline run from engine
`f4498b87343e` loaded with its exact 445-field configuration and exported weights.
Independent inference matched the bridge's world fields and agent states for
three ticks; trained save/load continuation matched as well. Unity then loaded
that engine, displayed the learned-policy provenance and advanced via its Step
control with no console errors. This is evidence for short-run inference, not
long-term survival, online learning or a completed game.
