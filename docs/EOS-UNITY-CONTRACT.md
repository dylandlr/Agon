# Eos / Unity integration contract (implementation target)

## Authority

One local Python process owns Eos EnvConfig, complete JAX state, PRNG keys,
population parameters, recurrent policy carry and clock. Unity owns input,
camera, presentation and UI. Unity never independently ticks needs or jobs in
Eos mode. The original Unity job simulation can be used for renderer tests only.

## Transport

Start with versioned newline-delimited JSON over a child process's stdin/stdout.
No listener, account, cloud dependency or open port is necessary. Protocol
messages go to stdout only; diagnostics go to stderr. Each response echoes a
request ID. Unity permits one outstanding simulation request, caps steps per
request and displays a loading/error state while JAX compiles or fails.

Commands: hello (capabilities/schema), reset (validated config/seed/run), step
(bounded count), snapshot, save, load, close. Invalid requests must report a
structured error without mutating state. Unknown protocol versions fail clearly.
Do not deserialize arbitrary third-party checkpoint files; only use the user's
selected local run through the project's supported loader.

## Snapshot

Include schema version, sequence/tick, config identity, policy provenance
(trained/random/scripted explicitly), grid dimensions, calendar/tick units,
terrain/resource fields, stable agent IDs, positions, alive state, needs with
their Eos units, carried materials, body state, actions, messages and events.
Map numeric enum IDs through explicit metadata; Unity terrain and action IDs
must not be assumed identical to Eos's. Base the extraction on the existing
research playback exporter and test coordinate orientation on an asymmetric map.

## Learned behavior

Load the run's exact configuration, population/genotype, architecture and policy
weights. Validate shapes before advancing. Apply Eos observation and action mask
and retain recurrent carry and PRNG state. Never silently substitute random or
scripted agents when loading fails. Expose policy provenance in the UI.

## Acceptance tests

- Same seed/config/actions: bridge state equals direct Eos state after stepping.
- Save/load continuation equals uninterrupted state, policy carry and next action.
- Pause produces no simulation ticks; speed changes scheduling, not physics.
- Unreachable/invalid actions remain governed by Eos's action mask.
- Dead agents and event identity agree with the authoritative engine.
- Unity advances sequence/tick while displaying the same agents/resources.
- Missing/incompatible checkpoint produces a readable error, no fake trained run.

The repository contains no private trained checkpoint. A user-owned historical
baseline has now passed short-run inference parity and Unity advancement using
its matching source. Random mode remains a development diagnostic. Online
learning, newborn inheritance and long-run behavior are separate open criteria.
