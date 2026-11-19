# Agon unified game workspace

The developing game is `unity-game/`. Eos is the simulation authority; Unity
provides presentation and controls. The original colony and research versions
remain preserved. The finished game is not yet complete.

User decisions: retain the colony game's existing RimWorld-style HTML/SVG art;
provide **Overseer** and **Director** modes; preserve learned behavior and research
mechanics. Windows desktop is the first working delivery target.

- [Migration plan and ordered milestones](docs/migration/PLAN.md)
- [Feature reconciliation matrix](docs/migration/FEATURE_MAP.md)
- [Architecture, modes and decisions](docs/migration/ARCHITECTURE.md)
- [Asset pipeline](docs/migration/ASSET_PIPELINE.md)
- [Asset gallery](asset-library/index.html) and [machine-readable catalog](asset-library/catalog.json)
- [Verification and release gates](docs/migration/VALIDATION.md)
- [Current measured evidence](docs/UNIFIED-GAME.md)
- [Running the current bridge](bridge/README.md)

Quick repository checks:

```sh
npm run check:repository
npm run assets:check
npm run verify
```

Do not use the historical root README as the current status report. It is retained
verbatim as part of the colony source. `RECONCILIATION.md` records the initial source
merge; the migration documents supersede its earlier runtime status.

Private policies, training histories, engine checkouts and local test results live
in ignored storage. A compatible trained baseline was verified locally; weights
are not part of this repository. Diagnostic random mode remains explicitly named.
