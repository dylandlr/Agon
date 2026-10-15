"""Named run settings, so every machine (CPU, Colab, HPRC) runs the identical experiment,
and a fingerprint of the simulation code, so runs from different versions are never mixed."""
from __future__ import annotations

import hashlib
from pathlib import Path

# The current design: a reproducing society on a RimWorld-sized 250 x 250 world (at 1-minute ticks
# a cell is 67 m, so the map is 16.7 km across, 280 km^2: with the plant ecology, enough land to feed
# a band; the old 32 x 32 was 4.6 km^2, which starves 16 people on realistic plants), survival as the only reward,
# sleep, the cognitive map and convex needs (all on by default in the environment), and the
# "caveman" slate: given survival skills, with social life left to learning; in the "living"
# world (biomes, dynamic weather, a climate cycle); with a learned world model that imagines.
PRESETS = {
    "society": dict(agents=16, founders=8, reproduction=True, reward_mode="survival",
                    grid=250, steps=192, encoder="conv", slate="caveman", world="living", world_model=True,
                    episode_years=10, tick_minutes=1.0),
    # The same society at half-hour ticks on a 32 x 32 map (a cell is 2 km at walking pace, so the map
    # is 64 km across): 30x less compute per game day than "society", so a 1e8-step budget lives ~50
    # game years per world instead of ~2, enough for many generations. Coarser in time (a tick is half
    # an hour of doing one thing), and each cell's plants are those of a 4 km^2 patch.
    # 64 slots: a cap for training worlds that the world, not the cap, should bind. With 16 a scripted
    # band filled every slot and only died of age (one of three seeds aged out by day 87); with 32 it
    # still reached the cap at its peaks; with 64 it reached 64 too, so this is the highest that is
    # affordable rather than one never reached: the logs count conceptions the cap stopped
    # (conceptions_blocked_by_slot_cap; raise --agents if it is not ~0). Continuous worlds grow their
    # slots as needed (scripts/continuous_world.py, lifesim/grow.py), so they have no cap at all.
    "society30": dict(agents=64, founders=8, reproduction=True, reward_mode="survival",
                      grid=32, steps=128, encoder="conv", slate="caveman", world="living", world_model=True,
                      episode_years=10, tick_minutes=30.0),
    # A band large enough for culture to accumulate: 64 slots (24 founders) on a 300 x 300 world
    # (RimWorld's large map; 400 km^2),
    # so food and space, not the slot count, limit numbers. About 4.5x the compute of "society".
    "tribe": dict(agents=64, founders=24, reproduction=True, reward_mode="survival",
                  grid=300, steps=192, encoder="conv", slate="caveman", world="living", world_model=True,
                  episode_years=10, tick_minutes=1.0),
    # Real time: a 365-day year and real biological time (9-month pregnancies, adults at 15
    # calendar years, lives of decades). For realism checks of a single generation: no one born in a
    # run of this length grows up, so it cannot test persistence or culture across generations.
    "realtime": dict(agents=16, founders=8, reproduction=True, reward_mode="survival",
                     grid=250, steps=192, encoder="conv", slate="caveman", world="living", world_model=True,
                     episode_years=10, days_per_year=365, days_per_bio_year=365.0, tick_minutes=1.0),
}


def code_version() -> str:
    """Short hash of the simulation and learning code (lifesim/*.py, and the training and
    continuous-world scripts, which set the episode length and how lives continue). Changes
    whenever the world, the rewards or the learner change; identical on every machine."""
    h = hashlib.sha256()
    here = Path(__file__).parent
    scripts = [here.parent / "scripts" / f for f in ("train_phase2.py", "continuous_world.py")]
    for f in sorted(here.glob("*.py")) + [f for f in scripts if f.exists()]:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]
