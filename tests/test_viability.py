"""The world can be lived in: a competent scripted band, using only the given skills and what each
agent has seen, keeps its numbers up and raises children who grow up able to have their own."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_a_competent_band_persists_and_its_children_reproduce():
    import argparse
    import viability as V
    # the society preset's world (250 x 250, living, ecology) at 30-minute ticks, so 60 days is 2,880
    # steps on CI's CPUs rather than 86,400 (the ecology is the same per km^2 at any tick length)
    cfg = V.build_cfg(argparse.Namespace(preset="society", world=None, food=1.0, tick_minutes=30.0))
    r = V.run(cfg, seed=1, days=60, knowledge="own")
    assert not r["extinct"] and r["alive_end"] >= 10
    # daughters born here grew to a fertile weight and had children (impossible before child growth)
    assert r["births_to_mothers_born_here"] >= 2
    # thirst per person who lived, not a count: more births put more pregnant, nursing and infant
    # lives at risk (seed 1: 2 of 24 before the fecundity change, 3 of 27 after; 15% is the bound)
    lived = cfg.n_founders + r["births"]
    assert r["deaths"].get("heat", 0) == 0 and r["deaths"].get("thirst", 0) <= 0.15 * lived
