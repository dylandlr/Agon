"""The world can be lived in: a competent scripted band, using only the given skills and what each
agent has seen, keeps its numbers up and raises children who grow up able to have their own."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_a_competent_band_persists_and_its_children_reproduce():
    import argparse
    import viability as V
    cfg = V.build_cfg(argparse.Namespace(preset="society", world=None, food=1.0))
    r = V.run(cfg, seed=1, days=60, knowledge="own")
    assert not r["extinct"] and r["alive_end"] >= 10
    # daughters born here grew to a fertile weight and had children (impossible before child growth)
    assert r["births_to_mothers_born_here"] >= 2
    assert r["deaths"].get("heat", 0) == 0 and r["deaths"].get("thirst", 0) <= 2
