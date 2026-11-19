"""Threat knobs and injected shocks."""
import pytest

from lifesim import threats
from lifesim.config import EnvConfig


def test_knobs_scale_the_pressures_they_name():
    base = EnvConfig()
    c = threats.with_threats(base, food=0.5, predators=2.0, disease=3.0, hazards=2.0)
    assert c.plant_abundance == 0.5 * base.plant_abundance
    assert c.predators_per_1000 == 2 * base.predators_per_1000 and c.n_predators >= base.n_predators
    assert c.contact_infection == 3 * base.contact_infection and c.water_infection == 3 * base.water_infection
    assert c.lightning_prob == 2 * base.lightning_prob and c.extreme_temp_shift == 2 * base.extreme_temp_shift
    harmless = threats.with_threats(base, predators=0.0)
    assert harmless.predator_attack_prob == 0.0 and harmless.predator_respawn == 0.0
    with pytest.raises(ValueError):
        threats.with_threats(base, food=-1)


def test_shocks_start_and_end_on_time_and_change_only_their_pressure():
    shocks = [threats.parse_shock("famine:40:10:0.2"), threats.parse_shock("cold:45:2:10")]
    assert threats.active(shocks, 39.9) == () and threats.active(shocks, 40) == (0,)
    assert threats.active(shocks, 46) == (0, 1) and threats.active(shocks, 50) == ()
    base = EnvConfig()
    c = threats.cfg_during(base, shocks, (0, 1))
    assert c.plant_abundance == pytest.approx(0.2 * base.plant_abundance)
    assert c.season_temp == tuple(t - 10 for t in base.season_temp)
    assert c.predator_attack_prob == base.predator_attack_prob
    bold = threats.cfg_during(base, [threats.parse_shock("predators:0:1:3")], (0,))
    assert bold.predator_attack_prob == pytest.approx(min(1.0, 3 * base.predator_attack_prob))
    assert bold.predator_rest_ticks < base.predator_rest_ticks
    with pytest.raises(ValueError):
        threats.parse_shock("locusts:1:1:1")
