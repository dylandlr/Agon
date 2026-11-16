"""Cause and effect: the recorder links events by the simulation's mechanics, and the story script
tells the linked events as chains."""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lifesim import env as E
from lifesim.config import EnvConfig
from lifesim.recorder import _stamp_events, extract_events

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
CFG = EnvConfig(grid_size=12, n_agents=3, climate_volatility=0.0, lightning_prob=0.0)


def _base():
    s = E.reset(jax.random.PRNGKey(0), CFG)
    _, _, _, info = E.step(s, jnp.full(3, E.REST), CFG)
    z = lambda x: jnp.zeros_like(x)
    info = info._replace(outcome=jnp.full(3, E.OUT_REST), hits=z(info.hits), struck_by_agent=z(info.struck_by_agent),
                         died=z(info.died), newly_infected=z(info.newly_infected), recovered=z(info.recovered),
                         gifts=z(info.gifts), touches=z(info.touches), new_conditions=z(info.new_conditions),
                         born=z(info.born), gave_birth=z(info.gave_birth), conceived=z(info.conceived),
                         mated=z(info.mated), woken=z(info.woken), fell_asleep=z(info.fell_asleep),
                         collapsed_now=z(info.collapsed_now), fell=z(info.fell), kills=z(info.kills),
                         attacked_by_predator=z(info.attacked_by_predator), weather_changed=jnp.bool_(False),
                         lightning=jnp.bool_(False), ignited=z(info.ignited))
    return s, info


def _strike(info, t, a, b):
    return info._replace(t=jnp.int32(t), outcome=info.outcome.at[a].set(E.OUT_STRIKE_AGENT),
                         hits=info.hits.at[a, b].set(True), struck_by_agent=info.struck_by_agent.at[b].set(True))


def test_a_feud_and_a_death_are_linked_by_their_mechanics():
    s, info = _base()
    mem, evs = {}, []
    evs += _stamp_events(extract_events(_strike(info, 10, 0, 1), None, mem), CFG)
    evs += _stamp_events(extract_events(_strike(info, 20, 1, 0), None, mem), CFG)       # 1 strikes back
    dead = info._replace(t=jnp.int32(30), died=info.died.at[0].set(True),
                         death_cause=info.death_cause.at[0].set(E.CAUSE_VIOLENCE))
    evs += _stamp_events(extract_events(dead, None, mem), CFG)
    by = {e["id"]: e for e in evs}
    assert by["10:1:struck_by_agent"]["parents"] == ["10:0:strike_agent"]
    back = by["20:1:strike_agent"]
    assert back["parents"] == ["10:1:struck_by_agent"] and "striking back" in back["because"][0]
    assert by["30:0:death"]["parents"] == ["20:0:struck_by_agent"]                    # the last blow, not "last event"
    from causal_story import tell
    lines, summary, n = tell(evs)
    assert n == 1
    story = " ".join(lines)
    assert "#0 struck #1 -> (#0's blow)" in story and "(striking back at #0)" in story
    assert story.rstrip().endswith("#0 died of violence")
    assert any("violence: 1 (1 began with: an act of aggression)" in x for x in summary)


def test_illness_is_traced_to_the_sick_neighbour_and_orphans_to_their_mother():
    s, info = _base()
    mem = {}
    sick = info._replace(t=jnp.int32(5), newly_infected=info.newly_infected.at[1].set(True))
    extract_events(sick, None, mem, s, s, CFG)
    s1 = s._replace(infection=s.infection.at[1].set(50), pos=s.pos.at[0].set(s.pos[1]))   # 0 beside sick 1
    caught = info._replace(t=jnp.int32(6), newly_infected=info.newly_infected.at[0].set(True))
    ev = [e for e in extract_events(caught, None, mem, s1, s1, CFG) if e["type"] == "infected"][0]
    assert ev["parents"] == ["5:1:infected"] and "sick #1" in ev["because"][0]
    # a child born to 2; 2 dies; the child then starves: its hunger is traced to the mother's death
    kid = 1
    born = info._replace(t=jnp.int32(7), born=info.born.at[kid].set(True), gave_birth=info.gave_birth.at[2].set(True))
    s2 = s1._replace(mother=s1.mother.at[kid].set(2))
    extract_events(born, None, mem, s2, s2, CFG)
    mom_dies = info._replace(t=jnp.int32(8), died=info.died.at[2].set(True))
    extract_events(mom_dies, None, mem, s2, s2, CFG)
    hungry = s2._replace(alive=s2.alive.at[2].set(False), satiety=s2.satiety.at[kid].set(0.05))
    evs = extract_events(info._replace(t=jnp.int32(9)), None, mem, hungry, s2, CFG)
    starving = [e for e in evs if e["type"] == "starving" and e["agent"] == kid][0]
    assert "8:2:death" in starving["parents"] and any("mother" in w for w in starving["because"])
    assert starving["life"] == 2                                                    # slot 1's second life
