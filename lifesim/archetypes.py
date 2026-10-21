"""Archetypes: named temperament profiles placed deliberately among the founders.

An archetype sets heritable traits (feelings and capacities), never behaviour: a "tyrant" is
born with a tyrant's feelings, and whether it becomes one depends on what it learns and how
others respond. Placing them is an intervention, like a field experiment; the roles agents
actually fall into are found separately by clustering behaviour (scripts/archetypes.py).

    --archetypes tyrant:1,saint:1     -> founder slot 0 is a tyrant, slot 1 a saint

Traits not named keep their random draw. `python -c "from lifesim import archetypes as a;
print(a.table())"` lists them with what each is meant to test. Values stay inside each trait's normal range
(genotype._BOUNDS and the log-normal clip of 0.25-4).
"""
from __future__ import annotations

PROFILES = {
    # strong, vengeful, low empathy, low guilt
    "tyrant": dict(strength=1.6, vengefulness=1.0, empathy=0.0, guilt=0.0, envy=0.8, pain_aversion=0.6),
    # selfless, empathetic, forgiving
    "saint": dict(selflessness=1.0, empathy=1.0, vengefulness=0.0, reciprocity=1.0, envy=0.0, guilt=0.8),
    # low sociability, high pain aversion
    "outcast": dict(social_reward=0.3, pain_aversion=2.5, novelty=1.5),
    # curious, low guilt, high reciprocity
    "trickster": dict(novelty=2.5, guilt=0.0, reciprocity=1.0, vengefulness=0.2),
    # high kin care, empathy and resilience (a woman: sex is set at placement)
    "matriarch": dict(kin_care=1.0, empathy=0.9, pain_aversion=0.6, temperament_drift=0.1,
                      disease_resistance=1.5),
    # would-be leader: strong, very sociable, far-sighted, firm but not cruel
    "chief": dict(strength=1.4, social_reward=2.0, patience=0.998, empathy=0.6, vengefulness=0.6,
                  envy=0.4, pain_aversion=0.7),
    # learns fast, from others above all, stays curious, lives long: a store of know-how
    "sage": dict(lr_scale=1.8, social_learning=1.0, novelty=1.6, lifespan=1.3, patience=0.999,
                 empathy=0.7, critical_period=20000.0),
    # copies others, seeks company, avoids the new
    "conformist": dict(social_learning=1.0, social_reward=2.0, novelty=0.4, lr_scale=0.7),
    # learns on its own, from trying things: curious, plastic, ignores what others do
    "innovator": dict(novelty=3.0, social_learning=0.0, lr_scale=2.0, patience=0.998, social_reward=0.6),
    # big, fast, fearless, quick to answer a blow
    "warrior": dict(strength=1.8, speed=1.5, build=1.2, pain_aversion=0.4, vengefulness=0.7, empathy=0.3),
    # cares for the young and the hurt, of either sex
    "nurturer": dict(kin_care=1.0, empathy=1.0, social_reward=2.0, guilt=0.7, reciprocity=0.8),
    # envious, without guilt or gratitude, frugal body
    "hoarder": dict(envy=1.0, guilt=0.0, reciprocity=0.0, empathy=0.1, metabolism=0.7),
    # dreads pain, never retaliates, weak
    "coward": dict(pain_aversion=3.5, vengefulness=0.0, strength=0.7, novelty=0.5),
    # answers in kind, both ways, and feels it when unfair: a punisher of cheats
    "enforcer": dict(vengefulness=1.0, reciprocity=1.0, guilt=1.0, empathy=0.6, strength=1.3),
    # feels others' pain, never retaliates, returns kindness, seeks company
    "peacemaker": dict(empathy=1.0, vengefulness=0.0, reciprocity=0.8, guilt=0.6, social_reward=1.8),
    # strong drive for intimacy and company
    "charmer": dict(libido=3.0, social_reward=2.5, empathy=0.5, guilt=0.2),
    # restless explorer, fast and hardy, little need for company
    "wanderer": dict(novelty=3.5, speed=1.4, insulation=1.3, social_reward=0.5),
}
# What each archetype is meant to test, for the report
ABOUT = {
    "tyrant": "does a strong, remorseless founder come to dominate?",
    "saint": "does selflessness survive, and does it spread?",
    "outcast": "does a loner stay one, and how does the group treat it?",
    "trickster": "does curiosity without guilt pay?",
    "matriarch": "does a caring, resilient woman anchor a lineage?",
    "chief": "do others follow a sociable, far-sighted founder (leadership)?",
    "sage": "does a fast social learner become a source of know-how?",
    "conformist": "does copying spread whatever is already done?",
    "innovator": "does individual learning find what copying misses?",
    "warrior": "does physical dominance become rank?",
    "nurturer": "does care raise more children to adulthood?",
    "hoarder": "does envy without guilt pay, or isolate?",
    "coward": "is avoiding every fight a viable life?",
    "enforcer": "does a punisher of harm make others gentler (norms)?",
    "peacemaker": "does a non-retaliating carer lower violence?",
    "charmer": "does a strong drive for intimacy change who has children?",
    "wanderer": "does an explorer find new ground, and do others follow?",
}
# Traits the learner reads (learning rate, social learning, discount): they must be set on the
# trainer's population too, not only in the world. patience acts only with --trait-gamma.
LEARNER_TRAITS = {"lr_scale", "social_learning", "patience", "critical_period"}
FEMALE_ONLY = {"matriarch"}


def parse(spec) -> tuple:
    """'tyrant:1,saint:2' -> (('tyrant', 1), ('saint', 2)); '' or None -> ()."""
    if not spec:
        return ()
    out = []
    for part in str(spec).split(","):
        name, _, count = part.strip().partition(":")
        name = name.strip().lower()
        if name not in PROFILES:
            raise ValueError(f"unknown archetype {name!r}; known: {', '.join(sorted(PROFILES))}")
        out.append((name, int(count or 1)))
    return tuple(out)


def slots(archetypes) -> list:
    """Founder slot index -> archetype name, in the order given."""
    names = [name for name, count in archetypes for _ in range(count)]
    return names


def apply(genotype, sex, archetypes, n_founders):
    """Set the traits of the first founders to their archetypes. Returns (genotype, sex)."""
    import jax.numpy as jnp
    names = slots(archetypes)
    if len(names) > n_founders:
        raise ValueError(f"{len(names)} archetypes but only {n_founders} founders")
    for i, name in enumerate(names):
        genotype = genotype._replace(**{t: getattr(genotype, t).at[i].set(v)
                                        for t, v in PROFILES[name].items()})
        if name in FEMALE_ONLY:
            sex = sex.at[i].set(0)
    return genotype, sex


def table() -> str:
    lines = []
    for name, prof in PROFILES.items():
        traits = ", ".join(f"{k} {v:g}" for k, v in prof.items())
        lines.append(f"{name:<11} {ABOUT[name]}\n            {traits}")
    return "\n".join(lines)
