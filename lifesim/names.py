"""Names exist only once agents invent them.

There are no names by default: agents are #0, #1, ... A name is created by
`names_over_time` only when the concept of naming is at play in the agents'
own speech, that is, when a message has become a shared, specific label for
one individual (at least two speakers use it, mostly around that individual;
see analysis.naming_index). The name is the agents' own message, read aloud
through a fixed syllable table so researchers can say it.
"""
from __future__ import annotations

SYLLABLES = ["", "ka", "lo", "mi", "ta", "ru", "se", "no", "va", "pi", "de", "zu", "ha", "yo", "be", "ni",
             "go", "fe", "ri", "mu", "sa", "te", "wo", "la", "ki", "po", "du", "ne", "si", "bo", "ja", "re"]


def spoken_form(message) -> str:
    """Read a message aloud: one syllable per non-silent symbol, e.g. 3-1-1 -> 'Mikaka'."""
    word = "".join(SYLLABLES[int(s)] for s in message if int(s) > 0)
    return word.capitalize() if word else ""


def names_over_time(records: list[dict], check_every: int = 48, **naming_kwargs) -> list[dict]:
    """Scan speech in time order; the first time a message qualifies as a name
    for an individual, that individual is named. Returns one entry per named
    individual: who, the message, how it is said, when, and by whom."""
    from .analysis import naming_index
    if not records:
        return []
    named, out = set(), []
    t_end = max(r["t"] for r in records)
    for t in range(check_every, t_end + check_every + 1, check_every):
        so_far = [r for r in records if r["t"] < t]
        for nm in naming_index(so_far, shuffles=10, **naming_kwargs)["names"]:
            if nm["target"] in named:
                continue
            named.add(nm["target"])
            out.append({"agent": nm["target"], "message": nm["message"], "name": spoken_form(nm["message"]),
                        "named_by": nm["speakers"], "t": min(t - 1, t_end), "specificity": nm["specificity"]})
    return out
