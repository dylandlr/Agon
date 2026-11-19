# System validation

Each check shows a behavior is possible with generic actions and pays off; none is trained or rewarded directly.

**9 of 9 checks pass.**

| Check | Result | Evidence |
| --- | --- | --- |
| Group hunting pays | Pass | a large animal falls in 5 ticks to two hunters vs 13 to one; it leaves 5.8 units of meat |
| Fire can be made with generic actions | Pass | place wood on rock, take stone in hand, strike: fire after 8 strikes |
| Heat cooks food, and cooked food is safe | Pass | meat dropped by the fire became 0.88 units cooked in 20 ticks; eaten with no infection, though raw meat here infects every time |
| Fire warms | Pass | 29.1 C beside the fire vs 14.1 C without |
| Huddling keeps warm | Pass | body temperature after 100 ticks at 0 C: -0.67 together vs -0.71 alone. Small, but here it is the difference between staying above the cold-damage threshold (-0.7) and falling below it |
| Standing together deters predators | Pass | attacks on an agent beside a predator over 20 ticks: 17 alone, 0 with a companion |
| Watching transmits behavior | Pass | an observer that only watches (no reward) picks the demonstrated action with probability 0.08 before and 1.00 after, seeing only what it could see |
| Invented names and meaning would be detected | Pass | shared words for individuals: 3 names found (Mikaka, Takaka, Rukaka); random speech: 0; messages tied to a predator carry 0.92 bits vs 0.00 for noise |
| Reward pays for no social or technological behavior | Pass | reward terms: staying alive, death, pain, novelty, touch, moral sentiments (alive_reward, death_penalty, pain_aversion, novelty_reward, touch_reward, moral). Nothing rewards hunting, giving, fire, cooking, speech, huddling, deterrence or company in general; two inborn sentiments do reward answering a past act in kind (reciprocity: giving to someone who gave to you; vengefulness: striking someone who struck you) |
