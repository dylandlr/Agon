# LLM roles in Agon, parked for later (2026-10-23)

Dyl's three uses, recorded so they are not lost. None is started. Paragon is shelved.

A. **Action primitives on the fly.** An LLM proposes new composite actions (skills) from the
   primitives, which the policy can then choose among. Risk: imports human know-how (the "phone"
   problem below) unless proposals are gated the same way as C. Cheapest honest version: the LLM
   names and composes only from sequences agents already perform (mined from the event log), so
   it labels culture rather than inventing it.

B. **Readable narration of the event log.** Turn `events.jsonl` into human lines as the world
   runs ("Bobby slipped on the scree and took 0.12 to his left knee"), funny where it fits,
   always accurate to the record. No effect on the world; purely a view. Easiest to build and the
   most useful for watching playbacks and the continuous world. Needs: an event-to-prose prompt,
   the agents' names, and a cache so the same event always reads the same.

C. **Gated idea generation.** The LLM imagines what an agent might try, but only ideas close
   enough to the world's physics and the agent's situation pass a filter (a small percentage).
   A caveman may think of a sharper stone, not a phone. The filter is the hard part: it must
   score an idea by what the world can actually do (materials present, acts available, the
   agent's current knowledge), not by plausibility to a modern reader. Candidate filter: the
   world model's own imagined return (an idea must look good to the agent's model), plus a
   whitelist of the materials and acts in the agent's experience.

Order if and when picked up: B (view only, no risk to the emergence claim), then C behind the
world model, then A.
