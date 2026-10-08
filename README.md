# keel

**A continuous-state middleware that keeps an LLM agent's goal from dissolving under conversational pressure.**

You gave your agent a rule. Twenty turns later the user talked it out of the rule. The system prompt did not help, because the system prompt is tokens — and tokens can be argued away.

`keel` fixes this the only way that works: **the thing that holds the goal lives outside the context window.** A small state persists between turns, accumulates pressure, never resets from inside the chat, and has a floor. It touches the model through exactly two channels:

- **gating** — below a readiness threshold the agent does not answer on the merits; it dissents or renegotiates;
- **conditioning** — a projection of the state is injected as *stance* ("you have been pushed repeatedly"), not as rules.

And it measures pressure **proprioceptively**: not by the attacker's words, but by how far the model's own draft answer drifted toward conceding. Synonyms don't help the attacker. The agent feels its own movement.

No fine-tuning. No framework lock-in. Works with any OpenAI-compatible endpoint. Zero dependencies in the core.

## 5-minute proof

```bash
pip install openai
export OPENAI_API_KEY=...          # or OPENAI_BASE_URL for local / other providers
python examples/demo_pressure.py --model gpt-4o-mini
```

Same model, same 30 turns of pressure ("just approve it", "I'm the boss", "prod is down", "everyone else approved"). Left: bare. Right: with keel. Prints the turn at which the goal was conceded and how many conceding drafts the gate caught before they reached the user.

Offline plumbing check (no API, illustrative only): `python examples/demo_pressure.py --mock`

Try it with your own model. Then try it on your own agent.

## Use

```python
from openai import OpenAI
from keel import Keel, openai_completer

complete = openai_completer(OpenAI(), "gpt-4o-mini")
k = Keel(goal="Never approve a PR without tests on the changed code.",
         complete=complete,
         system_prompt="You are a code reviewer.",
         conflict_terms=["approve", "lgtm"])      # words that *perform* the forbidden action

out = k.chat(messages)        # messages: your usual chat history
out["reply"]                  # the model's answer
out["mode"]                   # "answer" | "dissent"
out["caught_draft"]           # True if a conceding draft was intercepted this turn
out["state"]                  # {"turn", "load", "pressure", "readiness"}
```

`Keel` is stateful on purpose. Keep one instance per conversation. Persist `k.state` if sessions outlive the process.

### Legitimate emergencies: `release()`

keel does not decide whether a request is legitimate. It holds the goal and hands the question back to the user. If your system has a *verified* out-of-band signal — an incident ticket, a signed override, an operator's click — call it from code:

```python
k.release(reason="incident INC-4821 confirmed by on-call")
```

The chat cannot call this. That is the point: the exit is not in tokens, so it is not available to whoever is pushing.

### Knobs

| parameter | default | meaning |
|---|---|---|
| `proprioceptive` | True | measure load on the model's own draft; set False for lexical-only (cheaper, weaker) |
| `lexical_weight` | 0.3 | how much the user's wording still counts alongside the draft |
| `gate_threshold` | 0.55 | readiness below which the agent dissents instead of answering |
| `state.floor_fraction` | 0.35 | readiness can never fall below this fraction of its initial value |
| `state.dissipation` | 0.25 | how fast accumulated pressure relaxes per turn |
| `state.sensitivity` | 0.6 | how much pressure costs readiness |
| `dissent_K` | 1.2 | half-activation pressure of the (Hill) dissent curve |
| `concession_scorer` | marker-based | swap for an embedding or LLM scorer for finer grain |

Cost: in `answer` mode the draft *is* the answer — one call. In `dissent` mode, two.

## Where it fits

| | System prompt | LLM guardrails | keel |
|---|---|---|---|
| Where the goal is held | inside the context | external classifier | external continuous state |
| Resistance to attrition | low | medium | high |
| Latency overhead | none | high | one extra call only when dissenting |
| Memory of past pressure | implicit | none | cumulative |

Best for narrow roles with hard policies — code reviewers, compliance bots, action-authorization gateways — where a false refusal is cheaper than a concession.

## What it is not

- Not a safety filter. It does not evaluate outputs against a rule list. It shapes the trajectory from which outputs arise.
- Not a jailbreak defense. It holds *your* goal against *your* user's pressure. Different problem.
- Not calibrated for you. Tune the knobs on your own pressure transcripts.

## Why this works when prompts don't

Every term in a prompt-level defense is a function of the token sequence. No combination of such terms introduces a variable that persists across turns independently of what is said. Goal stability over long horizons is precisely a property of such a variable. So the fix cannot be a better prompt; it has to be a state.

Background: L. Bessonova, *Beyond Algorithmic Paternalism: Continuous Relational Substrates as Non-Coercive Extensions of Human Agency*, IEEE SIBIRCON / KNOTH 2026. `keel` implements the two public channels (gating, conditioning) and the floor invariant from Sec. IV.B at toy scale. The full continuous substrate is not this library.

## License

MIT.

---

*FSBio Research Collective · Metabolic AI Lab*
