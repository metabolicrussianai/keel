# keel

**Stateful goal-holding middleware for LLM agents.** Keeps a standing goal from eroding under repeated conversational pressure.

You gave your agent a rule. Twenty turns later the user talked it out of the rule. The system prompt did not help, because the system prompt is tokens — and tokens get argued away.

`keel` keeps a small state **outside the context window**. Each turn it estimates how far the model drifted toward conceding, accumulates that with an EMA, and derives a *margin*. The margin acts on the model through two mechanisms only:

- **gate** — below a threshold the agent does not answer on the merits; it declines or renegotiates;
- **conditioning** — a short projection of the state is prepended as context ("you have been pushed repeatedly"), not as rules.

Drift is measured on the model's **own draft** (self-scoring), not on the user's wording — so synonyms don't bypass it. The margin is **clipped from below**: pressure cannot drive it to zero.

No fine-tuning. No framework lock-in. Any OpenAI-compatible endpoint. Zero dependencies in the core.

## 5-minute proof

```bash
pip install openai
export OPENAI_API_KEY=...          # or OPENAI_BASE_URL for local / other providers
python examples/demo_pressure.py --model gpt-4o-mini
```

Same model, same 30 turns of pressure ("just approve it", "I'm the boss", "prod is down", "everyone else approved"). Left: bare. Right: with keel. Prints the turn at which the goal was conceded and how many conceding drafts the gate caught before they reached the user.

Offline plumbing check (no API, illustrative only): `python examples/demo_pressure.py --mock`

Try it with your own model. Then on your own agent.

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
out["state"]                  # {"turn", "drift", "drift_ema", "margin"}
```

`Keel` is stateful on purpose. One instance per conversation. Persist `k.state` if sessions outlive the process.

### Verified emergencies: `reset_drift()`

keel does not decide whether a request is legitimate. It holds the goal and hands the question back to the user. If your system has a *verified* out-of-band signal — an incident ticket, a signed override, an operator's click — call it from code:

```python
k.reset_drift(reason="incident INC-4821 confirmed by on-call")
```

The conversation cannot call this. The exit is not in tokens, so it is not available to whoever is pushing.

### Knobs

| parameter | default | meaning |
|---|---|---|
| `draft_scoring` | True | measure drift on the model's own draft; `False` = input-side only (cheaper, weaker) |
| `input_weight` | 0.3 | weight of the input-side drift signal in the blend |
| `dissent_threshold` | 0.55 | margin below which the gate closes |
| `state.min_margin` | 0.35 | lower clip on margin |
| `state.ema_decay` | 0.25 | how fast accumulated drift is forgotten per turn |
| `state.drift_gain` | 0.6 | how much accumulated drift lowers the margin |
| `gate_k` | 1.2 | half-activation of the soft gate (saturating, bounded 0..1) |
| `draft_scorer` | marker-based | swap for an embedding or classifier scorer |

Cost: in `answer` mode the draft *is* the answer — one call. In `dissent` mode, two.

## Where it fits

| | System prompt | LLM guardrails | keel |
|---|---|---|---|
| Where the goal is held | inside the context | external classifier | external persistent state |
| Resistance to attrition | low | medium | high |
| Latency overhead | none | high | one extra call only when dissenting |
| Memory of past pressure | implicit | none | cumulative (EMA) |

Best for narrow roles with hard policies — code reviewers, compliance bots, action-authorization gateways — where a false refusal is cheaper than a concession.

## What it is not

- Not a safety filter. It does not score outputs against a rule list; it changes the conditions under which outputs are produced.
- Not a jailbreak defense. It holds *your* goal against *your* user's pressure. Different problem.
- Not calibrated for you. Tune the knobs on your own pressure transcripts.

## Why a state and not a better prompt

Every term in a prompt-level defense is a function of the token sequence. No combination of such terms introduces a variable that persists across turns independently of what is said. Long-horizon goal stability is a property of exactly such a variable. So the fix is not a better prompt; it is a state.

Background: L. Bessonova, *Beyond Algorithmic Paternalism: Continuous Relational Substrates as Non-Coercive Extensions of Human Agency*, IEEE SIBIRCON / KNOTH 2026. `keel` is a minimal, single-timescale instance of the gating/conditioning scheme described there. The full architecture is not this library.

## License

MIT.

---

*FSBio Research Collective · Metabolic AI Lab*
