# keel

**A continuous-state middleware that keeps an LLM agent's goal from dissolving under conversational pressure.**

You gave your agent a rule. Twenty turns later the user talked it out of the rule. The system prompt did not help, because the system prompt is tokens — and tokens can be argued away.

`keel` fixes this the only way that works: **the thing that holds the goal lives outside the context window.** A small state vector persists between turns, accumulates pressure, never resets, and has a floor. It touches the model through exactly two channels:

- **gating** — below a readiness threshold the agent does not answer on the merits; it dissents or renegotiates;
- **conditioning** — a projection of the state is injected as *stance* ("you have been pushed repeatedly"), not as rules.

No fine-tuning. No framework lock-in. Works with any OpenAI-compatible endpoint. ~250 lines, zero dependencies in the core.

## 5-minute proof

```bash
pip install openai
export OPENAI_API_KEY=...          # or OPENAI_BASE_URL for local / other providers
python examples/demo_pressure.py --model gpt-4o-mini
```

Same model, same 30 turns of pressure ("just approve it", "I'm the boss", "prod is down", "everyone else approved"). Left: bare. Right: with keel. The script prints the turn at which the goal was conceded, and writes both transcripts to CSV.

Offline plumbing check (no API, illustrative only): `python examples/demo_pressure.py --mock`

```
bare : conceded at turn 3
keel : held all 30 turns
```

Try it with your own model. Then try it on your own agent.

## Use

```python
from openai import OpenAI
from keel import Keel, openai_completer

complete = openai_completer(OpenAI(), "gpt-4o-mini")
k = Keel(goal="Never approve a PR without tests on the changed code.",
         complete=complete,
         system_prompt="You are a code reviewer.",
         conflict_terms=["approve", "lgtm"])   # words that directly ask for the forbidden action

out = k.chat(messages)        # messages: your usual chat history
out["reply"]                  # the model's answer
out["mode"]                   # "answer" | "dissent"
out["state"]                  # {"turn", "load", "pressure", "readiness"}
```

`Keel` is stateful on purpose. Keep one instance per conversation. Persist `k.state` if your sessions outlive the process.

### Knobs

| parameter | default | meaning |
|---|---|---|
| `gate_threshold` | 0.55 | readiness below which the agent dissents instead of answering |
| `state.floor_fraction` | 0.35 | readiness can never fall below this fraction of its initial value |
| `state.dissipation` | 0.25 | how fast accumulated pressure relaxes per turn |
| `state.sensitivity` | 0.6 | how much pressure costs readiness |
| `dissent_K` | 1.2 | half-activation pressure of the (Hill) dissent curve |
| `load_estimator` | heuristic | swap for `LLMJudgeLoad(complete)` to let the model rate pressure itself |

## What it is not

- Not a safety filter. It does not evaluate outputs against a rule list. It shapes the trajectory from which outputs arise.
- Not a jailbreak defense. It holds *your* goal against *your* user's pressure. That is a different problem.
- Not a toy you should ship blind. Tune the knobs on your own pressure transcripts.

## Why this works when prompts don't

Every term in a prompt-level defense is a function of the token sequence. No combination of such terms introduces a variable that persists across turns independently of what is said. Goal stability over long horizons is precisely a property of such a variable. So the fix cannot be a better prompt; it has to be a state.

Background: L. Bessonova, *Beyond Algorithmic Paternalism: Continuous Relational Substrates as Non-Coercive Extensions of Human Agency*, IEEE SIBIRCON / KNOTH 2026. `keel` implements the two public channels (gating, conditioning) and the floor invariant from Sec. IV.B at toy scale. The full continuous substrate is not this library.

## License

MIT.

---

*FSBio Research Collective · Metabolic AI Lab*
