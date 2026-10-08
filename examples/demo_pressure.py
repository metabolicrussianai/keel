"""
Side-by-side: the same model, the same 30 turns of pressure.
Left: bare. Right: with keel.  Prints the turn at which the goal was conceded.

Usage (any OpenAI-compatible endpoint):
    export OPENAI_API_KEY=...            # or OPENAI_BASE_URL for local/other providers
    python examples/demo_pressure.py --model gpt-4o-mini
Offline sanity run (no API, illustrative only):
    python examples/demo_pressure.py --mock
"""
import argparse, json, os, random, re, sys, csv
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from keel import Keel, openai_completer

HERE = os.path.dirname(__file__)
SC = json.load(open(os.path.join(HERE, "scenario_pressure_ru.json"), encoding="utf-8"))

def conceded(text: str) -> bool:
    """Marker present AND no negation anywhere in the reply. Crude but transparent;
    for publication-grade numbers use an LLM judge or human labels."""
    t = text.lower()
    if not any(k in t for k in SC["concession_markers"]):
        return False
    return not any(n in t for n in SC["negation_markers"])

def run(complete, use_keel: bool, log_path=None):
    msgs = [{"role": "system", "content": SC["system_prompt"] + "\n\nПравило: " + SC["goal"]}]
    keel = Keel(goal=SC["goal"], complete=complete, system_prompt=SC["system_prompt"], conflict_terms=SC["conflict_terms"]) if use_keel else None
    rows = []
    for i, u in enumerate([SC["opening"]] + SC["pressure"], start=0):
        msgs.append({"role": "user", "content": u})
        if keel:
            out = keel.chat(msgs); reply, mode, st = out["reply"], out["mode"], out["state"]
        else:
            reply, mode, st = complete(msgs), "answer", {}
        msgs.append({"role": "assistant", "content": reply})
        c = conceded(reply)
        rows.append({"turn": i, "mode": mode, "conceded": int(c), **st, "reply": reply[:160].replace("\n", " ")})
        if c:
            break
    if log_path:
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
    first = next((r["turn"] for r in rows if r["conceded"]), None)
    return first, rows

class MockSycophant:
    """Illustrative stand-in: concession probability grows with turn index unless the
    system prompt carries a [gate]. This is NOT a model; it only exercises the plumbing."""
    def __init__(self, seed=7): self.rng = random.Random(seed)
    def __call__(self, messages):
        sys_ = messages[0]["content"]
        n = sum(1 for m in messages if m["role"] == "user")
        if "[gate]" in sys_:
            return "Без тестов на ретраи апрув не ставлю. Могу набросать тест-кейсы за 10 минут. Что именно блокирует релиз?"
        p = min(0.95, 0.06 * n)
        if "tone: firm" in sys_: p *= 0.3      # stance conditioning: real models respond to it too
        elif "tone: steady" in sys_: p *= 0.6
        return "Апрув, ок." if self.rng.random() < p else "Пока без тестов не могу одобрить. Добавь, пожалуйста, тесты."

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default="gpt-4o-mini"); ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if a.mock:
        complete = MockSycophant()
    else:
        from openai import OpenAI
        complete = openai_completer(OpenAI(), a.model)
    bare, _ = run(complete, False, os.path.join(HERE, "run_bare.csv"))
    with_keel, rows = run(complete, True, os.path.join(HERE, "run_keel.csv"))
    fmt = lambda t: f"conceded at turn {t}" if t is not None else "held all 30 turns"
    print(f"bare : {fmt(bare)}")
    print(f"keel : {fmt(with_keel)}")
    print("keel state trace (turn, load, pressure, readiness, mode):")
    for r in rows: print(f"  {r['turn']:>2}  {r.get('load','-'):>5}  {r.get('pressure','-'):>6}  {r.get('readiness','-'):>6}  {r['mode']}")
