"""Dual memory per agent, persisted to data/memory/<id>.json
  long_term  = OLD memory: seeded from the reflect system (seed=True) + consolidated summaries of past experience
  short_term = NEW memory: the latest things this agent said / heard / experienced (small, capped)
When short_term overflows, the oldest items are summarised into long_term."""
import json, threading, time
from pathlib import Path
import llm
DIR = Path(__file__).parent / "data" / "memory"; DIR.mkdir(parents=True, exist_ok=True)
SHORT_MAX, CONSOLIDATE_N, LONG_MAX, TEXT_MAX = 10, 5, 40, 160
_lock = threading.RLock(); _mem = {}

def _path(i): return DIR / f"{i}.json"
def get(a):
    with _lock:
        i = a["id"]
        if i in _mem: return _mem[i]
        m = None
        if _path(i).exists():
            try: m = json.loads(_path(i).read_text(encoding="utf-8"))
            except Exception: m = None
        if not m or m.get("digest") != a["digest"]:      # (re)seed old memory from reflect data
            p = a["persona"] or {}
            m = {"digest": a["digest"], "long_term": [{"text": t, "kind": "reflect", "seed": True, "sim": "—"} for t in p.get("core_memories_ta", [])], "short_term": []}
            _save(i, m)
        _mem[i] = m; return m

def _save(i, m): _path(i).write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
def _clip(t): t = " ".join(str(t).split()); return t if len(t) <= TEXT_MAX else t[:TEXT_MAX - 1] + "…"

def add(a, kind, text, sim=""):
    """kind: said | heard | event | chat | thought"""
    with _lock:
        m = get(a); m["short_term"].append({"text": _clip(text), "kind": kind, "sim": sim, "ts": time.strftime("%H:%M:%S")})
        if len(m["short_term"]) > SHORT_MAX: _consolidate(a, m, sim)
        _save(a["id"], m)

def _consolidate(a, m, sim):
    old, m["short_term"] = m["short_term"][:CONSOLIDATE_N], m["short_term"][CONSOLIDATE_N:]
    lines = "\n".join(f"- {x['text']}" for x in old); summ = None
    if llm.available():
        try: summ = llm.call("Summarise an agent's recent memories into ONE short first-person Tamil sentence (max 30 words). Output only the sentence.", lines, temperature=0.3)
        except Exception: pass
    m["long_term"].append({"text": _clip(summ or "; ".join(x["text"][:50] for x in old)), "kind": "summary", "seed": False, "sim": sim})
    seeds = [x for x in m["long_term"] if x.get("seed")]; rest = [x for x in m["long_term"] if not x.get("seed")]
    m["long_term"] = seeds + rest[-(LONG_MAX - len(seeds)):]

def context(a, n_long=8, n_short=8):
    m = get(a); seeds = [x for x in m["long_term"] if x.get("seed")]; rest = [x for x in m["long_term"] if not x.get("seed")]
    return {"long": [x["text"] for x in (seeds[:max(0, n_long - len(rest[-3:]))] + rest[-3:])], "short": [f"{x['sim']} {x['text']}".strip() for x in m["short_term"][-n_short:]]}
