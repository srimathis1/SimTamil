"""Append-only simulation log (data/logs/sim_log.jsonl) shown in the UI and downloadable."""
import json, threading, time
from pathlib import Path
F = Path(__file__).parent / "data" / "logs" / "sim_log.jsonl"; F.parent.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock(); _items = []
if F.exists():
    for ln in F.read_text(encoding="utf-8").splitlines():
        try: _items.append(json.loads(ln))
        except Exception: pass
def add(kind, text, agent=None, sim="", **extra):
    with _lock:
        e = {"seq": len(_items) + 1, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "sim": sim, "kind": kind, "agent": agent, "text": text, **extra}
        _items.append(e)
        with F.open("a", encoding="utf-8") as f: f.write(json.dumps(e, ensure_ascii=False) + "\n")
        return e
def since(seq=0, limit=200):
    with _lock: return [e for e in _items if e["seq"] > seq][-limit:]
