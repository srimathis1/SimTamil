"""Gemini wrapper: key from env/.env, retries for rate limits, JSON helper, call-rate gate."""
import os, re, json, time, threading
from pathlib import Path
HERE = Path(__file__).parent

def _load_env():
    for p in (HERE / ".env", HERE.parent / ".env"):
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("\"'"))
_load_env()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
KEY = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
MIN_GAP = float(os.getenv("GEMINI_MIN_GAP", "1.0"))        # seconds between calls
_client, _lock, _last = None, threading.Lock(), [0.0]
_gate = threading.Semaphore(2)

MODELS = [MODEL, "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]
# remove duplicates while preserving order
MODELS = list(dict.fromkeys(m for m in MODELS if m))

class LLMError(Exception): pass

def available():
    return bool(KEY) and os.getenv("SIM_MODE", "real") != "offline"

def _cli():
    global _client
    if _client is None:
        from google import genai
        _client = genai.Client(api_key=KEY)
    return _client

def call(system, prompt, json_mode=False, temperature=0.85, retries=3, max_tokens=None):
    """Return model text. Tries candidate models and retries on rate limits."""
    if not available():
        raise LLMError("no GEMINI_API_KEY")
    from google.genai import types
    kw = dict(system_instruction=system, temperature=temperature)
    if json_mode: kw["response_mime_type"] = "application/json"
    if max_tokens: kw["max_output_tokens"] = max_tokens
    try: kw["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    except Exception: pass
    cfg = types.GenerateContentConfig(**kw)
    err = None

    for m in MODELS:
        for i in range(retries):
            with _gate:
                with _lock:
                    wait = MIN_GAP - (time.time() - _last[0])
                    if wait > 0: time.sleep(wait)
                    _last[0] = time.time()
                try:
                    r = _cli().models.generate_content(model=m, contents=prompt, config=cfg)
                    t = (r.text or "").strip()
                    if t: return t
                    err = "empty response"
                except Exception as e:
                    err = str(e)
                    # if exhausted or not found, break retry loop to try next model
                    if re.search(r"404|RESOURCE_EXHAUSTED", err, re.I):
                        break
            time.sleep(1 + i * 2)
    raise LLMError(err or "unknown error")

def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    m = re.search(r"(\[.*\]|\{.*\})", t, re.S)
    if m:
        cleaned = m.group(1)
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
        try:
            return json.loads(cleaned)
        except Exception:
            pass
    # Object-by-object regex extractor if global json parse fails
    items = []
    for obj_m in re.finditer(r'\{[^{}]*"id"\s*:\s*"([^"]+)"[^{}]*\}', t):
        try:
            items.append(json.loads(obj_m.group(0)))
        except Exception:
            id_m = re.search(r'"id"\s*:\s*"([^"]+)"', obj_m.group(0))
            text_m = re.search(r'"text"\s*:\s*"([^"]+)"', obj_m.group(0))
            action_m = re.search(r'"action"\s*:\s*"([^"]+)"', obj_m.group(0))
            mood_m = re.search(r'"mood"\s*:\s*"([^"]+)"', obj_m.group(0))
            if id_m and text_m:
                items.append({
                    "id": id_m.group(1),
                    "text": text_m.group(1),
                    "action": action_m.group(1) if action_m else "approach",
                    "mood": mood_m.group(1) if mood_m else ""
                })
    if items:
        return items
    raise LLMError("Failed to parse JSON response: " + t[:200])

