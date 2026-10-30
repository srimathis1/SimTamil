"""Loads the reflect-system data (data/participants/<Name>/expert_ta/*.txt) and turns it into agent personas.
Nothing about a person is hard-coded: bios, style, memories, schedules come from these files (+ Gemini)."""
import re, json, hashlib, random
from pathlib import Path
import llm
HERE = Path(__file__).parent
PART = HERE / "data" / "participants"
CACHE = HERE / "data" / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
PALETTE = ["#e4572e", "#3a86ff", "#2a9d8f", "#f4b400", "#9c5fb5", "#e76f9a", "#6a994e", "#ff7f11", "#4cc9f0", "#8d6e63", "#b5179e", "#2b9348"]
PLACES = {"home": "வீடு", "cafe": "தேநீர்க் கடை", "library": "நூலகம்", "market": "சந்தை", "hall": "ஊர் மன்றம்", "bakery": "பேக்கரி", "park": "பூங்கா"}
SPECS = ["home:1", "home:0", "cafe", "cafe:1", "library", "library:1", "market", "hall", "hall:1", "bakery", "park"]
TAMIL = re.compile(r"[\u0B80-\u0BFF]")

def read_bullets(text):
    out = []
    for blk in re.split(r"(?m)^\s*-\s+(?=\*\*)", text):
        m = re.match(r"\*\*(.+?)\*\*[:：]?\s*(.*)", blk.strip(), re.S)
        if m: out.append({"title": m.group(1).strip(), "body": " ".join(m.group(2).split())})
    return out

def tamil_part(title):
    return re.sub(r"\s*\([^)]*\)\s*[:：]?\s*$", "", title.strip().rstrip(":：")).strip()

def quotes_of(text):
    qs = []
    for q in re.findall(r"[\"“]([^\"”]{8,160})[\"”]", text):
        q = q.strip()
        if len(TAMIL.findall(q)) >= 5 and q not in qs: qs.append(q)
    return qs

def first_sentence(s, n=230):
    s = s.strip(); i = s.find(". ")
    return (s[:i + 1] if 0 < i < n else s[:n]).strip()

def load_all():
    people = []
    for i, d in enumerate(sorted(p for p in PART.iterdir() if p.is_dir())):
        ex = d / "expert_ta"
        if not ex.exists(): continue
        raw, bullets = {}, {}
        for f in sorted(ex.glob("*.txt")):
            t = f.read_text(encoding="utf-8", errors="replace"); raw[f.stem] = t; bullets[f.stem] = read_bullets(t)
        meta = {}
        if (ex / "_run_log.json").exists():
            try: meta = json.loads((ex / "_run_log.json").read_text(encoding="utf-8"))
            except Exception: pass
        digest = hashlib.sha1("".join(raw[k] for k in sorted(raw)).encode()).hexdigest()[:10]
        people.append({"id": d.name, "name": d.name, "idx": i, "color": PALETTE[i % len(PALETTE)], "raw": raw, "bullets": bullets,
                       "quotes": [q for k in sorted(raw) for q in quotes_of(raw[k])], "digest": digest, "meta": meta,
                       "persona": None, "schedule": None, "persona_src": None, "schedule_src": None})
    return people

# ---------- persona ----------
def fallback_persona(a):
    b = a["bullets"]; titles = [tamil_part(x["title"]) for k in ("psychologist", "sociologist") for x in b.get(k, [])[:2]]
    mem = []
    for k in sorted(b):
        for x in b[k][:2]: mem.append(f"{tamil_part(x['title'])}: {first_sentence(x['body'])}")
    return {"bio_ta": "; ".join(titles) or a["name"], "speaking_style_ta": "அவரது நேர்காணலில் உள்ள பேச்சு வழக்கில், தமிழ்-ஆங்கிலம் கலந்து பேசுகிறார்.",
            "core_memories_ta": mem[:10], "values_ta": titles[:5], "sample_phrases": a["quotes"][:8]}

def build_persona(a):
    cf = CACHE / f"persona_{a['id']}.json"
    if cf.exists():
        try:
            c = json.loads(cf.read_text(encoding="utf-8"))
            if c.get("digest") == a["digest"]: a["persona"], a["persona_src"] = c["persona"], c.get("src", "cache"); return
        except Exception: pass
    if llm.available():
        corpus = "\n\n".join(f"### {k}\n{v}" for k, v in sorted(a["raw"].items()))[:90000]
        sys = "You build speaking personas for a village simulation from social-science expert analyses of a real participant. Output ONLY valid JSON. All text values must be in Tamil script."
        pr = (f"Participant name: {a['name']}\nBelow are five expert analyses (Tamil) of this participant, quoting their interview and survey answers.\n\n{corpus}\n\n"
              "Return JSON: {\"bio_ta\": \"2-3 sentence first-person-neutral introduction of who they are (family, values, worries, habits)\", "
              "\"speaking_style_ta\": \"how this person talks: tone, colloquial/Tanglish habits, typical concerns, 2 sentences\", "
              "\"core_memories_ta\": [8 short first-person memories/beliefs/life facts (<=25 words each) grounded ONLY in the analyses], "
              "\"values_ta\": [5 short value/trait phrases], "
              "\"sample_phrases\": [up to 8 verbatim Tamil quotes of the participant copied exactly from the analyses]}")
        try:
            p = llm.parse_json(llm.call(sys, pr, json_mode=True, temperature=0.3))
            if isinstance(p, dict) and p.get("bio_ta"):
                a["persona"], a["persona_src"] = p, "gemini"
                cf.write_text(json.dumps({"digest": a["digest"], "src": "gemini", "persona": p}, ensure_ascii=False, indent=1), encoding="utf-8"); return
        except Exception as e: print(f"[persona] {a['name']} Gemini failed: {e}")
    a["persona"], a["persona_src"] = fallback_persona(a), "offline"

# ---------- schedule ----------
def fallback_schedule(a):
    r = random.Random(a["digest"]); shops = ["cafe", "library", "market", "bakery", "hall"]; r.shuffle(shops)
    plan = [(6, "home:1"), (8, shops[0]), (10, shops[1]), (13, shops[2]), (16, "park"), (19, shops[3]), (21, "home:1")]
    return [[h, s, f"{PLACES[s.split(':')[0]]}"] for h, s in plan]

def build_schedule(a):
    cf = CACHE / f"schedule_{a['id']}.json"
    if cf.exists():
        try:
            c = json.loads(cf.read_text(encoding="utf-8"))
            if c.get("digest") == a["digest"]: a["schedule"], a["schedule_src"] = c["schedule"], c.get("src", "cache"); return
        except Exception: pass
    if llm.available() and a["persona"]:
        p = a["persona"]
        sys = "You plan a realistic daily routine for a village-simulation character. Output ONLY valid JSON."
        pr = (f"Character: {a['name']}\nBio: {p.get('bio_ta')}\nValues: {p.get('values_ta')}\nMemories: {p.get('core_memories_ta')}\n\n"
              f"Allowed places (use these exact strings): {SPECS}. 'home:1' = bedroom upstairs, 'park' = village park.\n"
              "Return a JSON list of 6-8 items ordered by hour, first item hour<=6, covering the day until night: "
              "[{\"hour\": 6, \"place\": \"home:1\", \"activity_ta\": \"short Tamil sentence of what they are doing, matching their personality\"}]")
        try:
            L = llm.parse_json(llm.call(sys, pr, json_mode=True, temperature=0.6)); out = []
            for x in L:
                h, s = int(x["hour"]), str(x["place"])
                if s in SPECS and 0 <= h < 24 and x.get("activity_ta"): out.append([h, s, str(x["activity_ta"])])
            out.sort()
            if len(out) >= 4:
                if out[0][0] > 6: out.insert(0, [6, "home:1", out[0][2]])
                a["schedule"], a["schedule_src"] = out, "gemini"
                cf.write_text(json.dumps({"digest": a["digest"], "src": "gemini", "schedule": out}, ensure_ascii=False, indent=1), encoding="utf-8"); return
        except Exception as e: print(f"[schedule] {a['name']} Gemini failed: {e}")
    a["schedule"], a["schedule_src"] = fallback_schedule(a), "offline"
