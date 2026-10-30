"""Tamil autonomous-agent village server.
Agents are built from the reflect-system files in data/participants/*/expert_ta (nothing hard-coded per person).
Gemini (GEMINI_API_KEY) generates Tamil speech grounded in persona + dual memory. Without a key an offline mode
replays the participants' own interview quotes. Everything is logged to data/logs/sim_log.jsonl."""
import sys, random, re, threading
if hasattr(sys.stdout, "reconfigure"):
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
if hasattr(sys.stderr, "reconfigure"):
    try: sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request, send_from_directory, send_file
import llm, memory, reflect, simlog, deliberation
HERE = Path(__file__).parent; DIST = HERE / "frontend" / "dist"
app = Flask(__name__, static_folder=str(DIST), static_url_path="")
AGENTS = reflect.load_all(); BY = {a["id"]: a for a in AGENTS}
ACTIONS = {"gather": "go to the town hall to discuss with others", "approach": "walk to where it happened to look/join in", "flee": "run away from it to a safe place",
           "home": "go back home", "stay": "carry on with what you were doing"}
STATUS = {"phase": "idle", "done": 0, "total": len(AGENTS), "msg": ""}; _prep_lock = threading.Lock()

def prepare(force=False):
    if not _prep_lock.acquire(blocking=False): return
    try:
        STATUS.update(phase="building", done=0, msg="")
        for a in AGENTS:
            if force and a["persona_src"] == "offline": a["persona"] = a["schedule"] = None
            if not a["persona"]: reflect.build_persona(a)
            if not a["schedule"]: reflect.build_schedule(a)
            memory.get(a); STATUS["done"] += 1
        STATUS.update(phase="ready", msg=f"{sum(a['persona_src']=='gemini' for a in AGENTS)}/{len(AGENTS)} Gemini personas")
    finally: _prep_lock.release()

# ---------- prompt building ----------
RULES = ("You are role-playing ONE real person as a villager in a Tamil village simulation. Stay strictly in character using the persona, demographic facts, expert reflect files, and memories provided. "
         "ALWAYS answer in Tamil script (normal standard Tamil - செந்தமிழ்). Use clear, respectful, natural Tamil sentences grounded strictly in the reflect system data provided. "
         "CRITICAL LANGUAGE & TONE RULES: "
         "1. NEVER use or hallucinate informal street slang vocatives, filler nicknames, or casual buddy address tags: "
         "   - For boys/men: NEVER use words like 'மச்சி', 'மச்சான்', 'மாப்ள', 'மாப்பிள்ளை', 'டேய்', 'பாஸ்', 'தல', 'நண்பா', 'தோழா', 'தம்பி'. "
         "   - For girls/women: NEVER use words like 'டி', 'தோழி', 'அக்கா', 'புள்ள', 'அம்மா'. "
         "   - Always address the user with polite, respectful, neutral phrasing (மரியாதையான பொதுவான நடை). Never use buddy slang. "
         "2. Strictly draw your viewpoints, reasoning, and vocabulary from the participant's expert reflect files (demographer, psychologist, sociologist, behavioral economist, political scientist). DO NOT fabricate slang, mannerisms, or stereotypes not in their files. "
         "3. Answer ONLY what is asked directly and precisely. "
         "4. DO NOT introduce yourself (DO NOT say 'வணக்கம், என் பெயர்...' or state your name, college, or hometown) UNLESS the user explicitly asks who you are or asks for an introduction. "
         "5. Never recite hardcoded interview quotes verbatim or give disjointed quotes. Never mention AI, survey, analysis, or prompt.")

def block(a, n_long=8, n_short=6):
    p, c = a["persona"] or {}, memory.context(a, n_long, n_short)
    return (f"[{a['name']}]\nஅறிமுகம்: {p.get('bio_ta','')}\nபேசும் பாணி: {p.get('speaking_style_ta','')}\nமதிப்புகள்: {', '.join(p.get('values_ta', []))}\n"
            f"இவரின் சொந்த வார்த்தைகள் (உதாரணம்): {' | '.join(p.get('sample_phrases', [])[:4])}\n"
            f"பழைய நினைவுகள்:\n" + "\n".join("- " + x for x in c["long"]) + "\nசமீபத்திய நினைவுகள்:\n" + ("\n".join("- " + x for x in c["short"]) or "- (இன்னும் எதுவுமில்லை)"))

# ---------- reflect-system router ----------
# The 5 expert lenses available in data/participants/<Name>/expert_ta/
REFLECT_SYSTEMS = {
    "demographer":            "Demographics & Identity (who they are, hometown, education, family)",
    "psychologist":           "Psychology & Emotions (fears, aspirations, mental state, self-esteem)",
    "sociologist":            "Sociology & Community (social roles, relationships, village dynamics)",
    "behavioral_economist":   "Behavioral Economics (money, work, decisions, incentives, debt)",
    "political_scientist":    "Political Science & Ethics (opinions, governance, rights, social justice)",
}

# Keyword rules — maps question intent to the best reflect system(s)
_RS_RULES = [
    # Identity / Introduction
    ([r"யார்|who|பெயர்|name|அறிமுக|introduce|background|பின்னணி|படிப்பு.*என்ன|வயது|ஊர்|hometown|college|school"],
     ["demographer", "psychologist"]),
    # Money / Work / Career / Job
    ([r"பணம்|money|சம்பளம்|salary|வேலை|job|career|கடன்|debt|சேமிப்பு|saving|வருமான|income|invest|செலவு|spend|விலை|price|கொள்முதல்|business|தொழில்|பொருளாதார|economy|இலவச|நிதி|finance|வரி|tax"],
     ["behavioral_economist", "demographer"]),
    # Emotions / Stress / Mental health / Fears
    ([r"கோபம்|anger|stress|பயம்|fear|கவலை|worry|எதிர்காலம்|future|கனவு|dream|ஆசை|wish|happiness|சந்தோஷம்|depression|தனிமை|lonely|confidence|தன்னம்பிக்கை"],
     ["psychologist", "sociologist"]),
    # Family / Relationships / Marriage / Community
    ([r"குடும்பம்|family|தாய்|mother|தந்தை|father|திருமண|marriage|பெற்றோர்|parents|உறவு|relation|நண்பர்|friend|neighbor|அக்கா|அண்ண|சகோதர"],
     ["sociologist", "psychologist"]),
    # Politics / Rights / Government / Social justice / Caste
    ([r"அரசு|government|அரசியல்|politics|உரிமை|rights|சாதி|caste|ஒதுக்கீடு|reservation|சட்டம்|law|vote|ஓட்டு|நீதி|justice|policy|கொள்கை|protest|போராட்டம்|திட்டம்|scheme|பேருந்து|bus|போக்குவரத்து|transport|பெண்|women"],
     ["political_scientist", "sociologist"]),
    # Values / Priorities / Life philosophy / Ethics
    ([r"முன்னுரிமை|priority|முக்கியம்|important|தத்துவம்|philosophy|மதிப்பு|value|நம்பிக்கை|belief|கொள்கை|principle|வாழ்க்கை.*எது|எது.*முதல்"],
     ["psychologist", "behavioral_economist"]),
    # Social success / Meritocracy / Influence
    ([r"திறமை|talent|உழைப்பு|hardwork|முன்னேற|advance|வெற்றி|success|செல்வாக்கு|influence|வாய்ப்பு|opportunity|equality|சமத்துவம்"],
     ["behavioral_economist", "political_scientist"]),
    # Insult / Controversy / Defense
    ([r"கொக்கரொச்சை|cockroach|சோம்பேறி|lazy|useless|திருடன்|thief|முட்டாள்|fool|waste|திமிர்|மோசம்|பயனில்லை"],
     ["psychologist", "sociologist"]),
]

def pick_reflect_systems(msg: str) -> list:
    """Classify question → return ordered list of 1-2 most relevant reflect system keys."""
    for patterns, systems in _RS_RULES:
        for pat in patterns:
            if re.search(pat, msg, re.I):
                return systems
    # Default fallback: persona + psychologist for generic questions
    return ["psychologist", "demographer"]

def block_with_reflect(a, systems: list, n_long=6, n_short=4):
    """Build a prompt block enriched with bullets from the chosen reflect systems."""
    p, c = a["persona"] or {}, memory.context(a, n_long, n_short)
    base = (f"[{a['name']}]\nஅறிமுகம்: {p.get('bio_ta','')}\nபேசும் பாணி: {p.get('speaking_style_ta','')}\nமதிப்புகள்: {', '.join(p.get('values_ta', []))}\n"
            f"இவரின் சொந்த வார்த்தைகள் (உதாரணம்): {' | '.join(p.get('sample_phrases', [])[:4])}\n"
            f"பழைய நினைவுகள்:\n" + "\n".join("- " + x for x in c["long"]) + "\nசமீபத்திய நினைவுகள்:\n" + ("\n".join("- " + x for x in c["short"]) or "- (இன்னும் எதுவுமில்லை)"))
    # Append expert lens bullets for chosen systems
    expert_sections = []
    for sys_key in systems:
        bullets = a.get("bullets", {}).get(sys_key, [])
        if bullets:
            label = REFLECT_SYSTEMS.get(sys_key, sys_key)
            bullet_lines = "\n".join(f"  • {b['title']}: {b['body'][:130]}" for b in bullets[:5])
            expert_sections.append(f"\n[நிபுணர் பார்வை — {label}]\n{bullet_lines}")
    return base + "".join(expert_sections)

def clock(sim):
    try: m = int(float(sim)) % 1440; return f"{m//60:02d}:{m%60:02d}"
    except Exception: return ""

def tokens(s): return set(re.findall(r"[\u0B80-\u0BFFa-zA-Z]{3,}", s.lower()))
def quote_for(a, hint=""):
    qs = a["quotes"]
    if not qs: return a["name"] + "…"
    h = tokens(hint); sc = [(len(h & tokens(q)), random.random(), q) for q in qs]; sc.sort(reverse=True); return sc[0][2]

def remember(a, kind, text, sim=""): memory.add(a, kind, text, clock(sim))

_SLANG_WORDS = [
    "மச்சி", "மச்சான்", "மாப்ள", "மாப்பிள்ளை",
    "டேய்", "பாஸ்", "தல", "பங்காளி",
    "தோழா", "தோழி", "புள்ள", "டி"
]

def clean_slang(text: str) -> str:
    """Proactively filter and strip any hallucinated informal slang vocatives like மச்சி, மாப்ள, டேய்."""
    if not text: return ""
    for w in _SLANG_WORDS:
        # Match standalone slang word with optional leading comma/spaces and optional trailing punctuation
        pat = r"(?:,\s*|\s+)?(?<![\u0B80-\u0BFF])" + re.escape(w) + r"(?![\u0B80-\u0BFF])([.!?])?"
        text = re.sub(pat, lambda m: m.group(1) or "", text)
    text = re.sub(r"^[,.\s]+", "", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def trim(t, n=350):
    t = clean_slang(" ".join(str(t).split()).strip('"“” '))

    if len(t) <= n: return t
    truncated = t[:n]
    last_stop = max(truncated.rfind("."), truncated.rfind("?"), truncated.rfind("!"), truncated.rfind(";"))
    if last_stop > int(n * 0.5):
        return truncated[:last_stop + 1].strip()
    return truncated.rstrip() + "…"

# ---------- routes ----------
@app.get("/")
def index():
    if not (DIST / "index.html").exists(): return "Frontend not built. cd frontend && npm install && npm run build", 503
    return send_from_directory(app.static_folder, "index.html")

@app.get("/api/agents")
def agents():
    return jsonify({"mode": "gemini" if llm.available() else "offline", "model": llm.MODEL, "agents": [
        {"id": a["id"], "name": a["name"], "color": a["color"], "bio": (a["persona"] or {}).get("bio_ta", ""), "schedule": a["schedule"] or [],
         "persona_src": a["persona_src"], "idx": a["idx"]} for a in AGENTS]})

@app.get("/api/status")
def status(): return jsonify({**STATUS, "mode": "gemini" if llm.available() else "offline", "personas": {a["id"]: a["persona_src"] for a in AGENTS}})

@app.post("/api/prepare")
def prep():
    threading.Thread(target=prepare, kwargs={"force": True}, daemon=True).start(); return jsonify({"ok": True})

@app.get("/api/memory/<aid>")
def mem(aid):
    a = BY.get(aid)
    if not a: return jsonify({"error": "unknown"}), 404
    m = memory.get(a); return jsonify({"long": m["long_term"], "short": m["short_term"], "persona": a["persona"], "src": a["persona_src"]})

@app.get("/api/log")
def getlog(): return jsonify({"items": simlog.since(int(request.args.get("after", 0)), 300)})

@app.get("/api/log/download")
def dl(): return send_file(simlog.F, as_attachment=True, download_name="sim_log.jsonl") if simlog.F.exists() else ("", 204)

@app.post("/api/logevent")
def logevent():
    d = request.get_json() or {}; simlog.add(d.get("kind", "info"), d.get("text", ""), d.get("agent"), clock(d.get("sim"))); return jsonify({"ok": True})

def gen_say(a, situation, sim, fallback_hint=""):
    """One in-character Tamil utterance. Returns (text, source)."""
    try:
        t = llm.call(RULES, f"{block(a)}\n\nஇப்போதைய நேரம்: {clock(sim)}\nசூழ்நிலை: {situation}\n\nஇந்த நபராக ஒரே ஒரு தமிழ் வாக்கியம் (அல்லது இரண்டு) மட்டும் பேசு. மேற்கோள் குறிகள் வேண்டாம்.", temperature=0.95, max_tokens=200)
        return trim(t), "gemini"
    except Exception as e:
        print("[say] fallback:", e); return trim(quote_for(a, fallback_hint)), "offline"

@app.post("/api/say")        # autonomous muse / arrival remark
def say():
    d = request.get_json() or {}; a = BY.get(d.get("id"))
    if not a: return jsonify({"error": "unknown agent"}), 404
    act, place = d.get("act", ""), d.get("place", "")
    t, src = gen_say(a, f"நீ {place} இல் இருக்கிறாய். நீ செய்வது: {act}. அதைப்பற்றி மனதில் தோன்றுவதை சுயமாகப் பேசு.", d.get("sim"), act)
    remember(a, "said", f"நான் சொன்னேன்: {t}", d.get("sim")); simlog.add("say", t, a["name"], clock(d.get("sim")), src=src, place=place)
    return jsonify({"text": t, "src": src})

@app.post("/api/chat")       # the user talks to one agent
def chat():
    d = request.get_json() or {}; a = BY.get(d.get("id"))
    if not a: return jsonify({"error": "unknown agent"}), 404
    msg = (d.get("message") or "").strip()[:500]; sim = d.get("sim")
    is_intro = bool(re.search(r"who|யார்|tell.*about|introduce|அறிமுகம்|உன்னைப்பற்றி|உங்களைப்பற்றி|பெயர்|name|பின்னணி|background", msg, re.I))
    is_insult = bool(re.search(r"cockroach|கொக்கரொச்சை|சோம்பேறி|lazy|useless|முட்டாள்|waste|stupid|திருடன்|பொய்|திமிர்|மோசம்", msg, re.I))
    if is_intro:
        directive = f"வழிமுறை: இந்த நபராக ({a['name']}) இயல்பாக உரையாடு. உன்னைப்பற்றி கேட்டிருப்பதால் உன் பெயர், படிப்பு/பணி மற்றும் ஊரை சுருக்கமாக அறிமுகப்படுத்து."
    elif is_insult:
        directive = f"வழிமுறை: இந்த நபராக ({a['name']}) உன் இயல்பான சுபாவம், தனிப்பட்ட மனோநிலை மற்றும் பேசும் பாணிக்கு ஏற்ப இயல்பாகப் பேசு. ஒருவர் உன்னை நேருக்கு நேர் கடுமையான வார்த்தைகளால் திட்டுகிறார் / அவமதிக்கிறார் ('{msg}'). உன் சுயமரியாதையோடு உன் சுபாவத்திற்கு ஏற்ப (கோபமாகவோ, மனவருத்தத்தோடோ, அல்லது அமைதியாகவோ) தற்காத்துப் பேசு. தேவையில்லாமல் பழைய கதைகளையோ அறிமுகத்தையோ பேசாதே."
    else:
        directive = f"வழிமுறை: இந்த நபராக ({a['name']}) இயல்பாக உரையாடு. கேட்கப்பட்ட கேள்விக்கு மட்டுமே நேரடியாகவும் துல்லியமாகவும் பதில் கூறு. உன்னைப் பற்றியோ, உன் பெயரையோ அல்லது ஊரையோ தேவையின்றி அறிமுகப்படுத்தாதே (நேரடியாகக் கருத்தைத் தொடங்கு). பழைய தொடர்பற்ற மேற்கோள்களைப் பேசாதே."

    # ── Reflect-system routing ─────────────────────────────────────────────────────────────────────
    chosen_systems = pick_reflect_systems(msg)
    labels = [f"{s} ({REFLECT_SYSTEMS[s].split('(')[0].strip()})" for s in chosen_systems]
    sep = "─" * 70
    print(f"\n{sep}")
    print(f"  🔍 REFLECT SYSTEM ROUTER  │  Agent: {a['name']}")
    print(f"  📩 Question : {msg[:80]}{'...' if len(msg)>80 else ''}")
    print(f"  📚 Chosen   : {' + '.join(labels)}")
    for s in chosen_systems:
        desc = REFLECT_SYSTEMS.get(s, s)
        print(f"     ↳ [{s}] → {desc}")
    print(f"{sep}\n", flush=True)
    # ─────────────────────────────────────────────────────────────────────────

    pr = (f"{block_with_reflect(a, chosen_systems, 6, 4)}\n"
          f"இப்போதைய நேரம்: {clock(sim)}. நீ: {d.get('act','')}.\n"
          f"ஒருவர் உன்னிடம் நேரடியாகக் கூறுகிறார் / கேட்கிறார்: \"{msg}\"\n\n"
          f"{directive}")
    try:
        t = trim(llm.call(RULES, pr, temperature=0.3, max_tokens=250), 350); src = "gemini"
    except Exception as e:
        print("[chat] fallback:", e)
        p = a.get("persona") or {}
        bio = p.get("bio_ta", "")
        if re.search(r"who|யார்|tell|பற்றி|அறிமுகம்|name|பெயர்", msg, re.I) and bio:
            t = trim(bio.split(".")[0] + f" ({a['name']}).")
        elif is_insult:
            t = f"மரியாதையா பேசுங்க! என்னை பார்த்து என்ன வார்த்தை சொல்றீங்க? ({a['name']})"
        else:
            t = trim(quote_for(a, msg))
        src = "offline"
    remember(a, "chat", f"என்னிடம் பேசினார்: “{msg[:80]}” — நான்: {t}", sim)
    simlog.add("chat", f"பயனர்: {msg}", "You", clock(sim)); simlog.add("chat", t, a["name"], clock(sim), src=src)
    return jsonify({"reply": t, "src": src, "reflect_systems": chosen_systems})

@app.post("/api/converse")   # two agents meet
def converse():
    d = request.get_json() or {}; a, b = BY.get(d.get("a")), BY.get(d.get("b"))
    if not a or not b: return jsonify({"error": "unknown agent"}), 404
    place, sim = d.get("place", ""), d.get("sim"); lines, src = [], "offline"
    try:
        sys = RULES.replace("ONE real participant", "TWO real participants").replace("Speak in first person, 1-2 short sentences (max 25 words).", "Write a natural dialogue; each line max 22 words.")
        pr = (f"{block(a, 6, 5)}\n\n{block(b, 6, 5)}\n\nஇடம்: {place}, நேரம்: {clock(sim)}. {a['name']} மற்றும் {b['name']} சந்திக்கிறார்கள். "
              f"சமீபத்திய நினைவுகளில் ஏதாவது இருந்தால் அதைப்பற்றியும் பேசலாம். 4-6 வரிகள் உரையாடல். "
              f"JSON மட்டும்: [{{\"s\":\"{a['id']}\",\"t\":\"...\"}},{{\"s\":\"{b['id']}\",\"t\":\"...\"}}]")
        L = llm.parse_json(llm.call(sys, pr, json_mode=True, temperature=0.95, max_tokens=700))
        lines = [{"s": x["s"], "t": trim(x["t"], 200)} for x in L if x.get("s") in (a["id"], b["id"]) and x.get("t")][:6]; src = "gemini"
    except Exception as e: print("[converse] fallback:", e)
    if not lines:
        lines = [{"s": a["id"], "t": trim(quote_for(a, place))}, {"s": b["id"], "t": trim(quote_for(b, lines and lines[-1]["t"] or place))}]
    for l in lines:
        sp, ot = (a, b) if l["s"] == a["id"] else (b, a)
        remember(sp, "said", f"{ot['name']}யிடம் சொன்னேன்: {l['t']}", sim); remember(ot, "heard", f"{sp['name']} சொன்னார்: {l['t']}", sim)
        simlog.add("talk", l["t"], BY[l["s"]]["name"], clock(sim), with_=ot["name"], place=place, src=src)
    return jsonify({"lines": lines, "src": src})

@app.post("/api/event")      # anything thrown / any policy announced: EVERY agent reacts
def event():
    d = request.get_json() or {}; kind, text, sim = d.get("kind", "custom"), (d.get("text") or "").strip()[:600], d.get("sim")
    ids = [i for i in d.get("agents", []) if i in BY] or list(BY)
    where = d.get("where", "village")
    desc = (f"ஊருக்கு ஒரு பொது அறிவிப்பு / கொள்கை வெளியிடப்பட்டுள்ளது: \"{text}\"" if kind == "announce" else f"ஊருக்குள் ({where}) யாரோ ஒருவர் இதை வீசினார் / கொண்டு வந்தார்: {text}")
    simlog.add("event", f"{' ' if kind=='announce' else ' '}{text}", "You", clock(sim), ekind=kind)
    out = {}; src = "offline"

    # Build concise agent profiles (NO bloated memories that confuse the model with irrelevant interview quotes)
    agent_summaries = []
    for i in ids:
        a = BY[i]
        p = a.get("persona") or {}
        bio = p.get("bio_ta", a.get("bio", ""))
        style = p.get("speaking_style_ta", "")
        values = ", ".join(p.get("values_ta", [])[:3])
        agent_summaries.append(
            f"- ID: {a['id']} ({a['name']})\n"
            f"  பின்னணி: {bio}\n"
            f"  குணம் & பேச்சு: {style}\n"
            f"  முக்கிய மதிப்புகள்: {values}"
        )
    agents_text = "\n\n".join(agent_summaries)

    sys_inst = (
        "You are a master Tamil dialogue writer role-playing distinct villagers in a Tamil village simulation. "
        "Every villager must react in authentic, natural colloquial spoken Tamil (பேச்சுத் தமிழ்). "
        "\n\n"
        "CRITICAL RULES FOR REACTIONS:\n"
        "1. DIRECT RELEVANCE: Every villager MUST react DIRECTLY to the exact announcement, question, or statement made by the user. "
        "   NEVER speak about unrelated topics, past memories, or background facts.\n"
        "2. QUESTION ANSWERING: If the statement is a question, every villager must give their OWN authentic personal perspective "
        "   and reason directly answering what was asked, reflecting their personal background and core values.\n"
        "3. EMOTIONAL REALISM & CONFLICT: If the user's statement is provocative, insulting, accusatory, or rude "
        "   (such as calling them 'cockroaches', 'lazy', 'useless', etc.), real humans DO NOT ignore the insult! "
        "   Villagers must react with genuine human emotions: anger (கோபம்), shock (அதிர்ச்சி), indignation (தன்மானம்), "
        "   counter-questioning (எதிர்த்து கேள்வி கேட்டல்), or fierce defense of their hard work — matching their personality!\n"
        "4. NO PARROTING: Absolutely NEVER copy or parrot sample quotes, past interview snippets, or background notes. "
        "   Generate a fresh, sharp, realistic 1-2 sentence response that directly addresses what was said.\n"
        "5. Return strict JSON array with an entry for every villager."
    )

    prompt = (
        f"கிராம மக்கள் விபரம்:\n{agents_text}\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"பொது அறிவிப்பு / பயனர் சொன்ன வார்த்தை:\n"
        f"\"{text}\"\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        f"வழிமுறை: மேலே உள்ள வாசகத்தை (\"{text}\") கேட்டு ஒவ்வொரு கிராமவாசியும் தங்கள் சொந்த குணங்கள், மதிப்புகள் மற்றும் வாழ்க்கைப் பின்னணியின்படி "
        f"நேரடியாக எதிர்வினை ஆற்ற வேண்டும். கேள்வி கேட்டிருந்தால் நேரடிப் பதிலும் காரணமும் சொல்ல வேண்டும்; திட்டினாலோ அவமதித்தாலோ நேரடியாகக் கோபப்பட்டு தன்மானத்துடன் எதிர்த்துப் பேச வேண்டும்.\n\n"
        f"action இவற்றில் ஒன்று: approach (நேரில் வரல்), stay (நின்று பேசல்), gather (கூடுதல்), flee (விலகிப் போதல்), home (வீட்டுக்குச் செல்லல்).\n"
        f"JSON வடிவம் மட்டும்:\n"
        f"[\n"
        f"  {{\"id\": \"Abiseik\", \"text\": \"நேரடி தமிழ் எதிர்வினை\", \"action\": \"approach\", \"mood\": \"கோபம்\"}},\n"
        f"  ...\n"
        f"]"
    )

    try:
        raw_resp = llm.call(sys_inst, prompt, json_mode=True, temperature=0.9, max_tokens=2500)
        parsed_resp = llm.parse_json(raw_resp)
        for x in parsed_resp:
            if x.get("id") in ids and x.get("text"):
                out[x["id"]] = {
                    "id": x["id"],
                    "text": trim(x["text"], 240),
                    "action": x.get("action") if x.get("action") in ACTIONS else "approach",
                    "mood": x.get("mood", "")
                }
        if len(out) >= len(ids) * 0.7:
            src = "gemini"
    except Exception as e:
        print("[event] LLM error:", e)

    is_insult = bool(re.search(r"cockroach|கொக்கரொச்சை|சோம்பேறி|lazy|useless|முட்டாள்|waste|stupid|திருடன்|பொய்|திமிர்|மோசம்", text, re.I))
    got = set(out)
    for i in ids:
        if i not in out:
            a = BY[i]
            if is_insult:
                fb_text = f"என்னது? எங்களை பார்த்து இப்படி பேச உனக்கு எவ்வளவு தைரியம்! நாங்க எவ்வளவு கஷ்டப்பட்டு உழைக்கிறோம்னு தெரியுமா?"
                fb_act = "approach"
                fb_mood = "கோபம்"
            else:
                q = quote_for(a, text)
                fb_text = q
                fb_act = "gather" if kind == "announce" else "approach"
                fb_mood = "அமைதி"
            out[i] = {"id": i, "text": trim(fb_text), "action": fb_act, "mood": fb_mood}

    res = []
    for i in ids:
        r = out[i]
        a = BY[i]
        r["src"] = src if i in got else "offline"
        res.append(r)
        remember(a, "event", f"{'அறிவிப்பு' if kind=='announce' else 'சம்பவம்'}: {text[:90]} — என் எதிர்வினை: {r['text']}", sim)
        simlog.add("react", r["text"], a["name"], clock(sim), action=r["action"], mood=r["mood"], src=r["src"])
    return jsonify({"reactions": res, "src": src})

# ---------- decentralized deliberation routes ----------
import sys, deliberation

@app.post("/api/deliberate/start")
def deliberate_start():
    try:
        d = request.get_json() or {}
        announcement = (d.get("announcement") or d.get("text") or "").strip()
        sim = clock(d.get("sim")) or "10:00"
        if not announcement:
            return jsonify({"error": "announcement required"}), 400
        session, turn_1 = deliberation.MANAGER.start_deliberation(announcement, sim, import_server=sys.modules[__name__])
        return jsonify({"session": session, "turn": turn_1})
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        try:
            with open("delib_error.txt", "w", encoding="utf-8") as f:
                f.write(tb)
        except Exception: pass
        return jsonify({"error": str(e), "traceback": tb}), 500

@app.post("/api/deliberate/turn")
def deliberate_turn():
    try:
        d = request.get_json() or {}
        sim = clock(d.get("sim")) or "10:05"
        session, turn = deliberation.MANAGER.next_turn(sim, import_server=sys.modules[__name__])
        if not session:
            return jsonify({"error": "no active deliberation"}), 400
        return jsonify({"session": session, "turn": turn})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.post("/api/deliberate/conclude")
def deliberate_conclude():
    try:
        d = request.get_json() or {}
        sim = clock(d.get("sim")) or "10:15"
        session = deliberation.MANAGER.conclude_deliberation(sim, import_server=sys.modules[__name__])
        if not session:
            return jsonify({"error": "no active deliberation"}), 400
        return jsonify({"session": session, "conclusion": session.get("conclusion")})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.get("/api/deliberate/session")
def deliberate_session():
    session = deliberation.MANAGER.get_session()
    return jsonify({"session": session})


threading.Thread(target=prepare, daemon=True).start()
if __name__ == "__main__":
    print(f"Tamil village on http://127.0.0.1:5001  (Gemini: {'ON '+llm.MODEL if llm.available() else 'OFF – offline quote mode; set GEMINI_API_KEY'})")
    app.run(host="127.0.0.1", port=5001, threaded=True)
