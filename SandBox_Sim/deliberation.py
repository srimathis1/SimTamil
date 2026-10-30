"""
Decentralized Multi-Agent Deliberation & Turn-Taking Engine for Tamil Village Simulation.
Features:
1. Dynamic Reflect-System Relevance Scoring for Turn 1 (Determines who is most impacted).
2. Shared Discussion Memory (கிராம சபை பொது நினைவகம்) across all agents.
3. Individual Agent Memory (தனிநபர் நினைவகம்) per agent via memory.py.
4. Decentralized Turn-Taking based on Ethical Friction, Moral Offense, or High Relevance to the prior statement.
5. Interactive Turn-by-Turn or Auto-Debate loop until stopped or concluded with consensus.
"""

import sys, io, os, re, json, time, threading
from pathlib import Path

# UTF-8 safety
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import llm, memory, simlog

LENS_INFO = {
    "demographer": {
        "en": "Demographics & Identity",
        "ta": "அடையாளம் & பின்னணி",
        "color": "#38bdf8"
    },
    "psychologist": {
        "en": "Psychology & Emotions",
        "ta": "உளவியல் & மனோநிலை",
        "color": "#f472b6"
    },
    "sociologist": {
        "en": "Sociology & Community",
        "ta": "சமூகவியல் & உறவுகள்",
        "color": "#a78bfa"
    },
    "behavioral_economist": {
        "en": "Behavioral Economics",
        "ta": "பொருளாதாரம் & வாழ்வாதாரம்",
        "color": "#fbbf24"
    },
    "political_scientist": {
        "en": "Political Science & Ethics",
        "ta": "அரசியல் & அறநெறி",
        "color": "#34d399"
    }
}

ACTIVE_SESSION = None
_lock = threading.RLock()

def get_lens_badge(lens_key):
    info = LENS_INFO.get(lens_key, {"ta": lens_key, "color": "#94a3b8"})
    return info["ta"], info["color"]

def score_initial_relevance(agents, announcement):
    """
    Evaluates all agents against the announcement.
    Returns ranked list: [{"id": ..., "score": float, "lens": ..., "reason": ...}]
    """
    summaries = []
    for a in agents:
        p = a.get("persona") or {}
        bullets = a.get("bullets", {})
        bullet_summaries = []
        for l_key in ["demographer", "psychologist", "sociologist", "behavioral_economist", "political_scientist"]:
            titles = [b["title"] for b in bullets.get(l_key, [])[:2]]
            if titles:
                bullet_summaries.append(f"{l_key}: {', '.join(titles)}")

        summaries.append({
            "id": a["id"],
            "name": a["name"],
            "bio": p.get("bio_ta", "")[:100],
            "values": ", ".join(p.get("values_ta", [])[:3]),
            "reflect_lenses": " | ".join(bullet_summaries[:4])
        })

    prompt = (
        f"கிராம சபை பொது அறிவிப்பு / கொள்கை / கேள்வி:\n"
        f"\"{announcement}\"\n\n"
        f"கிராமவாசிகள் மற்றும் அவர்களின் நிபுணர் பார்வைகள் (Reflect Systems):\n" +
        "\n".join(
            f"- {s['id']} ({s['name']}): {s['bio']}\n"
            f"  மதிப்புகள்: {s['values']}\n"
            f"  நிபுணர் விவரங்கள்: {s['reflect_lenses']}"
            for s in summaries
        ) +
        "\n\n"
        "பணி: மேலே உள்ள அறிவிப்பு யாருக்கு அவர்களது பின்னணி, பாலினம், பொருளாதார நிலை, மற்றும் நிபுணர் பார்வை (reflect system) அடிப்படையில் "
        "மிக அதிகமாகத் தொடர்புடையது என்று 1.0 முதல் 10.0 வரை துல்லியமாக மதிப்பிடு.\n"
        "Return strict JSON array with an entry for each agent:\n"
        "[\n"
        "  {\n"
        "    \"id\": \"AgentId\",\n"
        "    \"score\": 9.5,\n"
        "    \"lens\": \"political_scientist\",\n"
        "    \"reason\": \"காரணம் தமிழில் ஒரு சுருக்கமான வரியில் (எ.கா: பெண்கள் உரிமை மற்றும் கல்வி சார்ந்த விவகாரம் என்பதால்)\"\n"
        "  },\n"
        "  ...\n"
        "]"
    )

    try:
        raw = llm.call(
            "You are an impartial village council coordinator. Evaluate topic relevance for each villager strictly in JSON.",
            prompt, json_mode=True, temperature=0.15, max_tokens=1500
        )
        ranked = llm.parse_json(raw)
        if isinstance(ranked, list) and len(ranked) > 0:
            for item in ranked:
                try: item["score"] = float(item.get("score", 5.0))
                except Exception: item["score"] = 5.0
                if "lens" not in item or item["lens"] not in LENS_INFO:
                    item["lens"] = "sociologist"
            ranked.sort(key=lambda x: x["score"], reverse=True)
            return ranked
    except Exception as e:
        print("[deliberation] relevance scoring error:", e)

    # Heuristic fallback if LLM fails
    fallback = []
    ann_lower = announcement.lower()
    for a in agents:
        p = a.get("persona") or {}
        bio = p.get("bio_ta", "")
        sc = 6.0
        lens = "sociologist"
        if re.search(r"பணம்|வேலை|கடன்|நிதி|பொருளாதார", ann_lower):
            lens = "behavioral_economist"; sc = 8.0 if "விவசாய" in bio or "படிப்பு" in bio else 6.5
        elif re.search(r"பெண்|கல்வி|மாணவ|பஸ்|பயண", ann_lower):
            lens = "political_scientist"; sc = 9.0 if ("மாணவி" in bio or "பெண்" in bio) else 6.0
        fallback.append({
            "id": a["id"], "score": sc, "lens": lens,
            "reason": "சுயவிவர மற்றும் நிபுணர் பார்வையின் அடிப்படையில் தேர்ந்தெடுக்கப்பட்டார்."
        })
    fallback.sort(key=lambda x: x["score"], reverse=True)
    return fallback

def score_next_reaction(agents, announcement, shared_turns, last_speaker_id, import_server=None):
    """
    Evaluates which remaining agent feels the highest ethical friction, moral offense,
    or urge to respond to the previous speaker's statement based on their reflect systems.
    """
    last_turn = shared_turns[-1]
    last_speaker_name = last_turn["agent_name"]
    last_statement = last_turn["text"]

    # History snippet of last 3 turns
    history_lines = [f"{t['agent_name']}: \"{t['text']}\"" for t in shared_turns[-3:]]
    history_text = "\n".join(history_lines)

    # Count how many times each agent spoke so far to avoid monopoly
    turn_counts = {}
    for t in shared_turns:
        turn_counts[t["agent_id"]] = turn_counts.get(t["agent_id"], 0) + 1

    candidate_agents = [a for a in agents if a["id"] != last_speaker_id]

    summaries = []
    for a in candidate_agents:
        p = a.get("persona") or {}
        bullets = a.get("bullets", {})
        bullet_titles = []
        for l_key in ["demographer", "psychologist", "sociologist", "behavioral_economist", "political_scientist"]:
            titles = [b["title"] for b in bullets.get(l_key, [])[:2]]
            if titles:
                bullet_titles.append(f"{l_key}: {', '.join(titles)}")

        summaries.append({
            "id": a["id"],
            "name": a["name"],
            "bio": p.get("bio_ta", "")[:90],
            "values": ", ".join(p.get("values_ta", [])[:3]),
            "reflect_lenses": " | ".join(bullet_titles[:3]),
            "times_spoken": turn_counts.get(a["id"], 0)
        })

    prompt = (
        f"கிராம சபை விவாதப் பிரதான தலைப்பு (ORIGINAL DEBATE TOPIC - STRICT REQUIREMENT):\n"
        f"👉 \"{announcement}\"\n\n"
        f"விவாதத்தில் இதுவரை பேசியவை (Shared Discussion History):\n{history_text}\n\n"
        f"கடைசியாக பேசியவர்: [{last_speaker_name}]\n"
        f"அவர் கூறிய கருத்து:\n\"{last_statement}\"\n\n"
        f"அடுத்ததாகப் பேச வாய்ப்புள்ள கிராமவாசிகள்:\n" +
        "\n".join(
            f"- {s['id']} ({s['name']}): {s['bio']}\n"
            f"  மதிப்புகள்: {s['values']} | பார்வைகள்: {s['reflect_lenses']}\n"
            f"  (இதுவரை பேசிய முறை: {s['times_spoken']})"
            for s in summaries
        ) +
        "\n\n"
        "கண்டிப்பான வழிகாட்டுதல் (STRICT RULES):\n"
        f"1. தலைப்பில் மட்டுமே கவனம் (STAY ON TOPIC): விவாதத்தின் தலைப்பு \"{announcement}\". இதிலிருந்து விலகி வேறு பொதுவான உபதேசங்களை பேசுபவர்களைத் தவிர்க்கவும்.\n"
        f"2. முரண்பாடு உள்ளவர்கள் மட்டுமே (CONTRADICTION ONLY): முந்தைய பேச்சாளரின் ({last_speaker_name}) கருத்தை ஆமோதிப்பவர்கள், வழிமொழிபவர்கள் அல்லது ஏற்றுக்கொள்பவர்களை முற்றிலும் தேர்ந்தெடுக்கக் கூடாது! "
        f"யார் அவரது கூற்றை நேரடியாக மறுக்கிறார்கள், அதில் உள்ள பிழை அல்லது ஆபத்தை சுட்டிக்காட்டி கடுமையான மாற்றுக் கருத்தை (Contradiction / Opposition) முன்வைக்கிறார்கள் என்பதை மட்டுமே மதிப்பிட வேண்டும்.\n"
        "3. முரண்பாடு இல்லாத நபர்களுக்கு 1.0 முதல் 3.0 வரை மட்டுமே கொடு. உண்மையான மாற்றுக் கருத்து மற்றும் தர்க்க முரண்பாடு உள்ளவருக்கு 8.0 முதல் 10.0 வரை மதிப்பெண் கொடு.\n"
        "Return strict JSON array with candidates ranked by genuine contradiction against the prior statement:\n"
        "[\n"
        "  {\n"
        "    \"id\": \"AgentId\",\n"
        "    \"reaction_rate\": 9.4,\n"
        "    \"has_contradiction\": true,\n"
        "    \"lens\": \"political_scientist\",\n"
        "    \"reason\": \"காரணம் தமிழில்: [{last_speaker_name}] கூறிய இந்த கருத்தை இவர் எப்படி மறுக்கிறார்\"\n"
        "  },\n"
        "  ...\n"
        "]"
    )

    try:
        raw = llm.call(
            "You are a village council debate arbitrator. Identify which villager has a genuine contradiction or opposition against the prior statement on the topic. Reject anyone who merely agrees.",
            prompt, json_mode=True, temperature=0.15, max_tokens=1500
        )
        ranked = llm.parse_json(raw)
        if isinstance(ranked, list) and len(ranked) > 0:
            res = []
            for item in ranked:
                ag = next((a for a in agents if a["id"] == item.get("id")), None)
                if not ag:
                    continue
                try: r_rate = float(item.get("reaction_rate", 5.0))
                except Exception: r_rate = 5.0

                # Only include agents with genuine opposition / contradiction
                has_contra = item.get("has_contradiction", True)
                if not has_contra or r_rate < 4.5:
                    continue

                penalty = turn_counts.get(ag["id"], 0) * 1.5
                eff_score = max(1.0, r_rate - penalty)
                lens = item.get("lens", "sociologist")
                if lens not in LENS_INFO:
                    lens = "political_scientist"
                res.append({
                    "id": ag["id"],
                    "name": ag["name"],
                    "color": ag.get("color", "#4ade80"),
                    "reaction_rate": round(r_rate, 1),
                    "effective_score": round(eff_score, 1),
                    "is_offended_or_opposing": True,
                    "lens": lens,
                    "lens_ta": LENS_INFO[lens]["ta"],
                    "lens_en": LENS_INFO[lens]["en"],
                    "lens_color": LENS_INFO[lens]["color"],
                    "reason": item.get("reason", "தன்னுடைய அறநெறி மற்றும் சுயவிவரப் பார்வையின்படி மாற்றுக் கருத்தைக் கொண்டுள்ளார்.")
                })
            if res:
                res.sort(key=lambda x: x["effective_score"], reverse=True)
                return res
    except Exception as e:
        print("[deliberation] reaction friction error:", e)

    # Fallback: pick candidate who spoke least
    candidate_agents.sort(key=lambda a: turn_counts.get(a["id"], 0))
    chosen = candidate_agents[0]
    lens = "political_scientist"
    return [{
        "id": chosen["id"],
        "name": chosen["name"],
        "color": chosen.get("color", "#4ade80"),
        "reaction_rate": 7.5,
        "effective_score": 7.5,
        "is_offended_or_opposing": True,
        "lens": lens,
        "lens_ta": LENS_INFO[lens]["ta"],
        "lens_en": LENS_INFO[lens]["en"],
        "lens_color": LENS_INFO[lens]["color"],
        "reason": f"[{last_speaker_name}] முன்வைத்த கருத்தை மறுத்து, தலைப்புக்குரிய தன் பார்வையை முன்வைக்கிறார்."
    }]

def generate_speaker_turn(speaker, lens, reason, announcement, shared_turns, is_first=False, import_server=None):
    """Generates the speech turn for the chosen agent grounded in their reflect system and standard Tamil."""
    import server
    srv = import_server or server

    # Retrieve reflect system bullets
    chosen_lenses = [lens, "psychologist"] if lens != "psychologist" else ["psychologist", "demographer"]
    prompt_block = srv.block_with_reflect(speaker, chosen_lenses, n_long=6, n_short=4)

    # Format shared memory transcript
    if is_first:
        context_section = (
            f"══════════════════════════════════════════════════\n"
            f"🔒 TOPIC LOCK (Do NOT discuss anything else):\n"
            f"   \"{announcement}\"\n"
            f"══════════════════════════════════════════════════\n\n"
            f"தேர்வு காரணம்: {reason}\n\n"
            f"கட்டாய விதிகள் (MANDATORY RULES):\n"
            f"1. 🔒 STAY ON TOPIC: நீ கூறுவது 100% '{announcement}' என்ற தலைப்பிற்கு மட்டுமே தொடர்புடையதாக இருக்க வேண்டும். வேறு எந்த பொதுவான கேள்வியையும் அல்லது உபதேசத்தையும் பேசக் கூடாது.\n"
            f"2. உன் '{LENS_INFO[lens]['ta']}' நிபுணர் பார்வையிலிருந்து உன் உண்மையான, நேரடியான நிலைப்பாட்டை கூறு.\n"
            f"3. இயல்பான மரியாதையான தமிழ் நடையில் 2 எளிய வாக்கியங்களில் மட்டும் பேசு."
        )
    else:
        last_turn = shared_turns[-1]
        shared_history_lines = [f"{i+1}. [{t['agent_name']}]: \"{t['text']}\"" for i, t in enumerate(shared_turns[-4:])]
        shared_history_text = "\n".join(shared_history_lines)

        context_section = (
            f"══════════════════════════════════════════════════\n"
            f"🔒 ORIGINAL TOPIC LOCK (Never deviate from this):\n"
            f"   \"{announcement}\"\n"
            f"══════════════════════════════════════════════════\n\n"
            f"கிராம சபை விவாத வரலாறு (Recent turns):\n{shared_history_text}\n\n"
            f"கடைசியாகப் பேசியவர்: [{last_turn['agent_name']}]: \"{last_turn['text']}\"\n"
            f"உன் ஆட்சேபனை காரணம்: {reason}\n\n"
            f"கட்டாய விதிகள் (MANDATORY RULES - Violation = failure):\n"
            f"1. 🔒 TOPIC LOCK: உன் வாதம் '{announcement}' என்ற மூல தலைப்பில் மட்டுமே இருக்க வேண்டும். வேறு தலைப்பிற்கு போகக் கூடாது.\n"
            f"2. 🚫 NO AGREEMENT: [{last_turn['agent_name']}] கூறிய கருத்தை ஒத்துக் கொள்ளக் கூடாது. நேரடி மறுப்பையோ அல்லது மாற்றுக் கருத்தையோ மட்டுமே கூறு.\n"
            f"3. உன் '{LENS_INFO[lens]['ta']}' பார்வையிலிருந்து தன்மானத்தோடு மரியாதையாக வாதிடு.\n"
            f"4. இயல்பான தமிழில் 2 வாக்கியங்களில் மட்டும் பேசு."
        )


    full_prompt = f"{prompt_block}\n\n{context_section}"

    try:
        raw = llm.call(
            f"You are a Tamil village council debate moderator. The villager MUST speak ONLY about the announced topic: \"{announcement}\". "
            f"They must NOT agree with the last speaker. They must express a genuine counter-argument or different stance. "
            f"Use standard respectful Tamil (தமிழ்). 2 sentences maximum. Follow all MANDATORY RULES exactly.",
            full_prompt, temperature=0.3, max_tokens=220
        )
        speech = srv.trim(raw, 300)
    except Exception as e:
        print("[deliberation] speech generation error:", e)
        speech = srv.trim(srv.quote_for(speaker, announcement), 250)

    # Determine mood / action
    mood = "தீவிர விவாதம்"
    action = "approach" if is_first else "gather"
    if re.search(r"ஏற்க முடியாது|தவறு|அவசியம்|கோபம்|நியாயம்|ஆட்சேபனை|எதிர்ப்பு", speech):
        mood = "காரசாரமான மறுப்பு"
    else:
        mood = "மாற்றுக் கருத்து"

    return speech, mood, action

class DeliberationManager:
    def __init__(self):
        self.session = None

    def start_deliberation(self, announcement, sim_clock="10:00", import_server=None):
        import server
        srv = import_server or server
        agents = srv.AGENTS

        with _lock:
            # 1. Score initial relevance
            ranked = score_initial_relevance(agents, announcement)
            first_pick = ranked[0]
            speaker = srv.BY.get(first_pick["id"]) or agents[0]
            lens = first_pick.get("lens", "sociologist")
            reason = first_pick.get("reason", "இந்த அறிவிப்பு இவரது பின்னணிக்கு மிக நெருக்கமானது.")
            rel_score = first_pick.get("score", 9.0)

            # 2. Generate Turn 1
            speech, mood, action = generate_speaker_turn(speaker, lens, reason, announcement, [], is_first=True, import_server=srv)

            # 3. Create Shared Memory Session
            turn_1 = {
                "turn": 1,
                "agent_id": speaker["id"],
                "agent_name": speaker["name"],
                "color": speaker.get("color", "#4ade80"),
                "lens": lens,
                "lens_ta": LENS_INFO[lens]["ta"],
                "lens_en": LENS_INFO[lens]["en"],
                "lens_color": LENS_INFO[lens]["color"],
                "score": rel_score,
                "reason": reason,
                "is_offended": False,
                "text": speech,
                "action": action,
                "mood": mood,
                "time": sim_clock
            }

            self.session = {
                "id": f"delib_{int(time.time())}",
                "announcement": announcement,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sim_clock": sim_clock,
                "status": "active",
                "turns": [turn_1],
                "initial_rankings": ranked[:5],
                "participants": [speaker["id"]],
                "conclusion": None,
                "next_candidate": None,
                "candidate_rankings": [],
                "can_conclude": False
            }

            # 4. Pre-calculate candidate who has highest conflict / desperation to reply next
            upcoming = score_next_reaction(agents, announcement, [turn_1], speaker["id"], import_server=srv)
            self.session["next_candidate"] = upcoming[0] if upcoming else None
            self.session["candidate_rankings"] = upcoming[:3] if upcoming else []

            # 5. Update Memories
            # Speaker individual memory
            memory.add(speaker, "deliberate", f"கிராம சபை பொது அறிவிப்பில் ('{announcement[:50]}') நான் முதலில் பேசினேன்: '{speech}'", sim_clock)
            # Other agents hear in their individual memory
            for a in agents:
                if a["id"] != speaker["id"]:
                    memory.add(a, "deliberate", f"கிராம சபையில் {speaker['name']} கூறினார்: '{speech}'", sim_clock)

            # Log to village simulation log
            simlog.add("deliberate", speech, speaker["name"], sim_clock, lens=LENS_INFO[lens]["ta"], turn=1)

            # Terminal Banner
            self._print_terminal_banner(turn_1, is_first=True)

            return self.session, turn_1

    def next_turn(self, sim_clock="10:05", import_server=None):
        import server
        srv = import_server or server
        agents = srv.AGENTS

        with _lock:
            if not self.session or self.session["status"] != "active":
                return None, None

            shared_turns = self.session["turns"]
            announcement = self.session["announcement"]
            last_turn = shared_turns[-1]
            turn_num = len(shared_turns) + 1

            # 1. Pick candidate who has highest friction / desperation
            if self.session.get("next_candidate"):
                next_pick = self.session["next_candidate"]
            else:
                reactions = score_next_reaction(agents, announcement, shared_turns, last_turn["agent_id"], import_server=srv)
                next_pick = reactions[0]

            speaker = srv.BY.get(next_pick["id"]) or agents[0]
            lens = next_pick.get("lens", "political_scientist")
            reason = next_pick.get("reason", "கடைசி கருத்திற்கு மாற்றுக் கருத்தை முன்வைக்கிறார்.")
            r_rate = next_pick.get("reaction_rate", 8.5)
            is_offended = next_pick.get("is_offended_or_opposing", True)

            # 2. Generate Turn Response
            speech, mood, action = generate_speaker_turn(
                speaker, lens, reason, announcement, shared_turns, is_first=False, import_server=srv
            )

            # 3. Create Turn Record
            turn_record = {
                "turn": turn_num,
                "agent_id": speaker["id"],
                "agent_name": speaker["name"],
                "color": speaker.get("color", "#4ade80"),
                "lens": lens,
                "lens_ta": LENS_INFO[lens]["ta"],
                "lens_en": LENS_INFO[lens]["en"],
                "lens_color": LENS_INFO[lens]["color"],
                "score": r_rate,
                "reason": reason,
                "is_offended": is_offended,
                "text": speech,
                "action": action,
                "mood": mood,
                "time": sim_clock
            }

            self.session["turns"].append(turn_record)
            if speaker["id"] not in self.session["participants"]:
                self.session["participants"].append(speaker["id"])

            # 4. Pre-calculate next candidate for the upcoming turn
            upcoming = score_next_reaction(agents, announcement, self.session["turns"], speaker["id"], import_server=srv)
            self.session["next_candidate"] = upcoming[0] if upcoming else None
            self.session["candidate_rankings"] = upcoming[:3] if upcoming else []

            top_rate = upcoming[0].get("reaction_rate", 5.0) if upcoming else 0
            if top_rate < 5.0 or len(self.session["turns"]) >= 8:
                self.session["can_conclude"] = True
            else:
                self.session["can_conclude"] = False

            # 5. Update Memories
            memory.add(speaker, "deliberate", f"கிராம சபையில் நான் {last_turn['agent_name']}க்கு பதில் கூறினேன்: '{speech}'", sim_clock)
            for a in agents:
                if a["id"] != speaker["id"]:
                    memory.add(a, "deliberate", f"கிராம சபையில் {speaker['name']} வாதம் செய்தார்: '{speech}'", sim_clock)

            simlog.add("deliberate", speech, speaker["name"], sim_clock, lens=LENS_INFO[lens]["ta"], turn=turn_num)

            # Terminal Banner
            self._print_terminal_banner(turn_record, is_first=False, last_speaker=last_turn["agent_name"])

            return self.session, turn_record

    def conclude_deliberation(self, sim_clock="10:15", import_server=None):
        import server
        srv = import_server or server
        agents = srv.AGENTS

        with _lock:
            if not self.session:
                return None

            shared_turns = self.session["turns"]
            announcement = self.session["announcement"]

            turns_transcript = "\n".join(f"- {t['agent_name']} ({t['lens_ta']}): \"{t['text']}\"" for t in shared_turns)

            prompt = (
                f"கிராம சபை பொது அறிவிப்பு: \"{announcement}\"\n\n"
                f"கிராமவாசிகள் விவாதித்த கருத்துக்கள்:\n{turns_transcript}\n\n"
                "பணி: கிராம சபை விவாதம் முடிவுக்கு வருகிறது. அனைத்து கிராமவாசிகளின் வாதங்களையும் சமநிலையாகப் பார்த்து, "
                "ஊருக்குப் பொதுவான ஒருமித்த முடிவை அல்லது சமரசக் கருத்தை 2 வாக்கியங்களில் தமிழில் முடிவாகக் கூறு (Conclusion / Consensus).\n"
                "ஒரு வரியில் 'முடிவு (Consensus)' மற்றும் மற்றொரு வரியில் 'கிராமத்தின் நிலைப்பாடு (Village Agreement)' தருமாறு எழுது."
            )

            try:
                conclusion_text = srv.trim(llm.call(
                    "You are the village council elder synthesizing a consensus conclusion in standard Tamil.",
                    prompt, temperature=0.25, max_tokens=250
                ), 300)
            except Exception:
                conclusion_text = "அனைவரின் கருத்துக்களையும் கேட்டறிந்தோம்; கிராமத்தின் அமைதியையும் அனைத்து மக்களின் நலனையும் கருத்தில் கொண்டு ஒருமித்த முடிவெடுக்கப்படும்."

            self.session["status"] = "concluded"
            self.session["can_conclude"] = False
            self.session["next_candidate"] = None
            self.session["candidate_rankings"] = []
            self.session["conclusion"] = {
                "text": conclusion_text,
                "time": sim_clock,
                "total_turns": len(shared_turns),
                "participants_count": len(self.session["participants"])
            }

            # Update memories with the conclusion
            for a in agents:
                memory.add(a, "deliberate", f"கிராம சபை விவாதத்தின் இறுதி முடிவு: '{conclusion_text}'", sim_clock)

            simlog.add("deliberate_conclusion", conclusion_text, "கிராம சபை", sim_clock)

            try:
                print("\n" + "=" * 78)
                print("  VILLAGE DELIBERATION CONCLUDED")
                print(f"  Conclusion: {conclusion_text}")
                print(f"  Turns: {len(shared_turns)} | Participants: {len(self.session['participants'])}")
                print("=" * 78 + "\n", flush=True)
            except Exception:
                pass

            return self.session

    def get_session(self):
        with _lock:
            return self.session

    def _print_terminal_banner(self, turn, is_first=False, last_speaker=""):
        try:
            sep = "=" * 78
            print(f"\n{sep}")
            if is_first:
                print(f"  [VILLAGE DELIBERATION STARTED]  Turn 1")
                print(f"  Announcement : \"{self.session['announcement'][:65]}...\"")
                print(f"  First Speaker: {turn['agent_name']} (Relevance Score: {turn['score']}/10)")
                print(f"  Trigger Lens : {turn['lens']} — {turn['lens_ta']}")
                print(f"  Selection Why: {turn['reason']}")
                print(f"  Opinion      : \"{turn['text']}\"")
            else:
                print(f"  [DECENTRALIZED TURN-TAKING]  Turn {turn['turn']}")
                print(f"  Reply To     : [{last_speaker}]")
                print(f"  Next Speaker : {turn['agent_name']} (Reaction Rate: {turn['score']}/10)")
                print(f"  Trigger Lens : {turn['lens']} — {turn['lens_ta']}")
                print(f"  Ethical Why   : {turn['reason']}")
                print(f"  Counter-Arg  : \"{turn['text']}\"")
            print(f"{sep}\n", flush=True)
        except Exception:
            pass

MANAGER = DeliberationManager()

