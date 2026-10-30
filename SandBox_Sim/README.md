# Tamil Autonomous Village (reflect-system agents)

10 villagers are **built from your reflect-system files** (`data/participants/<Name>/expert_ta/*.txt`).
Nothing about a person is hard-coded: bio, speaking style, memories, daily routine and every spoken line come from those files + Gemini.
Add/remove a folder in `data/participants/` and the village changes (first run rebuilds that person's persona).

## Run
    pip install -r requirements.txt
    cp .env.example .env      # put your GEMINI_API_KEY inside  (or set the env var)
    python server.py          # http://127.0.0.1:5001   (frontend is prebuilt; or: cd frontend && npm i && npm run build)

On start the server builds a **persona card + daily schedule per participant with Gemini** (10 calls, cached in `data/cache/`, rebuilt only if the files change; the ↻ button forces it).
Free-tier friendly: calls are spaced (`GEMINI_MIN_GAP`, default 1.2 s) and an announcement uses ONE batched call for all villagers.

## How it works
- **Autonomous speech (Tamil)**: villagers follow their generated schedule, remark on what they're doing (on arrival + every 1-2 min, adjustable in the UI),
  and when two meet they hold a Gemini-written dialogue grounded in both personas and their recent memories.
- **Announce a policy / throw anything**: type any announcement (Tamil or English) -> every villager reacts in their own voice,
  and Gemini chooses their action (gather at the hall / approach / flee / go home / carry on). Objects (apple, ball, gift, rock, fire, party or any custom text) work the same way.
- **Talk to a villager**: select them, type in the chat box (Tamil/English) -> they answer in Tamil, in character.
- **Dual memory per agent** (`data/memory/<Name>.json`, visible in the memory panel):
  - old / long-term: seeded from the reflect system + auto-summaries of past experience.
  - new / short-term: the last ~10 things they said, heard, were told, or reacted to. When it overflows, the oldest 5 are summarised into long-term.
  Both are injected into every prompt, so an announcement made earlier shows up in later conversations.
- **Log**: everything (speech, chats, announcements, reactions, doors, throws) -> `data/logs/sim_log.jsonl`, shown live in the UI and downloadable.

## Offline mode (no API key)
Everything still runs: villagers speak real quotes from the participant's own interview (picked by keyword match to the situation),
schedules are a generic rotation. Context-aware Tamil dialogue needs the Gemini key.

## Kept from v4
Realistic canvas renderer, lockable doors, click-to-control a villager (WASD / click), day-night cycle, two-floor interiors. 10 villagers share 4 houses.

## Files
    server.py   Flask API: /api/agents /say /chat /converse /event /memory/<id> /log /prepare /status
    reflect.py  loads participants, builds persona + schedule (Gemini, cached; offline fallback)
    memory.py   dual memory + consolidation     llm.py  Gemini wrapper     simlog.py  persistent log
    frontend/   React UI (Tamil) + canvas engine    tests/test_pipeline_stub.py  runs the whole pipeline with a fake Gemini
