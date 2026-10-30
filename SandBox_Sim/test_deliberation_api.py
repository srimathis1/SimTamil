import sys, io, requests, json
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

BASE = "http://127.0.0.1:5001"
announcement = "தமிழ்நாட்டில் பெண்களுக்கு இலவசப் பேருந்து வசதி தேவையா? இது பெண்களின் சுதந்திரத்தையும் பொருளாதாரத்தையும் மேம்படுத்துகிறதா, அல்லது அரசுக்கு கூடுதல் நிதிச்சுமையை ஏற்படுத்துகிறதா?"

print("1. Starting Deliberation...")
r1 = requests.post(f"{BASE}/api/deliberate/start", json={"announcement": announcement, "sim": "630"}).json()
turn1 = r1["turn"]
print(f"Turn 1 Speaker : {turn1['agent_name']} (Score: {turn1['score']}/10) [{turn1['lens_ta']}]")
print(f"Selection Reason: {turn1['reason']}")
print(f"Opinion         : {turn1['text']}\n")

print("2. Generating Turn 2 (Decentralized Ethical Friction)...")
r2 = requests.post(f"{BASE}/api/deliberate/turn", json={"sim": "635"}).json()
turn2 = r2["turn"]
print(f"Turn 2 Speaker : {turn2['agent_name']} (Reaction: {turn2['score']}/10) [{turn2['lens_ta']}]")
print(f"Ethical Reason : {turn2['reason']}")
print(f"Counter-Arg    : {turn2['text']}\n")

print("3. Generating Turn 3 (Counter-Deliberation)...")
r3 = requests.post(f"{BASE}/api/deliberate/turn", json={"sim": "640"}).json()
turn3 = r3["turn"]
print(f"Turn 3 Speaker : {turn3['agent_name']} (Reaction: {turn3['score']}/10) [{turn3['lens_ta']}]")
print(f"Reason         : {turn3['reason']}")
print(f"Opinion        : {turn3['text']}\n")

print("4. Concluding Deliberation...")
r_end = requests.post(f"{BASE}/api/deliberate/conclude", json={"sim": "645"}).json()
conc = r_end["conclusion"]
print(f"Consensus Conclusion: {conc['text']}")
print(f"Total Turns: {conc['total_turns']} | Participants: {conc['participants_count']}")
