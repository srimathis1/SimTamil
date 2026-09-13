"""
scripts/semantic_harmonizer_eval.py -- Dynamic cross-wave matching without file duplication.
"""

import os
import re
import sys
import json
import time
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(override=True)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")

if not api_key:
    print("Error: No GEMINI_API_KEY or API_KEY found in .env")
    sys.exit(1)

client = genai.Client(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

def clean_text(text: str) -> str:
    clean = re.sub(r'\[.*?\]', '', text)
    clean = re.sub(r'^(question|item|scenario|economic game)\s*', '', clean, flags=re.IGNORECASE)
    return re.sub(r'[^\w\s]', '', clean).strip().lower()

def parse_survey(filepath: Path) -> dict:
    if not filepath.exists():
        return {}
    records = {}
    lines = filepath.read_text(encoding="utf-8").splitlines()
    pending_prompt = None

    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith(("#", "===", "---")):
            continue

        is_q = bool(re.search(r'^(question|item|scenario|economic game)\b', raw, flags=re.IGNORECASE) or "[" in raw)

        if is_q:
            parts = re.split(r'[:|=]\s*', raw, maxsplit=1)
            prompt_candidate = parts[1].strip() if len(parts) > 1 else raw
            same_line_val = None
            if len(parts) == 2 and re.match(r'^(Raw\s*=|Scored\s*=|[-+]?\d+$)', parts[1].strip(), re.IGNORECASE):
                same_line_val = parts[1].strip()
                prompt_candidate = parts[0].strip()

            if same_line_val is not None:
                key = clean_text(prompt_candidate)
                records[key] = {"prompt": prompt_candidate, "val": same_line_val}
                pending_prompt = None
            else:
                pending_prompt = prompt_candidate
            continue

        if pending_prompt and re.search(r'(participant response|response)', raw, flags=re.IGNORECASE):
            parts = re.split(r'[:|=]\s*', raw, maxsplit=1)
            val_str = parts[1].strip() if len(parts) > 1 else raw
            key = clean_text(pending_prompt)
            records[key] = {"prompt": pending_prompt, "val": val_str}
            pending_prompt = None
            continue

        if "|" in raw:
            parts = raw.split("|", 1)
            key = clean_text(parts[0])
            records[key] = {"prompt": parts[0].strip(), "val": parts[1].strip()}
        elif ":" in raw:
            parts = re.split(r'[:|=]\s*', raw, maxsplit=1)
            if len(parts) == 2 and not any(parts[0].lower().startswith(m) for m in ["structure", "participant", "module", "date"]):
                key = clean_text(parts[0])
                records[key] = {"prompt": parts[0].strip(), "val": parts[1].strip()}

    return records

def assemble_agent_memory() -> str:
    sections = []
    w1_file = DATA_DIR / "surveys_wave1.txt"
    if w1_file.exists():
        sections.append(f"=== WAVE 1 BASELINE DATA ===\n{w1_file.read_text(encoding='utf-8').strip()}")
    refl_dir = DATA_DIR / "Expert_reflection"
    if refl_dir.exists():
        for rf in sorted(refl_dir.glob("*.txt")):
            if rf.name != "evaluation.txt":
                sections.append(f"=== EXPERT ANALYSIS ({rf.stem.upper()}) ===\n{rf.read_text(encoding='utf-8').strip()}")
    return "\n\n".join(sections)

def parse_numeric(val: str):
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", str(val))
    return float(nums[0]) if nums else None

def clean_token(s: str) -> str:
    return re.sub(r"[^\w\s]", "", str(s)).strip().lower()

def run_evaluation():
    w1_records = parse_survey(DATA_DIR / "surveys_wave1.txt")
    w2_records = parse_survey(DATA_DIR / "surveys_wave2.txt")

    common_keys = [k for k in w2_records if k in w1_records]
    print(f"[*] Semantically aligned {len(common_keys)} items between Wave 1 and Wave 2 without file duplication.")

    if not common_keys:
        print("Error: No overlapping keys found.")
        return

    items_payload = {cid: w2_records[cid]["prompt"] for cid in common_keys}

    system_instruction = f"""
You are Menaka (மேனகா). Answer the follow-up survey items based strictly on your historical archive.
=== ARCHIVE ===
{assemble_agent_memory()}
=== RULES ===
Respond with a single valid JSON object mapping each key to your answer. Return ONLY raw JSON. No codeblocks.
"""

    gen_config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        response_mime_type="application/json"
    )

    prompt = f"Answer these items as Menaka:\n{json.dumps(items_payload, ensure_ascii=False, indent=2)}"
    chat = client.chats.create(model=MODEL_NAME, config=gen_config)

    agent_dict = {}
    for _ in range(3):
        try:
            res = chat.send_message(prompt)
            raw_text = res.text.strip()
            if raw_text.startswith("```"):
                raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
                raw_text = re.sub(r"\s*```$", "", raw_text)
            agent_dict = json.loads(raw_text)
            break
        except Exception as e:
            time.sleep(10)

    w1_all, w2_all, ag_all = [], [], []
    w1_nums, w2_nums, ag_nums = [], [], []

    report = ["=" * 105, " MENAKA DIGITAL TWIN: SEMANTIC HARMONIZED EVALUATION REPORT", "=" * 105]

    for idx, cid in enumerate(common_keys, 1):
        h_w1 = w1_records[cid]["val"]
        h_w2 = w2_records[cid]["val"]
        prompt_q = w2_records[cid]["prompt"]
        pred = str(agent_dict.get(cid, "N/A"))

        w1_all.append(h_w1)
        w2_all.append(h_w2)
        ag_all.append(pred)

        match_hh = clean_token(h_w1) == clean_token(h_w2)
        match_ag = clean_token(pred) == clean_token(h_w2)

        n1, n2, n_p = parse_numeric(h_w1), parse_numeric(h_w2), parse_numeric(pred)
        if n1 is not None and n2 is not None and n_p is not None:
            w1_nums.append(n1)
            w2_nums.append(n2)
            ag_nums.append(n_p)

        report.append(f"{idx:<3} | {prompt_q[:40]:<42} | {h_w1[:10]:<12} | {h_w2[:10]:<12} | {pred[:10]:<12} | {str(match_hh):<7} | {str(match_ag)}")

    acc_h = np.mean([clean_token(w1) == clean_token(w2) for w1, w2 in zip(w1_all, w2_all)]) * 100
    acc_ag = np.mean([clean_token(ag) == clean_token(w2) for ag, w2 in zip(ag_all, w2_all)]) * 100
    npa = acc_ag / acc_h if acc_h > 0 else 0.0

    report.append("\n" + "#" * 50)
    report.append(f"Human Test-Retest Agreement : {acc_h:.2f}%")
    report.append(f"Agent Predictive Accuracy   : {acc_ag:.2f}%")
    report.append(f"Normalized Accuracy (NPA)   : {npa:.3f}")
    if w1_nums and w2_nums:
        report.append(f"Human Retest MAE            : {np.mean(np.abs(np.array(w1_nums) - np.array(w2_nums))):.3f}")
        report.append(f"Agent Prediction MAE        : {np.mean(np.abs(np.array(ag_nums) - np.array(w2_nums))):.3f}")
    report.append("#" * 50)

    final_str = "\n".join(report)
    print(final_str)
    
    out_path = DATA_DIR / "Expert_reflection" / "normalized_accuracy.txt"
    out_path.write_text(final_str, encoding="utf-8")
    print(f"\n[*] Report saved to {out_path}")

if __name__ == "__main__":
    run_evaluation()