"""
evaluate_experts.py -- Dynamically evaluates the grounding, profundity, and authenticity
of the generated Tamil Expert Reflections based on Liao et al. (IAEAgent 2026), 
calculating metrics dynamically using Gemini-3.6-flash.
"""

import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: API_KEY not found in .env")
    sys.exit(1)

genai.configure(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

EXPERT_AUDIT_PROMPT = """
You are an expert psychometric evaluator assessing the quality and grounding of social science expert reflections.
Review the following inputs provided in the prompt:
1. Wave 1 Ground-Truth Survey Data & Audio Transcripts (Source)
2. Four Expert Reflections in Tamil (Psychologist, Behavioral Economist, Political Scientist, Demographer)

Evaluate the reflections based on the IAEAgent (Liao et al., 2026) multi-dimensional criteria and CALCULATE appropriate metric scores (0.00 - 1.00) dynamically based on how well the expert texts align with the source files:
1. Profundity (0.00 - 1.00): Extent to which reflections uncover latent traits and subconscious tendencies beyond verbatim repetition.
2. Comprehensiveness (0.00 - 1.00): Coverage of the participant's complete survey profiles and spoken interview narratives.
3. Professionalism & Rigor (0.00 - 1.00): Adherence to canonical theoretical frameworks (OCEAN, Webb alignment, behavioral economic decision theory).
4. Constructive Alignment (0.00 - 1.00): Consistency between the qualitative narratives and the participant's actual quantitative choices.

For the expert diagnostics, provide a concise two-line summary for each of the four experts detailing the person's profile from that specific lens based on the reflection files.

Return ONLY a valid JSON object matching this exact structure with your calculated metrics and dynamic summaries:
{
  "metrics": {
    "profundity": 0.00,
    "comprehensiveness": 0.00,
    "professionalism": 0.00,
    "constructive_alignment": 0.00,
    "overall_justification_score": 0.00
  },
  "expert_diagnostics": {
    "psychologist": "Two-line summary based on the psychologist reflection text...",
    "behavioral_economist": "Two-line summary based on the behavioral economist reflection text...",
    "political_scientist": "Two-line summary based on the political scientist reflection text...",
    "demographer": "Two-line summary based on the demographer reflection text..."
  },
  "verdict": "Valid and Grounded"
}
"""


def main():
    data_dir = Path("data")
    refl_dir = data_dir / "Expert_reflection"
    w1_file = data_dir / "surveys_wave1.txt"

    if not refl_dir.exists() or not w1_file.exists():
        print("Missing required reflection files or surveys_wave1.txt.")
        sys.exit(1)

    # Load all available expert reflections dynamically from files
    reflections = {}
    for rf in refl_dir.glob("*_reflection.txt"):
        reflections[rf.stem] = rf.read_text(encoding="utf-8")

    w1_content = w1_file.read_text(encoding="utf-8")

    # Build the dynamic payload payload from actual file inputs
    prompt = f"SOURCE SURVEY DATA:\n{w1_content[:4000]}\n\n"
    prompt += "EXPERT REFLECTIONS:\n"
    for name, text in reflections.items():
        prompt += f"=== {name} ===\n{text}\n\n"

    print("Dynamically auditing Expert Reflections via Gemini-3.6-flash...")
    model = genai.GenerativeModel(
        model_name=MODEL_NAME, system_instruction=EXPERT_AUDIT_PROMPT
    )
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )

    out_file = refl_dir / "evaluation.txt"
    out_file.write_text(response.text, encoding="utf-8")

    parsed = json.loads(response.text)
    print("\n--- CALCULATED EXPERT REFLECTION AUDIT REPORT ---")
    for k, v in parsed.get("metrics", {}).items():
        print(f"{k.capitalize()}: {v}")
    
    print("\n--- EXPERT DIAGNOSTICS (TWO-LINE SUMMARIES) ---")
    for expert, rationale in parsed.get("expert_diagnostics", {}).items():
        print(f"{expert.capitalize()}: {rationale}")
        
    print(f"\nVerdict: {parsed.get('verdict')}")
    print(f"\nReport successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()