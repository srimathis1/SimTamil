"""
evaluate_experts.py -- Evaluates the grounding, profundity, and authenticity
of the generated Tamil Expert Reflections based on Liao et al. (IAEAgent 2026).

Usage:
    python scripts/evaluate_experts.py
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
Review the following:
1. Wave 1 Ground-Truth Survey Data & Audio Transcripts (Source)
2. Four Expert Reflections in Tamil (Psychologist, Behavioral Economist, Political Scientist, Demographer)

Evaluate the reflections based on the IAEAgent (Liao et al., 2026) multi-dimensional criteria:
1. Profundity (0.00 - 1.00): Extent to which reflections uncover latent traits and subconscious tendencies beyond verbatim repetition.
2. Comprehensiveness (0.00 - 1.00): Coverage of the participant's complete survey profiles and spoken interview narratives.
3. Professionalism & Rigor (0.00 - 1.00): Adherence to canonical theoretical frameworks (OCEAN, Webb alignment, behavioral economic decision theory).
4. Constructive Alignment (0.00 - 1.00): Consistency between the qualitative narratives and the participant's actual quantitative choices.

Return ONLY a valid JSON object in this format:
{
  "metrics": {
    "profundity": 0.85,
    "comprehensiveness": 0.88,
    "professionalism": 0.91,
    "constructive_alignment": 0.87,
    "overall_justification_score": 0.88
  },
  "expert_diagnostics": {
    "psychologist": "Rationale on depth and vernacular accuracy...",
    "behavioral_economist": "Rationale on risk & time-preference grounding...",
    "political_scientist": "Rationale on institutional trust and societal view...",
    "demographer": "Rationale on living conditions and family dynamics..."
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

    # Load all 4 reflections
    reflections = {}
    for rf in refl_dir.glob("*_reflection.txt"):
        reflections[rf.stem] = rf.read_text(encoding="utf-8")

    w1_content = w1_file.read_text(encoding="utf-8")

    prompt = f"SOURCE SURVEY DATA:\n{w1_content[:4000]}\n\n"
    prompt += "EXPERT REFLECTIONS:\n"
    for name, text in reflections.items():
        prompt += f"=== {name} ===\n{text[:2000]}\n\n"

    print("Auditing Expert Reflections grounding via Gemini-3.6-flash...")
    model = genai.GenerativeModel(
        model_name=MODEL_NAME, system_instruction=EXPERT_AUDIT_PROMPT
    )
    response = model.generate_content(
        prompt, generation_config={"response_mime_type": "application/json"}
    )

    out_file = refl_dir / "evaluation.txt"
    out_file.write_text(response.text, encoding="utf-8")

    parsed = json.loads(response.text)
    print("\n--- EXPERT REFLECTION AUDIT REPORT ---")
    for k, v in parsed.get("metrics", {}).items():
        print(f"{k.capitalize()}: {v}")
    print(f"\nReport successfully saved to: {out_file}\n")


if __name__ == "__main__":
    main()