
import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# 1. CONFIGURATION
# ============================================================

PARTICIPANT_NAME = "Mounees"

BASE_DIR = Path(__file__).resolve().parent
PARTICIPANT_DIR = (
    BASE_DIR / "export" / "participants" / PARTICIPANT_NAME
)
EXPERT_DIR = PARTICIPANT_DIR / "expert_ta"
PARTICIPANT_JSON = PARTICIPANT_DIR / "all.json"

CLASSIFIER_MODEL = "gemini-3.8-flash"
PREDICTION_MODEL = "gemini-3.8-flash"


def load_env_file() -> None:
    """Load .env values without overriding variables already set in the shell."""
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def get_client():
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Add it to .env or set it in the environment."
        )
    return genai.Client(api_key=api_key)


# ============================================================
# 2. EXPERT CLASSIFICATION SCHEMA
# ============================================================

EXPERTS = [
    "Demographer",
    "Psychologist",
    "Behavioral Economist",
    "Regional Political Scientist",
    "Sociologist & Cultural Anthropologist",
]

EXPERT_FILES = {
    "Demographer": "demographer.txt",
    "Psychologist": "psychologist.txt",
    "Behavioral Economist": "behavioral_economist.txt",
    "Regional Political Scientist": "political_scientist.txt",
    "Sociologist & Cultural Anthropologist": "sociologist.txt",
}


class ExpertClassification(BaseModel):
    selected_expert: str = Field(
        description=(
            "Exactly one of: Demographer, Psychologist, "
            "Behavioral Economist, Regional Political Scientist, "
            "Sociologist & Cultural Anthropologist"
        )
    )
    reasoning: str = Field(
        description="One concise sentence explaining the domain choice."
    )


# ============================================================
# 3. LOAD PARTICIPANT DATA
# ============================================================

def load_participant_data() -> dict:
    """Load the fixed participant's all.json file."""

    if not PARTICIPANT_JSON.exists():
        raise FileNotFoundError(
            f"Participant data not found: {PARTICIPANT_JSON}"
        )

    with PARTICIPANT_JSON.open("r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"[Data] Loaded participant: {PARTICIPANT_NAME}")
    print(f"[Data] Source: {PARTICIPANT_JSON}")

    return data


# ============================================================
# 4. CLASSIFY THE TARGET QUESTION
# ============================================================

def classify_target_question(question: str) -> ExpertClassification:
    """Choose the most relevant expert domain for a question."""
    client = get_client()

    system_instruction = """
You are a domain router for a Tamil population digital-twin
simulation research project.

Select the single most relevant expert for predicting an individual's
response to the target question.

Available experts:

1. Demographer:
Age, education, income, occupation, marital status, household structure.

2. Psychologist:
Personality, BFI traits, emotions, coping, stress, self-esteem.

3. Behavioral Economist:
Financial trade-offs, risk, savings, spending, economic games,
trust, cooperation and resource allocation.

4. Regional Political Scientist:
Civic participation, government welfare schemes, regional identity,
public policy and political attitudes.

5. Sociologist & Cultural Anthropologist:
Family obligations, elders, marriage decisions, kinship, community
reputation, cultural expectations and social norms.

Choose based on the central issue in the question, not merely a
keyword. Return exactly one permitted expert name.
"""

    response = client.models.generate_content(
        model=CLASSIFIER_MODEL,
        contents=f"Target question:\n{question}",
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ExpertClassification,
            temperature=0,
        ),
    )

    if not response.text:
        raise RuntimeError("The expert classifier returned no response.")

    result = ExpertClassification.model_validate_json(response.text)

    if result.selected_expert not in EXPERTS:
        raise ValueError(
            f"Unexpected expert classification: {result.selected_expert}"
        )

    print(f"\n[Router] Selected expert: {result.selected_expert}")
    print(f"[Router] Reason: {result.reasoning}")

    return result


# ============================================================
# 5. LOAD ONLY THE SELECTED EXPERT REFLECTION
# ============================================================

def load_expert_reflection(expert_name: str) -> str:
    """Load the selected expert's .txt file from expert_ta only."""

    filename = EXPERT_FILES[expert_name]
    reflection_path = EXPERT_DIR / filename

    if not reflection_path.is_file():
        raise FileNotFoundError(
            f"Required expert reflection not found: {reflection_path}"
        )

    reflection = reflection_path.read_text(encoding="utf-8").strip()

    if not reflection:
        raise ValueError(
            f"Expert reflection file is empty: {reflection_path}"
        )

    print(f"[Expert] Loaded: {reflection_path}")

    return reflection


# ============================================================
# 6. PREDICT PARTICIPANT'S RESPONSE
# ============================================================

def predict_response(target_question: str) -> dict:
    """Predict Mouneeswaran's response to a target question."""

    target_question = target_question.strip()

    if not target_question:
        raise ValueError("Target question cannot be empty.")

    client = get_client()

    # A. Load the fixed participant's complete JSON data.
    participant_data = load_participant_data()

    # B. Classify the question.
    classification = classify_target_question(target_question)

    # C. Load ONLY the selected expert file from expert_ta.
    expert_reflection = load_expert_reflection(
        classification.selected_expert
    )

    # D. Construct the prediction prompt.
    system_instruction = """
You simulate a specific research participant using their provided
self-reported data. Predict the participant's likely response to the
target question.

Rules:
- Ground the prediction in the supplied participant data and expert notes.
- Treat survey responses and interview statements as evidence, not proof
  of how the participant will behave.
- Do not invent personal facts or unsupported biographical details.
- If evidence is insufficient or conflicting, acknowledge uncertainty.
- Follow the answer scale and response format specified by the question.
- Write the predicted answer, rationale, and uncertainty note only in Tamil.
- Use Tamil script for the response; do not answer in English.
- Keep unavoidable names, numbers, scales, and technical labels as written
  when they cannot be naturally translated.
- Give a concise explanation citing the relevant evidence.
- Do not reveal private chain-of-thought. Provide only the final prediction
  and a short evidence-based rationale.
- The expert reflection is contextual guidance, not a source of new facts
  about the participant.
"""

    user_prompt = f"""
PARTICIPANT
Name: {PARTICIPANT_NAME}

PARTICIPANT DATA FROM all.json
{json.dumps(participant_data, ensure_ascii=False, indent=2)}

SELECTED EXPERT
{classification.selected_expert}

EXPERT REFLECTION FROM expert_ta
{expert_reflection}

TARGET QUESTION
{target_question}

Return:
1. The predicted answer in the format requested by the question, written only in Tamil.
2. A concise evidence-based rationale written only in Tamil.
3. An uncertainty note in Tamil if the participant data is insufficient.

If the question requests a numerical scale, return a score within
the specified range and explain it briefly in Tamil.
"""

    # E. Generate the prediction.
    response = client.models.generate_content(
        model=PREDICTION_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )

    if not response.text:
        raise RuntimeError("The prediction model returned no response.")

    output = {
        "participant": PARTICIPANT_NAME,
        "target_question": target_question,
        "selected_expert": classification.selected_expert,
        "classification_reason": classification.reasoning,
        "prediction": response.text.strip(),
    }

    return output


# ============================================================
# 7. INTERACTIVE EXECUTION
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIMTAMIL - PARTICIPANT DIGITAL TWIN")
    print(f"Fixed participant: {PARTICIPANT_NAME}")
    print("=" * 60)

    try:
        question = input("\nEnter your target question: ").strip()

        result = predict_response(question)

        print("\n" + "=" * 60)
        print("PREDICTION RESULT")
        print("=" * 60)
        print(json.dumps(result, ensure_ascii=False, indent=2))

        # Save the latest prediction.
        output_path = BASE_DIR / "prediction_result.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"\n[Saved] {output_path}")

    except (FileNotFoundError, ValueError, EnvironmentError) as exc:
        print(f"\n[Error] {exc}")

    except Exception as exc:
        print(f"\n[Error] Prediction failed: {exc}")
