import json
import math
import re
import time
from pathlib import Path
from statistics import mean

from google import genai
from google.genai import types
from dotenv import load_dotenv


# ============================================================
# SimTamil - Wave-2 Agent Evaluation
# ============================================================
#
# Participant 01 ONLY
#
# Wave-1:
#   Used to build agent memory
#
# Wave-2:
#   NEVER sent to Gemini
#   Used only after prediction for evaluation
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRANSCRIPTS_DIR = DATA_DIR / "Transcripts"
EXPERT_DIR = DATA_DIR / "Expert_reflection"

WAVE1_FILE = DATA_DIR / "surveys_wave1.txt"
WAVE2_FILE = DATA_DIR / "surveys_wave2.txt"

PREDICTION_FILE = (
    EXPERT_DIR / "wave2_first20_predictions.json"
)

FINAL_REPORT = (
    EXPERT_DIR / "final_evaluation.txt"
)


# ============================================================
# GEMINI SETTINGS
# ============================================================

MODEL_NAME = "gemini-2.5-flash"
# We evaluate only first 20 matched Wave-2 questions
TOTAL_QUESTIONS = 20

# 4 questions per Gemini request
BATCH_SIZE = 4

# Therefore:
# 20 / 4 = 5 API requests
TOTAL_REQUESTS = math.ceil(
    TOTAL_QUESTIONS / BATCH_SIZE
)

REQUEST_DELAY = 8

MAX_RETRIES = 4

INITIAL_RETRY_WAIT = 15

MAX_RETRY_WAIT = 120


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv(PROJECT_ROOT / ".env")


API_KEY = (
    __import__("os").getenv("GEMINI_API_KEY")
    or __import__("os").getenv("API_KEY")
)


if not API_KEY:

    raise RuntimeError(
        "\nGemini API key not found.\n"
        "Add this to your .env file:\n\n"
        "GEMINI_API_KEY=YOUR_NEW_API_KEY\n"
    )


# ============================================================
# CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# BFI QUESTION INFORMATION
# ============================================================

BFI_TRAIT_MAP = {

    # Extraversion
    "BFI-EXT-1": "Extraversion",
    "BFI-EXT-2": "Extraversion",
    "BFI-EXT-3": "Extraversion",
    "BFI-EXT-4": "Extraversion",

    # Agreeableness
    "BFI-AGR-1": "Agreeableness",
    "BFI-AGR-2": "Agreeableness",
    "BFI-AGR-3": "Agreeableness",
    "BFI-AGR-4": "Agreeableness",

    # Conscientiousness
    "BFI-CON-1": "Conscientiousness",
    "BFI-CON-2": "Conscientiousness",
    "BFI-CON-3": "Conscientiousness",
    "BFI-CON-4": "Conscientiousness",

    # Neuroticism
    "BFI-NEU-1": "Neuroticism",
    "BFI-NEU-2": "Neuroticism",
    "BFI-NEU-3": "Neuroticism",
    "BFI-NEU-4": "Neuroticism",

    # Openness
    "BFI-OPE-1": "Openness",
    "BFI-OPE-2": "Openness",
    "BFI-OPE-3": "Openness",
    "BFI-OPE-4": "Openness",
}


# ============================================================
# HELPERS
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    return str(text).strip()


def normalize_answer(answer):

    if answer is None:
        return None

    value = str(answer).strip().lower()

    value = value.replace(".", "")
    value = value.replace("。", "")

    return value


def safe_float(value):

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


# ============================================================
# PARSE SURVEY FILE
# ============================================================

def parse_survey(file_path):

    if not file_path.exists():

        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    text = file_path.read_text(
        encoding="utf-8"
    )

    records = {}

    current_id = None
    current_question = None
    current_response = None

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        # ----------------------------------------------------
        # Question
        # ----------------------------------------------------

        match = re.match(
            r"Question\s+\[([^\]]+)\](?:\s+\[[^\]]+\])?(?:\(R\))?:\s*(.*)",
            line,
            re.IGNORECASE
        )

        if match:

            # Save previous question
            if current_id:

                records[current_id] = {
                    "question": current_question or "",
                    "response": current_response
                }

            current_id = match.group(1).strip()

            current_question = match.group(2).strip()

            current_response = None

            continue

        # ----------------------------------------------------
        # Participant response
        # ----------------------------------------------------

        if line.lower().startswith(
            "participant response:"
        ):

            response = line.split(
                ":",
                1
            )[1].strip()

            current_response = response

    # Save last record
    if current_id:

        records[current_id] = {
            "question": current_question or "",
            "response": current_response
        }

    return records


# ============================================================
# EXTRACT WAVE-2 SCORED BFI VALUE
# ============================================================

def extract_wave2_scored(response):

    if not response:
        return None

    # Example:
    #
    # Raw = 4, Scored = 2
    #

    match = re.search(
        r"Scored\s*=\s*(-?\d+(?:\.\d+)?)",
        response,
        re.IGNORECASE
    )

    if match:

        return float(
            match.group(1)
        )

    # Fallback
    number_match = re.search(
        r"-?\d+(?:\.\d+)?",
        response
    )

    if number_match:

        return float(
            number_match.group()
        )

    return None


# ============================================================
# EXTRACT WAVE-1 RAW VALUE
# ============================================================

def extract_wave1_value(response):

    if response is None:

        return None

    match = re.search(
        r"-?\d+(?:\.\d+)?",
        str(response)
    )

    if match:

        return float(
            match.group()
        )

    return None


# ============================================================
# DETERMINE REVERSE SCORING
# ============================================================

def is_reverse_question(question):

    if not question:

        return False

    return "(R)" in question


# ============================================================
# CALCULATE WAVE-1 SCORED BFI
# ============================================================

def calculate_wave1_bfi_score(
    question,
    response
):

    value = extract_wave1_value(
        response
    )

    if value is None:

        return None

    if is_reverse_question(question):

        return 6 - value

    return value


# ============================================================
# LOAD TRANSCRIPTS
# ============================================================

def load_transcripts():

    print("\nLoading merged transcripts...")

    if not TRANSCRIPTS_DIR.exists():

        raise FileNotFoundError(
            f"Transcript directory not found:\n"
            f"{TRANSCRIPTS_DIR}"
        )

    transcript_files = sorted(
        TRANSCRIPTS_DIR.glob("*.txt")
    )

    # Ignore chunk directory files
    transcript_files = [
        f for f in transcript_files
        if f.is_file()
    ]

    print(
        f"Merged transcript files found: "
        f"{len(transcript_files)}"
    )

    parts = []

    for file in transcript_files:

        try:

            content = file.read_text(
                encoding="utf-8"
            )

            if content.strip():

                parts.append(
                    f"\n===== {file.name} =====\n"
                    f"{content}"
                )

        except Exception as e:

            print(
                f"Warning: could not read "
                f"{file.name}: {e}"
            )

    transcript_text = "\n".join(
        parts
    )

    print(
        f"Transcript characters used: "
        f"{len(transcript_text):,}"
    )

    return transcript_text


# ============================================================
# LOAD EXPERT REFLECTIONS
# ============================================================

def load_expert_reflections():

    print(
        "\nLoading expert reflections..."
    )

    expert_files = [

        "psychologist_reflection.txt",

        "behavioral_economist_reflection.txt",

        "political_scientist_reflection.txt",

        "demographer_reflection.txt",

        "big_five_scores.txt",
    ]

    parts = []

    for filename in expert_files:

        path = EXPERT_DIR / filename

        if not path.exists():

            print(
                f"Warning: missing {filename}"
            )

            continue

        content = path.read_text(
            encoding="utf-8"
        )

        parts.append(
            f"\n===== {filename} =====\n"
            f"{content}"
        )

    expert_text = "\n".join(
        parts
    )

    print(
        f"Expert files loaded: "
        f"{len(parts)}"
    )

    print(
        f"Expert context characters: "
        f"{len(expert_text):,}"
    )

    return expert_text


# ============================================================
# BUILD AGENT MEMORY
# ============================================================

def build_agent_memory():

    print(
        "\n[4/7] Building Wave-1-only agent memory..."
    )

    # --------------------------------------------------------
    # Wave-1 survey
    # --------------------------------------------------------

    if not WAVE1_FILE.exists():

        raise FileNotFoundError(
            f"Wave-1 file not found:\n"
            f"{WAVE1_FILE}"
        )

    wave1_text = WAVE1_FILE.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Transcripts
    # --------------------------------------------------------

    transcript_text = load_transcripts()

    # --------------------------------------------------------
    # Expert reflections
    # --------------------------------------------------------

    expert_text = load_expert_reflections()

    memory = f"""
============================================================
SIMTAMIL PARTICIPANT MEMORY
============================================================

IMPORTANT:
This memory contains Wave-1 information only.

The model must answer as the participant based on:
1. Wave-1 survey responses
2. Interview transcripts
3. Expert reflections
4. Big Five analysis

Wave-2 responses are NOT included.

============================================================
WAVE-1 SURVEY
============================================================

{wave1_text}

============================================================
INTERVIEW TRANSCRIPTS
============================================================

{transcript_text}

============================================================
EXPERT REFLECTIONS
============================================================

{expert_text}

============================================================
END OF WAVE-1 MEMORY
============================================================
"""

    print(
        f"\nTotal memory characters: "
        f"{len(memory):,}"
    )

    return memory


# ============================================================
# GET FIRST 20 MATCHED QUESTIONS
# ============================================================

def get_first_20_questions(
    wave1_records,
    wave2_records
):

    matched = []

    # Preserve Wave-2 order
    for question_id, data in wave2_records.items():

        if question_id not in wave1_records:

            continue

        # Only BFI questions
        if not question_id.startswith(
            "BFI-"
        ):

            continue

        human_score = extract_wave2_scored(
            data.get("response")
        )

        if human_score is None:

            continue

        matched.append({

            "id": question_id,

            "question": data.get(
                "question",
                ""
            ),

            "wave1_response":
                wave1_records[
                    question_id
                ].get(
                    "response"
                ),

            "wave1_score":
                calculate_wave1_bfi_score(
                    wave1_records[
                        question_id
                    ].get("question", ""),
                    wave1_records[
                        question_id
                    ].get("response")
                ),

            "wave2_score":
                human_score,

            "trait":
                BFI_TRAIT_MAP.get(
                    question_id,
                    "Unknown"
                )
        })

        if len(matched) >= TOTAL_QUESTIONS:

            break

    return matched


# ============================================================
# DISPLAY QUESTIONS
# ============================================================

def display_questions(
    questions
):

    print(
        "\n[3/7] Selected Wave-2 questions:"
    )

    print(
        "================================================"
    )

    for index, item in enumerate(
        questions,
        start=1
    ):

        print(
            f"{index:02d}. "
            f"{item['id']} | "
            f"{item['trait']}"
        )

    print(
        "================================================"
    )


# ============================================================
# GEMINI JSON SCHEMA
# ============================================================

RESPONSE_SCHEMA = {

    "type": "array",

    "items": {

        "type": "object",

        "properties": {

            "id": {
                "type": "string"
            },

            "prediction": {
                "type": "integer"
            }

        },

        "required": [
            "id",
            "prediction"
        ]
    }
}


# ============================================================
# CREATE PROMPT
# ============================================================

def create_prompt(
    memory,
    questions
):

    question_block = []

    for item in questions:

        question_block.append(
            f"""
Question ID: {item['id']}
Trait: {item['trait']}
Question: {item['question']}
"""
        )

    questions_text = "\n".join(
        question_block
    )

    prompt = f"""
You are simulating one specific Tamil participant.

Your task is to predict how this participant would answer
the provided BFI questions.

Use ONLY the participant's Wave-1 memory below.

The memory contains:
- Wave-1 survey responses
- Interview transcript
- Expert reflections
- Big Five information

DO NOT use any Wave-2 information.

The Wave-2 answers are intentionally hidden from you.

============================================================
IMPORTANT
============================================================

For every BFI question, predict the participant's
SCORED response.

Valid prediction values are integers from 1 to 5.

Use the participant's own personality, experiences,
attitudes, habits, family context, social context,
and statements from the interview.

Do not answer based on what an average person would say.

Do not explain your reasoning.

Return ONLY the requested structured JSON output.

============================================================
WAVE-1 PARTICIPANT MEMORY
============================================================

{memory}

============================================================
QUESTIONS TO PREDICT
============================================================

{questions_text}

============================================================
OUTPUT
============================================================

Return exactly one prediction for every question.

Each prediction must contain:

"id": question ID

"prediction": integer from 1 to 5

Do not omit any question.
Do not add explanations.
"""

    return prompt


# ============================================================
# VALIDATE GEMINI PREDICTIONS
# ============================================================

def validate_predictions(
    predictions,
    expected_questions
):

    if not isinstance(
        predictions,
        list
    ):

        raise ValueError(
            "Gemini output is not a JSON list."
        )

    expected_ids = [
        q["id"]
        for q in expected_questions
    ]

    result = {}

    for item in predictions:

        if not isinstance(
            item,
            dict
        ):

            continue

        question_id = item.get(
            "id"
        )

        prediction = item.get(
            "prediction"
        )

        if question_id not in expected_ids:

            continue

        try:

            prediction = int(
                prediction
            )

        except (
            ValueError,
            TypeError
        ):

            continue

        if prediction < 1 or prediction > 5:

            continue

        result[
            question_id
        ] = prediction

    missing = [
        qid
        for qid in expected_ids
        if qid not in result
    ]

    if missing:

        raise ValueError(
            "Missing predictions for: "
            + ", ".join(missing)
        )

    return result


# ============================================================
# CHECK QUOTA ERROR
# ============================================================

def is_quota_error(error):

    text = str(error).lower()

    keywords = [

        "resource_exhausted",

        "quota",

        "rate limit",

        "rate_limit",

        "too many requests",

        "429",
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# CHECK TEMPORARY ERROR
# ============================================================

def is_temporary_error(error):

    text = str(error).lower()

    keywords = [

        "503",

        "500",

        "502",

        "504",

        "service unavailable",

        "deadline exceeded",

        "timeout",

        "temporarily unavailable",
    ]

    return any(
        keyword in text
        for keyword in keywords
    )


# ============================================================
# CALL GEMINI
# ============================================================

def generate_batch(
    memory,
    questions,
    batch_number
):

    prompt = create_prompt(
        memory,
        questions
    )

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        print(
            f"\nGemini batch "
            f"{batch_number}: "
            f"attempt {attempt}/{MAX_RETRIES}"
        )

        try:

            response = client.models.generate_content(

                model=MODEL_NAME,

                contents=prompt,

                config=types.GenerateContentConfig(

                    temperature=0,

                    response_mime_type=(
                        "application/json"
                    ),

                    response_schema=(
                        RESPONSE_SCHEMA
                    ),
                )
            )

            # ------------------------------------------------
            # Debug: show raw response if needed
            # ------------------------------------------------

            raw_text = response.text

            if not raw_text:

                raise ValueError(
                    "Gemini returned empty response."
                )

            # ------------------------------------------------
            # Parse JSON
            # ------------------------------------------------

            try:

                parsed = json.loads(
                    raw_text
                )

            except json.JSONDecodeError as e:

                print(
                    "\nGemini returned invalid JSON."
                )

                print(
                    "\nRaw Gemini response:"
                )

                print(
                    raw_text[:3000]
                )

                raise ValueError(
                    f"Invalid JSON: {e}"
                )

            # ------------------------------------------------
            # Validate
            # ------------------------------------------------

            predictions = validate_predictions(
                parsed,
                questions
            )

            print(
                f"✓ Batch {batch_number} "
                f"completed successfully."
            )

            return predictions

        except Exception as e:

            print(
                f"\nGemini error:\n{e}"
            )

            # ------------------------------------------------
            # Quota error
            # ------------------------------------------------

            if is_quota_error(e):

                raise RuntimeError(
                    "\nGemini quota/rate limit "
                    "was reached.\n\n"
                    "Your new project/API key is "
                    "being used, but its available "
                    "quota has been exhausted."
                )

            # ------------------------------------------------
            # Temporary error
            # ------------------------------------------------

            if is_temporary_error(e):

                if attempt < MAX_RETRIES:

                    wait_time = min(
                        INITIAL_RETRY_WAIT
                        * (2 ** (attempt - 1)),
                        MAX_RETRY_WAIT
                    )

                    print(
                        f"\nTemporary Gemini error."
                    )

                    print(
                        f"Waiting "
                        f"{wait_time} seconds..."
                    )

                    time.sleep(
                        wait_time
                    )

                    continue

            # ------------------------------------------------
            # JSON error
            # ------------------------------------------------

            if attempt < MAX_RETRIES:

                print(
                    "\nTrying the same batch "
                    "once more..."
                )

                time.sleep(
                    REQUEST_DELAY
                )

                continue

            raise


# ============================================================
# LOAD EXISTING CHECKPOINT
# ============================================================

def load_checkpoint():

    if not PREDICTION_FILE.exists():

        return {}

    try:

        data = json.loads(
            PREDICTION_FILE.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            data,
            dict
        ):

            return data

    except Exception:

        print(
            "Warning: checkpoint could "
            "not be loaded."
        )

    return {}


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    predictions
):

    EXPERT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PREDICTION_FILE.write_text(

        json.dumps(
            predictions,
            indent=4,
            ensure_ascii=False
        ),

        encoding="utf-8"
    )

    print(
        f"✓ Checkpoint saved:\n"
        f"{PREDICTION_FILE}"
    )


# ============================================================
# CALCULATE PEARSON
# ============================================================

def pearson_correlation(
    x,
    y
):

    if len(x) < 2:

        return None

    mean_x = mean(x)
    mean_y = mean(y)

    numerator = sum(
        (a - mean_x)
        * (b - mean_y)
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (a - mean_x) ** 2
            for a in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - mean_y) ** 2
            for b in y
        )
    )

    denominator = (
        denominator_x
        * denominator_y
    )

    if denominator == 0:

        return None

    return numerator / denominator


# ============================================================
# CALCULATE RMSE
# ============================================================

def calculate_rmse(
    errors
):

    if not errors:

        return None

    return math.sqrt(
        mean(
            error ** 2
            for error in errors
        )
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def calculate_metrics(
    questions,
    predictions
):

    agent_values = []
    human_values = []
    wave1_values = []

    rows = []

    for item in questions:

        question_id = item["id"]

        if question_id not in predictions:

            continue

        agent = float(
            predictions[
                question_id
            ]
        )

        human = float(
            item["wave2_score"]
        )

        wave1 = item.get(
            "wave1_score"
        )

        agent_values.append(
            agent
        )

        human_values.append(
            human
        )

        if wave1 is not None:

            wave1_values.append(
                (
                    float(wave1),
                    human
                )
            )

        error = abs(
            agent - human
        )

        rows.append({

            "id":
                question_id,

            "trait":
                item["trait"],

            "agent":
                agent,

            "human_wave2":
                human,

            "wave1":
                wave1,

            "absolute_error":
                error,

            "match":
                agent == human
        })

    # --------------------------------------------------------
    # Agent agreement
    # --------------------------------------------------------

    total = len(rows)

    agent_correct = sum(
        1
        for row in rows
        if row["match"]
    )

    agent_agreement = (

        agent_correct / total * 100

        if total > 0

        else None
    )

    # --------------------------------------------------------
    # Human test-retest agreement
    # --------------------------------------------------------

    human_correct = 0

    for wave1, wave2 in wave1_values:

        if wave1 == wave2:

            human_correct += 1

    human_total = len(
        wave1_values
    )

    human_agreement = (

        human_correct
        / human_total
        * 100

        if human_total > 0

        else None
    )

    # --------------------------------------------------------
    # Normalized Prediction Accuracy
    # --------------------------------------------------------

    if (
        agent_agreement is not None
        and human_agreement is not None
        and human_agreement > 0
    ):

        npa = (
            agent_agreement
            / human_agreement
        )

    else:

        npa = None

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    errors = [
        row["absolute_error"]
        for row in rows
    ]

    mae = (

        mean(errors)

        if errors

        else None
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = calculate_rmse(
        errors
    )

    # --------------------------------------------------------
    # Pearson
    # --------------------------------------------------------

    pearson = pearson_correlation(
        agent_values,
        human_values
    )

    # --------------------------------------------------------
    # Trait-level MAE
    # --------------------------------------------------------

    trait_errors = {}

    for row in rows:

        trait = row["trait"]

        trait_errors.setdefault(
            trait,
            []
        )

        trait_errors[
            trait
        ].append(
            row["absolute_error"]
        )

    trait_mae = {}

    for trait, values in (
        trait_errors.items()
    ):

        trait_mae[trait] = mean(
            values
        )

    return {

        "rows": rows,

        "total_questions":
            total,

        "agent_correct":
            agent_correct,

        "agent_agreement":
            agent_agreement,

        "human_test_retest_correct":
            human_correct,

        "human_test_retest_total":
            human_total,

        "human_test_retest_agreement":
            human_agreement,

        "normalized_prediction_accuracy":
            npa,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "Pearson":
            pearson,

        "trait_MAE":
            trait_mae,
    }


# ============================================================
# SAVE FINAL REPORT
# ============================================================

def save_final_report(
    metrics
):

    FINAL_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        FINAL_REPORT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "================================================\n"
        )

        f.write(
            "       SimTamil Agent Evaluation\n"
        )

        f.write(
            "       Participant 01 - Wave-2\n"
        )

        f.write(
            "================================================\n\n"
        )

        # ----------------------------------------------------
        # Configuration
        # ----------------------------------------------------

        f.write(
            "1. EVALUATION CONFIGURATION\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            f"Model: {MODEL_NAME}\n"
        )

        f.write(
            f"Questions evaluated: "
            f"{metrics['total_questions']}\n"
        )

        f.write(
            "Agent memory: Wave-1 only\n"
        )

        f.write(
            "Wave-2 answers hidden during prediction\n"
        )

        f.write(
            f"Gemini batch size: "
            f"{BATCH_SIZE}\n"
        )

        f.write(
            f"Gemini requests: "
            f"{TOTAL_REQUESTS}\n\n"
        )

        # ----------------------------------------------------
        # Agent agreement
        # ----------------------------------------------------

        f.write(
            "2. WAVE-2 AGENT AGREEMENT\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            "Metric: Exact-Match Agreement\n\n"
        )

        f.write(
            f"Correct predictions: "
            f"{metrics['agent_correct']}\n"
        )

        f.write(
            f"Total questions: "
            f"{metrics['total_questions']}\n"
        )

        f.write(
            f"Agent Agreement: "
            f"{metrics['agent_agreement']:.2f}%\n\n"
            if metrics["agent_agreement"] is not None
            else
            "Agent Agreement: N/A\n\n"
        )

        # ----------------------------------------------------
        # Human test-retest
        # ----------------------------------------------------

        f.write(
            "3. HUMAN TEST-RETEST AGREEMENT\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            f"Correct Wave-1 → Wave-2: "
            f"{metrics['human_test_retest_correct']}\n"
        )

        f.write(
            f"Comparable items: "
            f"{metrics['human_test_retest_total']}\n"
        )

        f.write(
            f"Human Test-Retest Agreement: "
            f"{metrics['human_test_retest_agreement']:.2f}%\n\n"
            if metrics["human_test_retest_agreement"] is not None
            else
            "Human Test-Retest Agreement: N/A\n\n"
        )

        # ----------------------------------------------------
        # NPA
        # ----------------------------------------------------

        f.write(
            "4. NORMALIZED PREDICTION ACCURACY\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            "Formula:\n"
        )

        f.write(
            "NPA = Agent Agreement / Human Test-Retest Agreement\n\n"
        )

        if metrics[
            "normalized_prediction_accuracy"
        ] is not None:

            f.write(
                f"NPA: "
                f"{metrics['normalized_prediction_accuracy']:.4f}\n"
            )

            f.write(
                f"NPA (% of human consistency): "
                f"{metrics['normalized_prediction_accuracy'] * 100:.2f}%\n\n"
            )

        else:

            f.write(
                "NPA: N/A\n\n"
            )

        # ----------------------------------------------------
        # Error metrics
        # ----------------------------------------------------

        f.write(
            "5. BFI PREDICTION ERROR\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            "Mean Absolute Error (MAE): "
        )

        if metrics["MAE"] is not None:

            f.write(
                f"{metrics['MAE']:.4f}\n"
            )

        else:

            f.write(
                "N/A\n"
            )

        f.write(
            "Root Mean Square Error (RMSE): "
        )

        if metrics["RMSE"] is not None:

            f.write(
                f"{metrics['RMSE']:.4f}\n"
            )

        else:

            f.write(
                "N/A\n"
            )

        f.write(
            "Pearson Correlation: "
        )

        if metrics["Pearson"] is not None:

            f.write(
                f"{metrics['Pearson']:.4f}\n\n"
            )

        else:

            f.write(
                "N/A\n\n"
            )

        # ----------------------------------------------------
        # Trait-wise
        # ----------------------------------------------------

        f.write(
            "6. TRAIT-WISE MAE\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        for trait, value in (
            metrics["trait_MAE"].items()
        ):

            f.write(
                f"{trait:<20}: "
                f"{value:.4f}\n"
            )

        # ----------------------------------------------------
        # Question-level results
        # ----------------------------------------------------

        f.write(
            "\n7. QUESTION-LEVEL RESULTS\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        for row in metrics["rows"]:

            f.write(
                f"\n{row['id']}\n"
            )

            f.write(
                f"Trait: "
                f"{row['trait']}\n"
            )

            f.write(
                f"Wave-1: "
                f"{row['wave1']}\n"
            )

            f.write(
                f"Agent Prediction: "
                f"{row['agent']}\n"
            )

            f.write(
                f"Human Wave-2: "
                f"{row['human_wave2']}\n"
            )

            f.write(
                f"Absolute Error: "
                f"{row['absolute_error']:.2f}\n"
            )

            f.write(
                f"Exact Match: "
                f"{row['match']}\n"
            )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        f.write(
            "\n\n8. METHODOLOGY\n"
        )

        f.write(
            "------------------------------------------------\n"
        )

        f.write(
            "The agent was constructed using Wave-1 "
            "participant information, including interview "
            "transcripts, survey responses, expert reflections "
            "and Big Five information. Wave-2 responses were "
            "hidden from the model during prediction and were "
            "used only for post-hoc evaluation.\n\n"
        )

        f.write(
            "Agent Agreement measures the proportion of "
            "Wave-2 responses exactly reproduced by the agent.\n\n"
        )

        f.write(
            "Human Test-Retest Agreement measures the "
            "participant's own consistency between Wave-1 "
            "and Wave-2.\n\n"
        )

        f.write(
            "Normalized Prediction Accuracy compares agent "
            "agreement with the participant's own "
            "test-retest agreement.\n\n"
        )

        f.write(
            "================================================\n"
        )

        f.write(
            "                 COMPLETE\n"
        )

        f.write(
            "================================================\n"
        )

    print(
        f"\n✓ Final report saved:\n"
        f"{FINAL_REPORT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n================================================"
    )

    print(
        "       SimTamil Wave-2 Evaluation"
    )

    print(
        "       Participant 01"
    )

    print(
        "================================================"
    )

    print(
        f"\nGemini model: {MODEL_NAME}"
    )

    print(
        f"Questions: {TOTAL_QUESTIONS}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"API requests required: {TOTAL_REQUESTS}"
    )

    # ========================================================
    # 1. PARSE WAVE-1
    # ========================================================

    print(
        "\n[1/7] Loading Wave-1..."
    )

    wave1_records = parse_survey(
        WAVE1_FILE
    )

    print(
        f"Wave-1 records parsed: "
        f"{len(wave1_records)}"
    )

    # ========================================================
    # 2. PARSE WAVE-2
    # ========================================================

    print(
        "\n[2/7] Loading Wave-2..."
    )

    wave2_records = parse_survey(
        WAVE2_FILE
    )

    print(
        f"Wave-2 records parsed: "
        f"{len(wave2_records)}"
    )

    # ========================================================
    # 3. SELECT FIRST 20
    # ========================================================

    questions = get_first_20_questions(
        wave1_records,
        wave2_records
    )

    if not questions:

        raise RuntimeError(
            "No matched BFI Wave-2 questions found."
        )

    print(
        f"\nMatched BFI questions selected: "
        f"{len(questions)}"
    )

    display_questions(
        questions
    )

    # ========================================================
    # 4. BUILD MEMORY
    # ========================================================

    memory = build_agent_memory()

    print(
        "\nDATA LEAKAGE CHECK"
    )

    print(
        "Wave-2 answers are NOT included "
        "in the Gemini memory."
    )

    # ========================================================
    # 5. GENERATE PREDICTIONS
    # ========================================================

    print(
        "\n[5/7] Generating predictions..."
    )

    predictions = load_checkpoint()

    print(
        f"\nCompleted already: "
        f"{len(predictions)}/{len(questions)}"
    )

    remaining = [

        item
        for item in questions

        if item["id"] not in predictions
    ]

    print(
        f"Remaining: "
        f"{len(remaining)}"
    )

    if remaining:

        print(
            f"\nGemini requests required: "
            f"{math.ceil(len(remaining) / BATCH_SIZE)}"
        )

        print(
            f"Questions per request: "
            f"{BATCH_SIZE}"
        )

    batch_number = 0

    while remaining:

        batch_number += 1

        batch = remaining[
            :BATCH_SIZE
        ]

        print(
            f"\n================================================"
        )

        print(
            f"BATCH {batch_number}"
        )

        print(
            "Questions:"
        )

        for item in batch:

            print(
                f"  {item['id']}"
            )

        print(
            "================================================"
        )

        batch_predictions = generate_batch(
            memory,
            batch,
            batch_number
        )

        predictions.update(
            batch_predictions
        )

        save_checkpoint(
            predictions
        )

        remaining = [

            item
            for item in questions

            if item["id"] not in predictions
        ]

        if remaining:

            print(
                f"\nWaiting "
                f"{REQUEST_DELAY} seconds "
                f"before next batch..."
            )

            time.sleep(
                REQUEST_DELAY
            )

    print(
        "\n✓ All predictions completed."
    )

    # ========================================================
    # 6. CALCULATE METRICS
    # ========================================================

    print(
        "\n[6/7] Calculating evaluation metrics..."
    )

    metrics = calculate_metrics(
        questions,
        predictions
    )

    print(
        "\n================================================"
    )

    print(
        "RESULTS"
    )

    print(
        "================================================"
    )

    print(
        f"Agent Agreement: "
        f"{metrics['agent_agreement']:.2f}%"
        if metrics["agent_agreement"] is not None
        else
        "Agent Agreement: N/A"
    )

    print(
        f"Human Test-Retest: "
        f"{metrics['human_test_retest_agreement']:.2f}%"
        if metrics["human_test_retest_agreement"] is not None
        else
        "Human Test-Retest: N/A"
    )

    print(
        f"NPA: "
        f"{metrics['normalized_prediction_accuracy']:.4f}"
        if metrics[
            "normalized_prediction_accuracy"
        ] is not None
        else
        "NPA: N/A"
    )

    print(
        f"MAE: "
        f"{metrics['MAE']:.4f}"
        if metrics["MAE"] is not None
        else
        "MAE: N/A"
    )

    print(
        f"RMSE: "
        f"{metrics['RMSE']:.4f}"
        if metrics["RMSE"] is not None
        else
        "RMSE: N/A"
    )

    print(
        f"Pearson: "
        f"{metrics['Pearson']:.4f}"
        if metrics["Pearson"] is not None
        else
        "Pearson: N/A"
    )

    # ========================================================
    # 7. SAVE REPORT
    # ========================================================

    print(
        "\n[7/7] Saving final evaluation..."
    )

    save_final_report(
        metrics
    )

    print(
        "\n================================================"
    )

    print(
        "       EVALUATION COMPLETE"
    )

    print(
        "================================================"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()