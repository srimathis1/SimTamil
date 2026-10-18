import re
import math
from pathlib import Path
from statistics import mean

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None


# ============================================================
# SimTamil - Participant 01
# Big Five Statistical Evaluation
# ============================================================

BASE_DIR = Path(
    r"C:\Users\SRIMATHI S\OneDrive\Desktop\simTamil"
)

SURVEY_FILE = (
    BASE_DIR
    / "data"
    / "surveys_wave1.txt"
)

AI_SCORE_FILE = (
    BASE_DIR
    / "data"
    / "Expert_reflection"
    / "big_five_scores.txt"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "Expert_reflection"
    / "bfi_statistical_evaluation.txt"
)


# ============================================================
# Big Five configuration
# ============================================================

TRAITS = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]

PREFIXES = {
    "Openness": "BFI-OPE",
    "Conscientiousness": "BFI-CON",
    "Extraversion": "BFI-EXT",
    "Agreeableness": "BFI-AGR",
    "Neuroticism": "BFI-NEU"
}


# ============================================================
# Extract Human BFI-44 scores
# ============================================================

def get_human_bfi_scores():

    text = SURVEY_FILE.read_text(
        encoding="utf-8"
    )

    trait_values = {
        trait: []
        for trait in TRAITS
    }

    current_code = None
    current_question = None

    for line in text.splitlines():

        # --------------------------------------------
        # Question line
        # --------------------------------------------

        match = re.search(
            r"Question\s*\[(BFI-[A-Z]+-\d+)\]\s*:\s*(.*)",
            line,
            re.IGNORECASE
        )

        if match:

            current_code = match.group(1).upper()
            current_question = match.group(2)

            continue

        # --------------------------------------------
        # Response line
        # --------------------------------------------

        match = re.search(
            r"Participant Response\s*:\s*(.*)",
            line,
            re.IGNORECASE
        )

        if not match or current_code is None:
            continue

        response = match.group(1).strip()

        number = re.search(
            r"-?\d+(?:\.\d+)?",
            response
        )

        if not number:
            continue

        raw_score = float(number.group())

        # --------------------------------------------
        # Identify trait
        # --------------------------------------------

        trait = None

        for trait_name, prefix in PREFIXES.items():

            if current_code.startswith(prefix):
                trait = trait_name
                break

        if trait is None:
            continue

        # --------------------------------------------
        # Reverse scoring
        #
        # 1 -> 5
        # 2 -> 4
        # 3 -> 3
        # 4 -> 2
        # 5 -> 1
        #
        # Formula = 6 - raw
        # --------------------------------------------

        reverse_item = (
            current_question
            and "(R)" in current_question
        )

        if reverse_item:
            scored_value = 6 - raw_score
        else:
            scored_value = raw_score

        trait_values[trait].append(
            scored_value
        )

        current_code = None
        current_question = None

    # --------------------------------------------
    # Average item scores for each trait
    # --------------------------------------------

    scores = {}

    for trait in TRAITS:

        values = trait_values[trait]

        if values:

            scores[trait] = mean(values)

    return scores


# ============================================================
# Extract AI Big Five scores
# ============================================================

def extract_ai_scores():

    text = AI_SCORE_FILE.read_text(
        encoding="utf-8"
    )

    scores = {}

    # English labels
    patterns = {
        "Openness": [
            r"Openness.{0,200}?(?:Score|score|:)\s*([1-5](?:\.\d+)?)",
            r"Openness.{0,100}?([1-5](?:\.\d+)?)"
        ],

        "Conscientiousness": [
            r"Conscientiousness.{0,200}?(?:Score|score|:)\s*([1-5](?:\.\d+)?)",
            r"Conscientiousness.{0,100}?([1-5](?:\.\d+)?)"
        ],

        "Extraversion": [
            r"Extraversion.{0,200}?(?:Score|score|:)\s*([1-5](?:\.\d+)?)",
            r"Extraversion.{0,100}?([1-5](?:\.\d+)?)"
        ],

        "Agreeableness": [
            r"Agreeableness.{0,200}?(?:Score|score|:)\s*([1-5](?:\.\d+)?)",
            r"Agreeableness.{0,100}?([1-5](?:\.\d+)?)"
        ],

        "Neuroticism": [
            r"Neuroticism.{0,200}?(?:Score|score|:)\s*([1-5](?:\.\d+)?)",
            r"Neuroticism.{0,100}?([1-5](?:\.\d+)?)"
        ]
    }

    for trait in TRAITS:

        for pattern in patterns[trait]:

            match = re.search(
                pattern,
                text,
                re.IGNORECASE |
                re.DOTALL
            )

            if match:

                scores[trait] = float(
                    match.group(1)
                )

                break

    return scores


# ============================================================
# Pearson correlation
# ============================================================

def calculate_pearson(x, y):

    if len(x) < 2:
        return None

    x_mean = mean(x)
    y_mean = mean(y)

    numerator = sum(
        (a - x_mean) *
        (b - y_mean)
        for a, b in zip(x, y)
    )

    denominator_x = math.sqrt(
        sum(
            (a - x_mean) ** 2
            for a in x
        )
    )

    denominator_y = math.sqrt(
        sum(
            (b - y_mean) ** 2
            for b in y
        )
    )

    denominator = (
        denominator_x *
        denominator_y
    )

    if denominator == 0:
        return None

    return numerator / denominator


# ============================================================
# Main evaluation
# ============================================================

def main():

    print("\n==============================================")
    print("      SimTamil - Participant 01")
    print("      Big Five Statistical Evaluation")
    print("==============================================\n")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not SURVEY_FILE.exists():

        raise FileNotFoundError(
            f"Human BFI file not found:\n{SURVEY_FILE}"
        )

    if not AI_SCORE_FILE.exists():

        raise FileNotFoundError(
            f"AI Big Five file not found:\n{AI_SCORE_FILE}"
        )

    # --------------------------------------------------------
    # Get scores
    # --------------------------------------------------------

    human = get_human_bfi_scores()
    ai = extract_ai_scores()

    print("Human BFI Scores:")
    print("-----------------")

    for trait in TRAITS:

        if trait in human:

            print(
                f"{trait:<20} "
                f"{human[trait]:.3f}"
            )

    print("\nAI Big Five Scores:")
    print("-------------------")

    for trait in TRAITS:

        if trait in ai:

            print(
                f"{trait:<20} "
                f"{ai[trait]:.3f}"
            )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    common_traits = [
        trait
        for trait in TRAITS
        if trait in human and trait in ai
    ]

    if len(common_traits) < 2:

        raise RuntimeError(
            "Not enough matching Big Five scores found."
        )

    # --------------------------------------------------------
    # Per-trait calculations
    # --------------------------------------------------------

    absolute_errors = {}
    squared_errors = {}

    for trait in common_traits:

        error = (
            ai[trait]
            - human[trait]
        )

        absolute_errors[trait] = abs(
            error
        )

        squared_errors[trait] = (
            error ** 2
        )

    # --------------------------------------------------------
    # Overall metrics
    # --------------------------------------------------------

    mae = mean(
        absolute_errors.values()
    )

    mse = mean(
        squared_errors.values()
    )

    rmse = math.sqrt(mse)

    human_values = [
        human[t]
        for t in common_traits
    ]

    ai_values = [
        ai[t]
        for t in common_traits
    ]

    pearson = calculate_pearson(
        human_values,
        ai_values
    )

    # --------------------------------------------------------
    # Write output
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "============================================================\n"
        )

        f.write(
            "       SimTamil - Participant 01\n"
        )

        f.write(
            "       BIG FIVE STATISTICAL EVALUATION\n"
        )

        f.write(
            "============================================================\n\n"
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        f.write(
            "1. HUMAN vs AI BIG FIVE SCORES\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            f"{'Trait':<22}"
            f"{'Human BFI':>14}"
            f"{'AI Score':>14}"
            f"{'Abs Error':>14}\n"
        )

        f.write(
            "-" * 64 + "\n"
        )

        for trait in common_traits:

            f.write(
                f"{trait:<22}"
                f"{human[trait]:>14.3f}"
                f"{ai[trait]:>14.3f}"
                f"{absolute_errors[trait]:>14.3f}\n"
            )

        # ----------------------------------------------------
        # Main metrics
        # ----------------------------------------------------

        f.write(
            "\n\n2. STATISTICAL METRICS\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            f"Mean Absolute Error (MAE) : {mae:.4f}\n"
        )

        f.write(
            f"Mean Squared Error (MSE)  : {mse:.4f}\n"
        )

        f.write(
            f"Root Mean Squared Error   : {rmse:.4f}\n"
        )

        if pearson is not None:

            f.write(
                f"Pearson Correlation (r)   : {pearson:.4f}\n"
            )

        else:

            f.write(
                "Pearson Correlation (r)   : N/A\n"
            )

        # ----------------------------------------------------
        # Interpretation
        # ----------------------------------------------------

        f.write(
            "\n\n3. METRIC INTERPRETATION\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            "MAE:\n"
        )

        f.write(
            "Average absolute difference between AI and human "
            "Big Five scores.\n\n"
        )

        f.write(
            "MSE:\n"
        )

        f.write(
            "Average squared prediction error. Larger errors "
            "receive greater penalty.\n\n"
        )

        f.write(
            "RMSE:\n"
        )

        f.write(
            "Square root of MSE, expressed on the original "
            "1-5 Big Five score scale.\n\n"
        )

        f.write(
            "Pearson Correlation:\n"
        )

        f.write(
            "Measures the linear association between the human "
            "and AI Big Five profiles.\n"
        )

        # ----------------------------------------------------
        # Formulas
        # ----------------------------------------------------

        f.write(
            "\n\n4. FORMULAS\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            "Absolute Error = |AI Score - Human Score|\n\n"
        )

        f.write(
            "MAE = Sum(|AI Score - Human Score|) / Number of Traits\n\n"
        )

        f.write(
            "MSE = Sum((AI Score - Human Score)^2) / Number of Traits\n\n"
        )

        f.write(
            "RMSE = sqrt(MSE)\n\n"
        )

        f.write(
            "Pearson r = Covariance(Human, AI) /\n"
        )

        f.write(
            "            (SD_Human × SD_AI)\n"
        )

        # ----------------------------------------------------
        # Methodology
        # ----------------------------------------------------

        f.write(
            "\n\n5. EVALUATION METHODOLOGY\n"
        )

        f.write(
            "------------------------------------------------------------\n"
        )

        f.write(
            "Human ground truth: BFI-44 Wave-1 responses.\n"
        )

        f.write(
            "AI prediction: Big Five scores generated from the "
            "participant transcript and Wave-1 data.\n"
        )

        f.write(
            "Reverse-coded BFI items are scored using 6 - raw response.\n"
        )

        f.write(
            "Each Big Five trait is calculated as the mean of its "
            "corresponding BFI items.\n"
        )

        f.write(
            "All reported evaluation metrics are calculated "
            "statistically; no LLM judge is used.\n"
        )

        f.write(
            "\nNote: Pearson correlation here is descriptive because "
            "this Participant-01 analysis contains only five aggregate "
            "trait observations. A larger participant sample or "
            "item-level predictions provides a stronger basis for "
            "correlation-based inference.\n"
        )

        f.write(
            "\n\n============================================================\n"
        )

        f.write(
            "                    COMPLETE\n"
        )

        f.write(
            "============================================================\n"
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\nStatistical Results")
    print("-------------------")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")

    if pearson is not None:
        print(
            f"Pearson r : {pearson:.4f}"
        )

    print(
        f"\n✓ Saved to:\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()