def parse_survey(file_path):

    records = []

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found:\n{file_path}"
        )

    text = file_path.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lines = text.splitlines()

    current_question = None
    current_response = None
    current_module = "Other"

    def save_record():

        nonlocal current_question
        nonlocal current_response

        if current_question and current_response is not None:

            question = clean_text(
                current_question
            )

            response = clean_text(
                current_response
            )

            if question:

                records.append({
                    "id": len(records) + 1,
                    "question": question,
                    "prompt": question,
                    "response": response,
                    "val": response,
                    "module": current_module
                })

        current_question = None
        current_response = None

    for raw_line in lines:

        line = raw_line.strip()

        if not line:
            continue

        lower = line.lower()

        # ====================================================
        # MODULE DETECTION
        # ====================================================

        if (
            "big five inventory" in lower
            or "bfi" in lower
            or "personality inventory" in lower
        ):
            current_module = "BFI"

        elif (
            "general social survey" in lower
            or re.search(r"\bgss\b", lower)
        ):
            current_module = "GSS"

        elif (
            "behavioral vignette" in lower
            or "vignette" in lower
        ):
            current_module = "Behavioral Vignette"

        elif (
            "economic game" in lower
            or "economic games" in lower
        ):
            current_module = "Economic Game"

        # ====================================================
        # QUESTION FORMAT
        #
        # Question [BFI-1]: ...
        # Question [GSS-1]: ...
        # ====================================================

        question_match = re.match(
            r"^Question\s*\[[^\]]+\]\s*:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if question_match:

            save_record()

            current_question = (
                question_match.group(1).strip()
            )

            continue

        # ====================================================
        # SCENARIO FORMAT
        #
        # Scenario [BEH-1]: ...
        # ====================================================

        scenario_match = re.match(
            r"^Scenario\s*\[[^\]]+\]\s*:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if scenario_match:

            save_record()

            current_question = (
                scenario_match.group(1).strip()
            )

            current_module = "Behavioral Vignette"

            continue

        # ====================================================
        # VIGNETTE FORMAT
        # ====================================================

        vignette_match = re.match(
            r"^Vignette\s*\[[^\]]+\].*?:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if vignette_match:

            save_record()

            current_question = (
                vignette_match.group(1).strip()
            )

            current_module = "Behavioral Vignette"

            continue

        # ====================================================
        # ECONOMIC GAME FORMAT
        #
        # Economic Game [ECON-1]: ...
        # ====================================================

        game_match = re.match(
            r"^Economic Game\s*\[[^\]]+\].*?:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if game_match:

            save_record()

            current_question = (
                game_match.group(1).strip()
            )

            current_module = "Economic Game"

            continue

        # ====================================================
        # PARTICIPANT RESPONSE
        #
        # Participant Response: ...
        # ====================================================

        response_match = re.match(
            r"^Participant\s+Response\s*:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if response_match:

            current_response = (
                response_match.group(1).strip()
            )

            continue

        # ====================================================
        # FALLBACK RESPONSE FORMAT
        # ====================================================

        response_match = re.match(
            r"^(Response|Answer)\s*:\s*(.*)$",
            line,
            re.IGNORECASE
        )

        if response_match:

            current_response = (
                response_match.group(2).strip()
            )

            continue

    # Save final record
    save_record()

    return records