import json
from pathlib import Path

# =====================================================================
# WAVE 1 COMPLETE QUESTION BANK (BFI 44 + GSS 100 + VIG 5 + GAMES 5)
# =====================================================================

WAVE_1_BFI_ORDER = [
    # Extraversion (8 items)
    ("BFI-EXT-1", "Enjoys being the center of attention in a group"),
    ("BFI-EXT-2", "Feels energized after spending time around lots of people"),
    ("BFI-EXT-3", "Speaks up easily in conversations, even with strangers"),
    ("BFI-EXT-4", "Prefers quiet, low-key settings over lively gatherings (R)"),
    ("BFI-EXT-5", "Finds it easy to introduce myself to new people"),
    ("BFI-EXT-6", "Tends to hold back rather than take the lead (R)"),
    ("BFI-EXT-7", "Gets excited about upcoming social events"),
    ("BFI-EXT-8", "Feels most comfortable working alone rather than in a group (R)"),

    # Agreeableness (9 items)
    ("BFI-AGR-1", "Goes out of my way to help people, even at personal cost"),
    ("BFI-AGR-2", "Assumes people generally mean well"),
    ("BFI-AGR-3", "Finds it easy to forgive others after a conflict"),
    ("BFI-AGR-4", "Can be sharp or critical when annoyed (R)"),
    ("BFI-AGR-5", "Enjoys cooperating with others rather than competing"),
    ("BFI-AGR-6", "Tends to be suspicious of others' motives (R)"),
    ("BFI-AGR-7", "Feels genuine concern for people I don't know well"),
    ("BFI-AGR-8", "Argues just to win, even when I might be wrong (R)"),
    ("BFI-AGR-9", "Tries to see things from other people's point of view"),

    # Conscientiousness (9 items)
    ("BFI-CON-1", "Keeps my belongings and workspace well organized"),
    ("BFI-CON-2", "Follows through on commitments, even small ones"),
    ("BFI-CON-3", "Plans tasks carefully before starting them"),
    ("BFI-CON-4", "Leaves things until the last minute (R)"),
    ("BFI-CON-5", "Pays close attention to details in my work"),
    ("BFI-CON-6", "Loses focus easily and jumps between tasks (R)"),
    ("BFI-CON-7", "Sets clear goals and works steadily toward them"),
    ("BFI-CON-8", "Forgets appointments or deadlines fairly often (R)"),
    ("BFI-CON-9", "Double-checks my work before calling it finished"),

    # Neuroticism (8 items)
    ("BFI-NEU-1", "Worries about things long after they've happened"),
    ("BFI-NEU-2", "Stays calm even when plans fall apart (R)"),
    ("BFI-NEU-3", "Gets stressed easily by everyday hassles"),
    ("BFI-NEU-4", "Feels emotionally steady most of the time (R)"),
    ("BFI-NEU-5", "Overthinks decisions, even small ones"),
    ("BFI-NEU-6", "Bounces back quickly from disappointment (R)"),
    ("BFI-NEU-7", "Feels anxious in unfamiliar situations"),
    ("BFI-NEU-8", "Rarely feels overwhelmed by pressure (R)"),

    # Openness (10 items)
    ("BFI-OPE-1", "Enjoys discussing abstract ideas or theories"),
    ("BFI-OPE-2", "Seeks out new experiences, even ones that feel risky"),
    ("BFI-OPE-3", "Appreciates art, music, or writing that's unconventional"),
    ("BFI-OPE-4", "Prefers familiar routines over trying new things (R)"),
    ("BFI-OPE-5", "Gets curious about topics I know nothing about"),
    ("BFI-OPE-6", "Comes up with unusual or creative solutions to problems"),
    ("BFI-OPE-7", "Sticks to tried-and-true methods rather than experimenting (R)"),
    ("BFI-OPE-8", "Enjoys imagining possibilities that don't yet exist"),
    ("BFI-OPE-9", "Finds philosophical or scientific questions fascinating"),
    ("BFI-OPE-10", "Feels uneasy around ideas that challenge my usual beliefs (R)"),
]


WAVE_1_GSS_ORDER = [
    ("GSS-1", "What is your age group?"),
    ("GSS-2", "Code respondent's sex"),
    ("GSS-3", "Are you currently married, widowed, divorced, separated, or never married?"),
    ("GSS-4", "Highest level of formal education"),
    ("GSS-5", "Respondent's highest degree"),
    ("GSS-6", "How many children have you ever had?"),
    ("GSS-7", "What is your religious preference?"),
    ("GSS-8", "Generally speaking, would you say most people can be trusted or can't be too careful?"),
    ("GSS-9", "Can people be trusted or can't you be too careful in dealing with people?"),
    ("GSS-10", "Do most people try to take advantage of you, or try to be fair?"),
    ("GSS-11", "Do people try to be helpful, or mostly look out for themselves?"),
    ("GSS-12", "How much confidence do you have in the press?"),
    ("GSS-13", "How much confidence do you have in the scientific community?"),
    ("GSS-14", "How much confidence do you have in banks and financial institutions?"),
    ("GSS-15", "How much confidence do you have in organized labor?"),
    ("GSS-16", "How much confidence do you have in organized religion?"),
    ("GSS-17", "Were you living with both parents around age 16?"),
    ("GSS-18", "It's better if the man works outside and the woman takes care of the home."),
    ("GSS-19", "Same-sex couples should have the right to marry."),
    ("GSS-20", "A working mother can have just as warm a relationship with her children."),
    ("GSS-21", "A preschool child is likely to suffer if his or her mother works."),
    ("GSS-22", "Relationship satisfaction with partner (happy/routine/unhappy)"),
    ("GSS-23", "Sometimes necessary to discipline a child with a hard spanking."),
    ("GSS-24", "What is the ideal number of children for a family to have?"),
    ("GSS-25", "How important is being married to you?"),
    ("GSS-26", "How important is having children to you?"),
    ("GSS-27", "Should divorce be easier or more difficult to obtain?"),
    ("GSS-28", "Family life suffers because women concentrate too much on work."),
    ("GSS-29", "Do people get ahead by hard work, or luck/help from others?"),
    ("GSS-30", "How important is it for a child to learn to work hard?"),
    ("GSS-31", "How important is it for a child to learn to help others?"),
    ("GSS-32", "Do local courts deal too harshly or not harshly enough with criminals?"),
    ("GSS-33", "Premarital sexual relations: wrong or not wrong at all?"),
    ("GSS-34", "Favor or oppose the death penalty for murder?"),
    ("GSS-35", "How important is it for a child to learn to think for themselves?"),
    ("GSS-36", "How important is it for a child to learn to obey?"),
    ("GSS-37", "If you had enough money to live comfortably for life, would you keep working?"),
    ("GSS-38", "Compared to parents at your age, is your standard of living better or worse?"),
    ("GSS-39", "Will your children's standard of living be better or worse than yours?"),
    ("GSS-40", "Importance of being cultured"),
    ("GSS-41", "How often do you attend religious services?"),
    ("GSS-42", "About how often do you pray?"),
    ("GSS-43", "Do you believe there is life after death?"),
    ("GSS-44", "Belief in God / Higher Power"),
    ("GSS-45", "Strength of religious affiliation"),
    ("GSS-46", "To what extent do you consider yourself a religious person?"),
    ("GSS-47", "To what extent do you consider yourself a spiritual person?"),
    ("GSS-48", "Participation in place of worship activities (besides services)"),
    ("GSS-49", "Importance of having faith in God"),
    ("GSS-50", "Satisfaction with present financial situation"),
    ("GSS-51", "Financial situation change during the last few years (better/worse/same)"),
    ("GSS-52", "Self-identified social class (lower/working/middle/upper)"),
    ("GSS-53", "Self-placement on a 1-10 social ladder scale"),
    ("GSS-54", "Likelihood of losing job or being laid off in next 12 months"),
    ("GSS-55", "Ease of finding an equally good job elsewhere"),
    ("GSS-56", "Home ownership status (own, rent, other)"),
    ("GSS-57", "Enjoyment of participating in surveys"),
    ("GSS-58", "Unemployed and looking for work for a month or more in past 10 years"),
    ("GSS-59", "Hunting participation by self or spouse"),
    ("GSS-60", "Are you self-employed or work for someone else?"),
    ("GSS-61", "Most recent work: full-time or part-time?"),
    ("GSS-62", "Frequency of being treated with less courtesy or respect"),
    ("GSS-63", "Afraid to walk alone at night in local area?"),
    ("GSS-64", "Importance of having nice things"),
    ("GSS-65", "Do you use a computer?"),
    ("GSS-66", "Mobile internet access at home?"),
    ("GSS-67", "Household has broadband access?"),
    ("GSS-68", "Household uses cellular data for internet?"),
    ("GSS-69", "Frequency of internet use in past 12 months"),
    ("GSS-70", "Frequency of using internet to search information"),
    ("GSS-71", "Frequency of home internet access"),
    ("GSS-72", "Reliability of news on social media"),
    ("GSS-73", "General happiness these days"),
    ("GSS-74", "Is life exciting, routine, or dull?"),
    ("GSS-75", "General health self-rating"),
    ("GSS-76", "Job satisfaction rating"),
    ("GSS-77", "Frequency of social evenings with relatives"),
    ("GSS-78", "Frequency of social evenings with neighbors"),
    ("GSS-79", "Frequency of social evenings with friends outside neighborhood"),
    ("GSS-80", "Daily hours spent watching television"),
    ("GSS-81", "Who is mainly involved when a major household decision has to be made? [TN1]"),
    ("GSS-82", "Duty to personally take care of aging parents rather than paid care [TN2]"),
    ("GSS-83", "Importance of caste/community (jaati) identity [TN3]"),
    ("GSS-84", "Comfort with close family marrying from a different caste [TN4]"),
    ("GSS-85", "Belief that arranged marriages are more stable than love marriages [TN5]"),
    ("GSS-86", "Acceptability of dowry in today's society [TN6]"),
    ("GSS-87", "Who mainly controls day-to-day household financial decisions? [TN7]"),
    ("GSS-88", "Frequency of visiting places of worship of different faiths [TN8]"),
    ("GSS-89", "Frequency of unpaid community/temple contributions [TN9]"),
    ("GSS-90", "First recourse for serious problems: family/community or formal institutions? [TN10]"),
    ("GSS-91", "Degree of English mixing (code-switching) in Tamil conversation [TN11]"),
    ("GSS-92", "Participation level in traditional Tamil festivals [TN12]"),
    ("GSS-93", "Immediate family member living away for work [TN13]"),
    ("GSS-94", "Preference between joint family and nuclear family [TN14]"),
    ("GSS-95", "Worry about actions affecting family honor/reputation (maanam) [TN15]"),
    ("GSS-96", "Family pressure to pursue a specific respectable career path [TN16]"),
    ("GSS-97", "Influence of regional/Tamil identity vs national identity [TN17]"),
    ("GSS-98", "Support for caste-based reservation quotas [TN18]"),
    ("GSS-99", "Younger members should follow elders' decisions even when disagreeing [TN19]"),
    ("GSS-100", "Comfort discussing personal problems openly vs keeping strictly private [TN20]"),
]


WAVE_1_VIGNETTES = [
    ("BEH-1", "Family Obligation vs Personal Opportunity: Likelihood to accept distant job against parental comfort (1-10)"),
    ("BEH-2", "Respecting Elders' Decisions: Comfort in openly disagreeing with father on major life choice (1-10)"),
    ("BEH-3", "Community Reputation & Judgment: Impact on how community views family if fired for poor performance (1-10)"),
    ("BEH-4", "Trust Within Family vs Outsiders: Comfort borrowing from formal financial institutions vs informal networks (1-10)"),
    ("BEH-5", "Marriage Decision-Making: Likelihood to prioritize parents' suggested partner over personal choice (1-10)"),
]


WAVE_1_GAMES = [
    ("ECON-1", "Dictator Game: Amount shared with a distant relative from an unexpected ₹500 windfall (0-500)"),
    ("ECON-2", "Trust Game: Likelihood of lending ₹2,000 to a local shopkeeper/moneylender with a written note (1-10)"),
    ("ECON-3", "Fairness Game: Amount given to a struggling daily-wage laborer in your street from ₹500 (0-500)"),
    ("ECON-4", "Reputation & Default: Impact on lender's willingness to trust if delayed payment is made public (1-10)"),
    ("ECON-5", "Joint Family Financial Decisions: Likelihood to consult family elders before making an investment (1-10)"),
]


def load_json_answers(file_path: Path) -> dict:
    if not file_path.exists():
        return {}

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        return data.get("answers", {})
    except Exception as e:
        print(f"Error loading {file_path.name}: {e}")
        return {}


def generate_wave1(data_dir: Path):
    # CHANGED ONLY FOR SURYA:
    json_dir = data_dir / "json"

    bfi_answers = load_json_answers(json_dir / "bfi-personality-inventory.json")
    gss_answers = load_json_answers(json_dir / "general-social-survey.json")
    beh_answers = load_json_answers(json_dir / "behavioral-vignettes.json")
    econ_answers = load_json_answers(json_dir / "economic-games.json")

    lines = [
        "=== SURYA: WAVE 1 FULL SURVEY RESPONSES ===",
        "Structure: BFI (44 items) + GSS (100 items) + Vignettes (5) + Games (5)\n",
    ]

    # Module A: BFI
    matched_bfi = 0
    lines.append("--- MODULE A: BIG FIVE INVENTORY (BFI - 44 ITEMS) ---")
    for code, q_text in WAVE_1_BFI_ORDER:
        ans = bfi_answers.get(code, "Not Answered")
        if ans != "Not Answered":
            matched_bfi += 1
        lines.append(f"Question [{code}]: {q_text}")
        lines.append(f"Participant Response: {ans}\n")

    # Module B: GSS
    matched_gss = 0
    lines.append("--- MODULE B: GENERAL SOCIAL SURVEY (GSS - 100 ITEMS) ---")
    for code, q_text in WAVE_1_GSS_ORDER:
        ans = gss_answers.get(code, "Not Answered")
        if ans != "Not Answered":
            matched_gss += 1
        lines.append(f"Question [{code}]: {q_text}")
        lines.append(f"Participant Response: {ans}\n")

    # Module C: Behavioral Vignettes
    matched_beh = 0
    lines.append("--- MODULE C: BEHAVIORAL VIGNETTES (5 ITEMS) ---")
    for code, q_text in WAVE_1_VIGNETTES:
        ans = beh_answers.get(code, "Not Answered")
        if ans != "Not Answered":
            matched_beh += 1
        lines.append(f"Scenario [{code}]: {q_text}")
        lines.append(f"Participant Response: {ans}\n")

    # Module D: Economic Games
    matched_econ = 0
    lines.append("--- MODULE D: ECONOMIC GAMES (5 ITEMS) ---")
    for code, q_text in WAVE_1_GAMES:
        ans = econ_answers.get(code, "Not Answered")
        if ans != "Not Answered":
            matched_econ += 1
        lines.append(f"Game [{code}]: {q_text}")
        lines.append(f"Participant Response: {ans}\n")

    out_file = data_dir / "surveys_wave1.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")

    total_matched = matched_bfi + matched_gss + matched_beh + matched_econ
    total_items = (
        len(WAVE_1_BFI_ORDER)
        + len(WAVE_1_GSS_ORDER)
        + len(WAVE_1_VIGNETTES)
        + len(WAVE_1_GAMES)
    )

    print(
        f"✓ Created: {out_file} "
        f"(Matched {total_matched}/{total_items}: "
        f"BFI {matched_bfi}/44, "
        f"GSS {matched_gss}/100, "
        f"Vignettes {matched_beh}/5, "
        f"Games {matched_econ}/5)"
    )


def generate_wave2(data_dir: Path):
    # CHANGED ONLY FOR SURYA:
    w2_dir = data_dir / "wave2"

    w2_files = list(w2_dir.glob("*.json"))

    if not w2_files:
        print("Warning: No JSON file found in wave2/")
        return

    raw = json.loads(w2_files[0].read_text(encoding="utf-8"))
    sections = raw.get("sections", {})
    participant_name = raw.get("participant_id", "Surya")

    lines = [
        f"=== PARTICIPANT: {participant_name} (WAVE 2 REDUCED INSTRUMENT) ===",
        "Structure: BFI (20 items) + GSS (50 items) + Vignettes (5) + Games (5)\n",
    ]

    # BFI (20)
    bfi_items = sections.get("bfi", {}).get("responses", {})
    if bfi_items:
        lines.append("--- MODULE A: PERSONALITY INVENTORY (20 items) ---")
        for code, d in bfi_items.items():
            trait = d.get("trait", "")
            q_text = d.get("text", "")
            raw_v = d.get("raw", "")
            scored_v = d.get("scored", "")
            rev = " (R)" if d.get("reverse") else ""
            lines.append(f"Question [{code}] [{trait}]{rev}: {q_text}")
            lines.append(f"Participant Response: Raw = {raw_v}, Scored = {scored_v}\n")

    # GSS (50)
    gss_items = sections.get("gss", {}).get("responses", {})
    if gss_items:
        lines.append("--- MODULE B: GENERAL SOCIAL SURVEY (50 items) ---")
        for code, d in gss_items.items():
            q_text = d.get("question", "")
            ans = d.get("answer", "")
            lines.append(f"Question [{code}]: {q_text}")
            lines.append(f"Participant Response: {ans}\n")

    # Vignettes (5)
    vig_items = sections.get("vignettes", {}).get("responses", {})
    if vig_items:
        lines.append("--- MODULE C: BEHAVIORAL VIGNETTES (5 items) ---")
        for code, d in vig_items.items():
            cond = d.get("condition", "")
            q_text = d.get("question", "")
            ans = d.get("answer", "")
            lines.append(f"Vignette [{code}] [Condition: {cond}]: {q_text}")
            lines.append(f"Participant Response: {ans} / 10\n")

    # Games (5)
    game_items = sections.get("games", {}).get("responses", {})
    if game_items:
        lines.append("--- MODULE D: ECONOMIC GAMES (5 items) ---")
        for code, d in game_items.items():
            cond = d.get("condition", "")
            q_text = d.get("question", "")
            ans = d.get("answer", "")
            lines.append(f"Economic Game [{code}] [Condition: {cond}]: {q_text}")
            lines.append(f"Participant Response: {ans}\n")

    out_file = data_dir / "surveys_wave2.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Created: {out_file}")


def main():
    base_data = Path("data")
    generate_wave1(base_data)
    generate_wave2(base_data)


if __name__ == "__main__":
    main()