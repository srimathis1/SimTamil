"""
reflect_and_score.py -- Stage 2: Grounded Reflections in Authentic Tamil.

Combines surveys.txt and audio transcripts in Transcripts/, runs them through
the 4 expert personas, and saves plain text reflection files in natural Tamil slang
inside data/Expert_reflection/.

Usage:
    python scripts/reflect_and_score.py
"""

import os
import sys
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_KEY")
# Setup API Key
os.environ["GEMINI_API_KEY"] = api_key
genai.configure(api_key=api_key)
MODEL_NAME = "gemini-3.6-flash"

REFLECTION_PERSONAS = {
    "psychologist": {
        "label": "உளவியல் நிபுணர் (Psychologist)",
        "system_prompt": (
            "நீங்கள் ஒரு அனுபவமிக்க உளவியல் நிபுணர் (Clinical & Personality Psychologist). "
            "கொடுக்கப்பட்டுள்ள நேர்காணல் (Tamil Transcripts) மற்றும் கணக்கெடுப்பு பதில்களை (Surveys) ஆழமாக ஆராயுங்கள். "
            "முக்கிய விதிமுறை: உங்கள் ஆய்வறிக்கை முழுவதும் இயல்பான, எளிமையான அன்றாட பேச்சுத் தமிழ் நடையில் (Tamil conversational style & natural slang) இருக்க வேண்டும். "
            "ஆடியோவில் அந்த நபர் எப்படி பேசினாரோ அந்த வட்டார வழக்கு, உணர்ச்சி வெளிப்பாடுகள், தயக்கங்களை உள்வாங்கி விளக்குங்கள்.\n\n"
            "நீங்கள் விவரிக்க வேண்டியவை:\n"
            "1. பிக் ஃபைவ் குணாதிசயங்கள் (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) மற்றும் அவற்றிற்கான காரணங்கள்.\n"
            "2. அடிப்படை எண்ணங்கள், லட்சியங்கள் மற்றும் சுய உந்துதல்கள்.\n"
            "3. மன அழுத்தம், கஷ்டமான சூழ்நிலைகளை அந்த நபர் கையாளும் விதம் (Coping mechanisms) மற்றும் உணர்ச்சி நிலைகள்.\n"
            "4. வெளிப்படையாகத் தெரியாத மறைமுக எண்ணங்கள் (Implicit biases), அவர்களின் தனித்துவமான பேச்சு பாணி (Vernacular idioms).\n"
            "எதையும் சுருக்கி விடுபடாமல், தரவுகளில் உள்ள உண்மை சம்பவங்களை மேற்கோள் காட்டி தமிழில் தெளிவாக எழுதவும்."
        ),
    },
    "behavioral_economist": {
        "label": "நடத்தை பொருளாதார நிபுணர் (Behavioral Economist)",
        "system_prompt": (
            "நீங்கள் ஒரு நடத்தை பொருளாதார நிபுணர் (Behavioral Economist). "
            "பங்கேற்பாளரின் நேர்காணல் மற்றும் பொருளாதார முடிவுகளை (Economic Games / Surveys) மதிப்பாய்வு செய்யுங்கள். "
            "முக்கிய விதிமுறை: உங்கள் அறிக்கை முழுவதும் இயல்பான பேச்சுத் தமிழில் (conversational Tamil with authentic tone) எழுதப்பட வேண்டும்.\n\n"
            "நீங்கள் விவரிக்க வேண்டியவை:\n"
            "1. ரிஸ்க் எடுக்கும் குணம் (பணம், தொழில், வாழ்க்கை முடிவுகளில் அவர்கள் காட்டும் துணிவு அல்லது தயக்கம்).\n"
            "2. கால முன்னுரிமை (Time preference - இப்போதைய தேவையா அல்லது எதிர்கால சேமிப்பா?).\n"
            "3. முடிவெடுக்கும் பாணி (யோசித்து எடுக்கும் முடிவா அல்லது உடனடி உள்ளுணர்வா?).\n"
            "4. செலவு செய்யும் பழக்கம், குடும்பத்திற்கு செய்யும் நிதி பொறுப்புகள் மற்றும் பணத்தின் மீதான பார்வை.\n"
            "அவர் ஆடியோவில் சொன்ன உண்மையான வாக்கியங்களை மேற்கோள் காட்டி தமிழில் அலசவும்."
        ),
    },
    "political_scientist": {
        "label": "அரசியல் மற்றும் சமூக ஆய்வாளர் (Political Scientist)",
        "system_prompt": (
            "நீங்கள் ஒரு அரசியல் மற்றும் சமூக அறிவியல் நிபுணர். "
            "பங்கேற்பாளரின் சமூகப் பார்வை, சமூகம் சார்ந்த கேள்விகளுக்கான பதில்களை ஆராயுங்கள். "
            "முக்கிய விதிமுறை: முழுவதும் எளிய மற்றும் இயல்பான பேச்சுத் தமிழில் (Authentic Tamil slang & phrasing) எழுதவும்.\n\n"
            "நீங்கள் விவரிக்க வேண்டியவை:\n"
            "1. அவர்களின் சமூக பார்வை மற்றும் கருத்தியல் சார்பு.\n"
            "2. சமூகம், சட்டம் மற்றும் பொது அமைப்புகள் (Institutions) மீதான அவர்களின் நம்பிக்கை அளவு.\n"
            "3. சமூக பிரச்சனைகள் குறித்து அவர்களின் கருத்துக்கள் எவ்வளவு உறுதியானவை அல்லது மாறக்கூடியவை.\n"
            "4. சக மனிதர்கள் மற்றும் அதிகார மையங்களை (Authority) அவர்கள் பார்க்கும் விதம்.\n"
            "ஆடியோ மற்றும் கருத்துக்கணிப்பு அடிப்படையில் மட்டுமே தமிழில் எழுதவும்."
        ),
    },
    "demographer": {
        "label": "மக்கள் தொகை மற்றும் சமூகச் சூழல் ஆய்வாளர் (Demographer)",
        "system_prompt": (
            "நீங்கள் ஒரு சமூகச் சூழல் மற்றும் மக்கள்தொகை ஆய்வாளர் (Demographer). "
            "பங்கேற்பாளரின் அன்றாட வாழ்க்கை முறை, சூழல் மற்றும் வாழ்வியல் பின்னணியை தொகுக்கவும். "
            "முக்கிய விதிமுறை: உங்கள் அறிக்கை முழுவதையும் இயல்பான தமிழ் நடையில் வெளிப்படுத்தவும்.\n\n"
            "நீங்கள் விவரிக்க வேண்டியவை:\n"
            "1. தற்போதைய வாழ்க்கை நிலை, குடும்ப சூழல் மற்றும் பொருளாதார பின்னணி.\n"
            "2. அன்றாட வழக்கங்கள் (Daily routines), கல்லூரி/வேலை பயணங்கள் மற்றும் வாழ்க்கை முறை.\n"
            "3. சமூகம், நண்பர்கள் மற்றும் குடும்பத்துடனான பிணைப்பு.\n"
            "4. அவர்களின் சூழல் அவர்களின் எதிர்கால தேர்வுகளை எவ்வாறு கட்டுப்படுத்துகிறது அல்லது வழிநடத்துகிறது.\n"
            "அவர்களின் சொந்த வாழ்க்கை சூழலை அப்பட்டமாக தமிழில் பதிவு செய்யவும்."
        ),
    },
}

BIG_FIVE_SYSTEM_PROMPT = (
    "நீங்கள் ஒரு ஆளுமை மதிப்பீட்டு நிபுணர் (Personality Psychometrician). "
    "பங்கேற்பாளரின் ஆடியோ உரையாடல்கள் மற்றும் கணக்கெடுப்புத் தரவுகளை மட்டுமே ஆதாரமாகக் கொண்டு, "
    "பிக் ஃபைவ் (OCEAN) ஆளுமைப் பண்புகளுக்கு 1 முதல் 5 வரை மதிப்பெண் (Score) வழங்க வேண்டும். "
    "முக்கிய விதிமுறை: உங்கள் விளக்கம், மேற்கோள்கள் அனைத்தும் இயல்பான பேச்சுத் தமிழில் (Tamil) இருக்க வேண்டும்.\n"
    "ஒவ்வொரு பண்புக்கும்:\n"
    "- பண்பின் பெயர் (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)\n"
    "- மதிப்பெண் (1 - மிகக் குறைவு முதல் 5 - மிக அதிகம் வரை)\n"
    "- நேர்காணல் ஆடியோவில் இருந்து அவர்கள் பேசிய அசல் தமிழ் வரிகள் மற்றும் காரணங்களை தமிழில் விளக்கவும்."
)


def run_llm(system_prompt: str, prompt: str) -> str:
    model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        system_instruction=system_prompt,
    )
    response = model.generate_content(prompt)
    return response.text.strip()


def main():
    data_dir = Path("data")
    transcripts_dir = data_dir / "Transcripts"
    surveys_file = data_dir / "surveys.txt"
    json_dir = data_dir / "participant01" / "json"
    expert_refl_dir = data_dir / "Expert_reflection"

    expert_refl_dir.mkdir(parents=True, exist_ok=True)

    combined_parts = []

    # 1. Load surveys
    if surveys_file.exists():
        combined_parts.append(f"=== QUANTITATIVE SURVEY DATA ===\n{surveys_file.read_text(encoding='utf-8').strip()}")
    elif json_dir.exists():
        print(f"Reading survey modules from {json_dir}...")
        for jf in sorted(json_dir.glob("*.json")):
            combined_parts.append(f"=== SURVEY MODULE: {jf.name} ===\n{jf.read_text(encoding='utf-8').strip()}")
    else:
        print("Warning: Neither data/surveys.txt nor JSON files found.")

    # 2. Load audio transcripts in order
    if transcripts_dir.exists():
        transcript_files = sorted(transcripts_dir.glob("audio_*.txt"))
        if not transcript_files:
            print(f"Error: No audio_*.txt files found in {transcripts_dir}.")
            sys.exit(1)
        for tf in transcript_files:
            content = tf.read_text(encoding="utf-8").strip()
            combined_parts.append(f"=== AUDIO TRANSCRIPT: {tf.name} ===\n{content}")
    else:
        print(f"Error: {transcripts_dir} directory not found.")
        sys.exit(1)

    full_context = "\n\n".join(combined_parts)
    print(f"Loaded context. Generating authentic Tamil expert reflections using {MODEL_NAME}...\n")

    # 3. Generate the 4 Expert Reflections in Tamil
    for key, persona in REFLECTION_PERSONAS.items():
        print(f"Generating Tamil reflection: {persona['label']}...")
        reflection_text = run_llm(
            persona["system_prompt"],
            f"அனைத்து தரவுகளும் (SURVEYS + AUDIO TRANSCRIPTS):\n{full_context}\n\nமேலே உள்ள விதிகளின்படி அசல் பேச்சுத் தமிழில் உங்கள் விரிவான ஆய்வை வழங்கவும்."
        )
        file_content = f"நிபுணர் பிரிவு: {persona['label']}\n{'=' * 50}\n\n{reflection_text}\n"

        out_file = expert_refl_dir / f"{key}_reflection.txt"
        out_file.write_text(file_content, encoding="utf-8")
        print(f"  Saved -> {out_file.name}")

    # 4. Generate Big Five Psychometric Scores in Tamil
    print("\nGenerating Big Five scores in Tamil...")
    big_five_text = run_llm(
        BIG_FIVE_SYSTEM_PROMPT,
        f"அனைத்து தரவுகளும் (SURVEYS + AUDIO TRANSCRIPTS):\n{full_context}\n\nபிக் ஃபைவ் ஆளுமை பண்புகளின் மதிப்பெண்களையும் அதற்கான அசல் தமிழ் ஆதாரங்களையும் வழங்கவும்."
    )
    big_five_content = f"ஆளுமை மதிப்பீடு: Big Five Personality Assessment (Tamil)\n{'=' * 50}\n\n{big_five_text}\n"

    big_five_file = expert_refl_dir / "big_five_scores.txt"
    big_five_file.write_text(big_five_content, encoding="utf-8")
    print(f"  Saved -> {big_five_file.name}\n")

    print("Stage 2 Complete! All expert reflections are saved in authentic Tamil in data/Expert_reflection/.")


if __name__ == "__main__":
    main()