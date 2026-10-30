"""
transcribe_user.py -- turn one participant's interview audio into Tamil text.

Give a username; it finds that person's recordings (int_01.webm ... int_50.webm,
as downloaded by export_participant.py), transcribes each one in Tamil with
AI4Bharat IndicConformer, and writes, under the same username folder:

  export/participants/<username>/interview_transcript_ta.txt
      Q1. <the interview question>
      Transcript: <what the participant said, in Tamil script>
      (then Q2, Q3, ... in order)
  export/participants/<username>/interview_transcript_ta.json   (same, structured)
  export/participants/<username>/transcribe_cache/...           (resume data)

If the audio has not been downloaded yet, it runs the export first (needs your
Supabase details in .env.local).

Usage (after running scripts/setup_transcribe.sh or .bat and activating the venv):
  python scripts/transcribe_user.py raja_23
  python scripts/transcribe_user.py raja_23 meena_7
  python scripts/transcribe_user.py all            # every folder already exported
  python scripts/transcribe_user.py raja_23 --force        # ignore cached work
  python scripts/transcribe_user.py raja_23 --no-fetch     # never contact Supabase

Resumable: every ~30 s chunk is saved the moment it is done, so a crash or Ctrl+C
loses nothing; run the same command again to continue.
If a participant re-records an answer, delete that audio file (and re-export) or
use --force so the old transcript is not reused.
"""
import argparse, json, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
DEFAULT_ROOT = ROOT / "export" / "participants"
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
AUDIO_EXT = {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".mp4", ".mpeg"}

_MODEL = None


# ---------- questions ----------
def load_questions(root, usernames):
    """Load interview questions from questions.json or an exported participant report."""
    question_file = HERE / "questions.json"
    if question_file.exists():
        data = json.loads(question_file.read_text(encoding="utf-8"))
        out, n = {}, 0
        for section, items in data["interview"]:
            for text in items:
                n += 1
                out[f"int_{n:02d}"] = (section, text)
        return out

    for username in usernames:
        report_file = root / username / "all.json"
        if not report_file.exists():
            continue
        report = json.loads(report_file.read_text(encoding="utf-8"))
        interview = report.get("interview")
        if interview:
            return {
                item["key"]: (item["section"], item["question"])
                for item in interview
                if item.get("key") and item.get("question")
            }

    raise FileNotFoundError(
        f"Interview questions not found. Add {question_file} or export a participant first."
    )


# ---------- audio ----------
def load_audio(path):
    """Decode any audio format (webm/mp4/wav...) to 16 kHz mono float32 numpy via ffmpeg."""
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav = f.name
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SAMPLE_RATE), wav], check=True)
        audio, sr = sf.read(wav, dtype="float32")
    finally:
        Path(wav).unlink(missing_ok=True)
    assert sr == SAMPLE_RATE
    return audio


def chunk_audio(audio):
    """Overlapping ~30 s chunks. Short answers (15-60 s) give 1-2 chunks."""
    step = (CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE
    size = CHUNK_SECONDS * SAMPLE_RATE
    for start in range(0, len(audio), step):
        piece = audio[start:start + size]
        if start == 0 or len(piece) > SAMPLE_RATE:  # skip trailing slivers under 1 s
            yield piece


def merge_texts(texts, max_overlap_words=10):
    """Join chunk texts, dropping words repeated in the 2 s overlap between chunks."""
    merged = []
    for t in texts:
        words = t.split()
        if not words:
            continue
        for k in range(min(max_overlap_words, len(merged), len(words)), 0, -1):
            if merged[-k:] == words[:k]:
                words = words[k:]
                break
        merged += words
    return " ".join(merged)


# ---------- model ----------
def get_model(device):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        print(f"Loading {MODEL_NAME} on {device} (first run downloads ~2.4 GB) ...")
        _MODEL = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(device).eval()
    return _MODEL


def transcribe_chunk(model, chunk, lang, decoder, device):
    import torch
    with torch.no_grad():
        out = model(torch.from_numpy(chunk).unsqueeze(0).to(device), lang, decoder)
    return str(out[0] if isinstance(out, (list, tuple)) else out).strip()


def transcribe_file(audio_path, cache_dir, args, model_loader):
    """Transcribe one recording; chunk results are cached so reruns skip finished work."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    size_file = cache_dir / ".size"
    size = str(audio_path.stat().st_size)
    if args.force or (size_file.exists() and size_file.read_text() != size):
        for old in cache_dir.glob("chunk_*.txt"):
            old.unlink()
    size_file.write_text(size)

    audio = load_audio(audio_path)
    seconds = round(len(audio) / SAMPLE_RATE, 1)
    texts = []
    for i, chunk in enumerate(chunk_audio(audio)):
        f = cache_dir / f"chunk_{i:04d}.txt"
        if f.exists():
            texts.append(f.read_text(encoding="utf-8"))
        else:
            t = transcribe_chunk(model_loader(), chunk, args.lang, args.decoder, args.device)
            f.write_text(t, encoding="utf-8")
            texts.append(t)
    return merge_texts(texts), seconds


# ---------- audio location ----------
def audio_files(user_dir):
    d = user_dir / "audio"
    if not d.is_dir():
        return {}
    return {p.stem: p for p in sorted(d.iterdir()) if p.suffix.lower() in AUDIO_EXT}


def ensure_audio(username, root, fetch):
    user_dir = root / username
    if audio_files(user_dir) or not fetch:
        return
    print(f"No audio found for {username}; downloading from Supabase ...")
    sys.path.insert(0, str(HERE))
    import export_participant
    old = sys.argv
    sys.argv = ["export_participant.py", username]
    try:
        export_participant.main()
    except SystemExit as e:
        if e.code not in (None, 0):
            print(f"  export stopped: {e.code}")
    finally:
        sys.argv = old


# ---------- reports ----------
def build_report(username, questions, results):
    """results: key -> {'transcript', 'seconds', 'status'} -> (text, json_rows). Pure, testable."""
    lines = [f"AUDIO INTERVIEW TRANSCRIPT (Tamil) - {username}", "=" * 50, ""]
    rows, section = [], None
    for key, (sec, q) in questions.items():
        num = int(key.split("_")[1])
        if sec != section:
            section = sec
            lines += [f"--- {sec} ---", ""]
        r = results.get(key)
        if r is None:
            text = "(no audio recorded)"
        elif r["status"] == "failed":
            text = f"[transcription failed: {r['error']}]"
        else:
            text = r["transcript"] or "(no speech detected)"
        lines += [f"Q{num}. {q}", f"Transcript: {text}", ""]
        rows.append({"number": num, "key": key, "section": sec, "question": q,
                     "seconds": (r or {}).get("seconds"), "transcript": text})
    return "\n".join(lines), rows


def run_user(username, root, questions, args, loader):
    ensure_audio(username, root, not args.no_fetch)
    user_dir = root / username
    files = audio_files(user_dir)
    if not files:
        print(f"{username}: no audio files in {user_dir / 'audio'} - skipped")
        return False
    print(f"\n=== {username}: {len(files)} recording(s) ===")
    results, n_done = {}, 0
    for key in questions:
        path = files.get(key)
        if path is None:
            continue
        n_done += 1
        try:
            text, secs = transcribe_file(path, user_dir / "transcribe_cache" / key, args, loader)
            results[key] = {"status": "ok", "transcript": text, "seconds": secs}
            print(f"  [{n_done:>2}/{len(files)}] {key} ({secs} s): {text[:70]}{'...' if len(text) > 70 else ''}")
        except Exception as exc:  # keep going; one bad file must not lose the rest
            results[key] = {"status": "failed", "error": str(exc), "transcript": "", "seconds": None}
            print(f"  [{n_done:>2}/{len(files)}] {key} FAILED: {exc}")
    text, rows = build_report(username, questions, results)
    (user_dir / "interview_transcript_ta.txt").write_text(text, encoding="utf-8")
    (user_dir / "interview_transcript_ta.json").write_text(
        json.dumps({"username": username, "language": args.lang, "model": MODEL_NAME, "answers": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  saved -> {user_dir / 'interview_transcript_ta.txt'}")
    return True


def main():
    ap = argparse.ArgumentParser(description="Transcribe a participant's interview audio to Tamil text.")
    ap.add_argument("usernames", nargs="+", help="username(s), or 'all' for every exported participant folder")
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="participants folder (default: export/participants)")
    ap.add_argument("--lang", default="ta", help="language code (default: ta)")
    ap.add_argument("--decoder", default="rnnt", choices=["rnnt", "ctc"], help="decoding method (default: rnnt)")
    ap.add_argument("--device", default=None, help="cuda or cpu (default: cuda if available)")
    ap.add_argument("--force", action="store_true", help="ignore cached chunks and redo everything")
    ap.add_argument("--no-fetch", action="store_true", help="do not download missing audio from Supabase")
    args = ap.parse_args()

    try:
        import torch  # noqa: F401
        import soundfile  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as e:
        sys.exit(f"Missing package ({e.name}). Run scripts/setup_transcribe.sh (or .bat), then activate .venv-transcribe.")
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found. Install it (Windows: winget install ffmpeg | Mac: brew install ffmpeg | Linux: apt install ffmpeg).")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    names = args.usernames
    if [n.lower() for n in names] == ["all"]:
        names = sorted(p.name for p in root.iterdir() if p.is_dir() and audio_files(p))
        if not names:
            sys.exit("No exported participants found. Run export_participant.py first.")

    questions = load_questions(root, names)
    loader = lambda: get_model(args.device)
    ok = sum(bool(run_user(u, root, questions, args, loader)) for u in names)
    print(f"\nDone. {ok}/{len(names)} participant(s) transcribed.")


if __name__ == "__main__":
    main()
