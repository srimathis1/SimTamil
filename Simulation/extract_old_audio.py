#!/usr/bin/env python3
"""Transcribe all audio files for one participant into a single Tamil .txt file.

Expected layout (place this script beside the participant folders):
    Data_Processor/
      extract_pr.py
      Sharmila/
        audio/
          Sharmila_1.webm
          Sharmila_2.webm
      Surya/
        audio/
          ...

Run:
    python extract_pr.py Sharmila
    python extract_pr.py --list
    python extract_pr.py all

Uses AI4Bharat IndicConformer. The model is downloaded on first run.
Requires ffmpeg on PATH.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
MODEL_NAME = "ai4bharat/indic-conformer-600m-multilingual"
AUDIO_EXTENSIONS = {
    ".mp3", ".m4a", ".wav", ".webm", ".ogg", ".opus", ".flac",
    ".aac", ".mp4", ".mpeg", ".mpg", ".wma", ".amr", ".3gp",
    ".mov", ".mkv", ".aiff",
}
_MODEL = None


def natural_key(value):
    """Sort file names naturally: clip_2 before clip_10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", value)]


def find_child_dir(parent: Path, name: str):
    """Find a direct child directory without case sensitivity."""
    if not parent.is_dir():
        return None
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == name.lower():
            return child
    return None


def find_user_dir(base: Path, username: str):
    """Find the participant folder, case-insensitively."""
    if not base.is_dir():
        return None
    for child in base.iterdir():
        if child.is_dir() and child.name.lower() == username.lower():
            return child
    return None


def list_users(base: Path):
    users = []
    if not base.is_dir():
        return users
    for folder in base.iterdir():
        if folder.is_dir() and not folder.name.startswith((".", "_")):
            if find_child_dir(folder, "audio") is not None:
                users.append(folder.name)
    return sorted(users, key=str.lower)


def get_audio_files(audio_dir: Path, any_file=False):
    return sorted(
        [p for p in audio_dir.iterdir()
         if p.is_file() and not p.name.startswith((".", "~"))
         and (any_file or p.suffix.lower() in AUDIO_EXTENSIONS)],
        key=lambda p: natural_key(p.name),
    )


def load_audio(path: Path):
    """Convert any supported input to mono, 16 kHz float audio using ffmpeg."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        wav_path = Path(temp.name)
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), str(wav_path)],
            check=True,
        )
        audio, _sample_rate = sf.read(str(wav_path), dtype="float32")
        return audio
    finally:
        wav_path.unlink(missing_ok=True)


def chunk_audio(audio):
    """Yield ~30-second chunks with a 2-second overlap."""
    size = CHUNK_SECONDS * SAMPLE_RATE
    step = (CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE
    for start in range(0, len(audio), step):
        chunk = audio[start:start + size]
        if start == 0 or len(chunk) > SAMPLE_RATE:
            yield chunk


def get_model(device):
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        print(f"Loading {MODEL_NAME} on {device}; first run downloads the model.")
        _MODEL = AutoModel.from_pretrained(
            MODEL_NAME, trust_remote_code=True
        ).to(device).eval()
    return _MODEL


def transcribe_chunk(audio_chunk, device):
    import torch
    model = get_model(device)
    audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).to(device)
    with torch.no_grad():
        result = model(audio_tensor, "ta", "rnnt")
    if isinstance(result, (list, tuple)):
        result = result[0]
    return str(result).strip()


def merge_chunk_texts(texts, max_overlap_words=12):
    """Remove exact repeated words caused by chunk overlap where possible."""
    output = []
    previous_tail = []
    for text in texts:
        words = text.split()
        matched = 0
        for count in range(min(max_overlap_words, len(previous_tail), len(words)), 0, -1):
            if previous_tail[-count:] == words[:count]:
                matched = count
                break
        words = words[matched:]
        if words:
            output.append(" ".join(words))
            previous_tail = (previous_tail + words)[-max_overlap_words:]
    return output


def file_fingerprint(path: Path):
    stat = path.stat()
    raw = f"{stat.st_size}-{stat.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def transcribe_recording(audio_path: Path, cache_dir: Path, device: str, force=False):
    """Transcribe a recording with resumable per-chunk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    info_path = cache_dir / "info.json"
    lines_path = cache_dir / "lines.json"
    fingerprint = file_fingerprint(audio_path)

    info = {}
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = {}

    if force or info.get("fingerprint") != fingerprint:
        for cached_chunk in cache_dir.glob("chunk_*.txt"):
            cached_chunk.unlink(missing_ok=True)
        lines_path.unlink(missing_ok=True)
        info = {"fingerprint": fingerprint}
        info_path.write_text(json.dumps(info), encoding="utf-8")

    if info.get("done") and lines_path.exists():
        return json.loads(lines_path.read_text(encoding="utf-8")), info["seconds"]

    audio = load_audio(audio_path)
    duration = round(len(audio) / SAMPLE_RATE, 1)
    chunks = list(chunk_audio(audio))
    chunk_texts = []

    for index, chunk in enumerate(chunks):
        chunk_path = cache_dir / f"chunk_{index:04d}.txt"
        if not chunk_path.exists():
            text = transcribe_chunk(chunk, device)
            chunk_path.write_text(text, encoding="utf-8")
        chunk_texts.append(chunk_path.read_text(encoding="utf-8"))
        print(f"\r      chunk {index + 1}/{len(chunks)}", end="", flush=True)
    if chunks:
        print()

    lines = merge_chunk_texts(chunk_texts)
    lines_path.write_text(json.dumps(lines, ensure_ascii=False), encoding="utf-8")
    info.update(done=True, seconds=duration, chunks=len(chunks))
    info_path.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
    return lines, duration


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe all audio for a participant into one Tamil .txt file."
    )
    parser.add_argument("names", nargs="*", help="Participant folder name(s), or 'all'.")
    parser.add_argument("--list", action="store_true", help="List participant folders with audio.")
    parser.add_argument("--base", default=str(HERE), help="Directory containing participant folders.")
    parser.add_argument("--out", help="Output .txt path (only for one participant).")
    parser.add_argument("--plain", action="store_true", help="Write only transcript text, without headers.")
    parser.add_argument("--one-paragraph", action="store_true", help="Join each recording's chunks into one paragraph.")
    parser.add_argument("--any-file", action="store_true", help="Try every file in audio/, regardless of extension.")
    parser.add_argument("--force", action="store_true", help="Re-transcribe even if cached results exist.")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Inference device; default auto-detects CUDA.")
    args = parser.parse_args()

    base = Path(args.base).expanduser().resolve()
    if not base.is_dir():
        sys.exit(f"Base directory does not exist: {base}")

    if args.list or not args.names:
        users = list_users(base)
        print(f"Participant folders with an audio/ directory in {base}:")
        print("\n".join(f"  {user}" for user in users) if users else "  (none found)")
        if not args.list:
            print("\nRun: python extract_pr.py <participant_name>")
        if args.list or not args.names:
            return

    names = list_users(base) if [name.lower() for name in args.names] == ["all"] else args.names
    if args.out and len(names) != 1:
        sys.exit("--out can only be used with one participant.")

    try:
        import torch  # noqa: F401
        import soundfile  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        sys.exit(f"Missing Python package '{exc.name}'. Install requirements first.")

    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg was not found on PATH. Install FFmpeg and reopen the terminal.")

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    all_ok = True
    for username in names:
        user_dir = find_user_dir(base, username)
        if user_dir is None:
            print(f"\n[Error] Participant folder not found: {username}")
            print("Available participants: " + ", ".join(list_users(base)))
            all_ok = False
            continue

        audio_dir = find_child_dir(user_dir, "audio")
        if audio_dir is None:
            print(f"\n[Error] No audio folder for {user_dir.name}")
            all_ok = False
            continue

        audio_files = get_audio_files(audio_dir, any_file=args.any_file)
        if not audio_files:
            print(f"\n[Error] No supported audio files found in {audio_dir}")
            print("Use --any-file if the files have unusual extensions.")
            all_ok = False
            continue

        print(f"\n=== {user_dir.name}: {len(audio_files)} audio file(s) ===")
        results = []
        for index, audio_path in enumerate(audio_files, 1):
            print(f"[{index}/{len(audio_files)}] {audio_path.name}")
            try:
                lines, duration = transcribe_recording(
                    audio_path,
                    user_dir / ".transcribe_cache" / audio_path.stem,
                    args.device,
                    force=args.force,
                )
                text = (" " if args.one_paragraph else "\n").join(lines)
                results.append((audio_path.name, text, duration, None))
                print(f"      done ({format_duration(duration)})")
            except KeyboardInterrupt:
                print("\nInterrupted. Run the same command again to resume from cached chunks.")
                return
            except Exception as exc:
                results.append((audio_path.name, "", 0, str(exc)))
                print(f"      FAILED: {exc}")
                all_ok = False

        output_path = Path(args.out).expanduser().resolve() if args.out else user_dir / f"{user_dir.name}_transcript.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_lines = []
        if not args.plain:
            output_lines.extend([
                f"TRANSCRIPT - {user_dir.name}",
                f"Language: Tamil | Model: {MODEL_NAME}",
                f"Recordings: {len(results)}",
                "=" * 60,
                "",
            ])
        for filename, transcript, duration, error in results:
            if not args.plain:
                output_lines.extend([f"[{filename}] ({format_duration(duration) if duration else 'duration unknown'})", ""])
            if error:
                output_lines.append(f"[TRANSCRIPTION FAILED: {error}]")
            else:
                output_lines.append(transcript or "(no speech detected)")
            output_lines.append("")

        output_path.write_text("\n".join(output_lines).rstrip() + "\n", encoding="utf-8")
        print(f"[Saved] {output_path}")

    print("\nFinished." if all_ok else "\nFinished with one or more errors.")


if __name__ == "__main__":
    main()
