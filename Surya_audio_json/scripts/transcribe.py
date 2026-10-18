"""
transcribe.py -- Stage 1 of the Tamil grounded-agent pipeline.

Chunks each input audio file into overlapping ~30s "batches" and
transcribes them one batch at a time with AI4Bharat's IndicConformer
(RNN-T decoding). Saves directly to:
    data/Transcripts/audio_<participant_id>.txt

Usage:
    python scripts/transcribe.py
"""

import argparse
import os
import sys
from pathlib import Path

# --- FIX WINDOWS PATH FOR FFMPEG & FFPROBE VIA STATIC-FFMPEG ---
import static_ffmpeg
static_ffmpeg.add_paths()
# --------------------------------------------------------------

from pydub import AudioSegment
import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16000
CHUNK_SECONDS = 30
OVERLAP_SECONDS = 2
AUDIO_EXTENSIONS = {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".mpeg", ".mpg", ".mp4"}

_MODEL = None
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_audio(path: str) -> torch.Tensor:
    """
    Decode audio to 16kHz mono float32 tensor using pydub,
    handling any non-standard/corrupted MPEG frames cleanly.
    """
    sound = AudioSegment.from_file(path)
    sound = sound.set_frame_rate(SAMPLE_RATE).set_channels(1)
    
    samples = np.array(sound.get_array_of_samples()).astype(np.float32)
    # Normalize 16-bit integer PCM to [-1.0, 1.0]
    samples = samples / (1 << (8 * sound.sample_width - 1))
    return torch.from_numpy(samples)


def chunk_audio(waveform: torch.Tensor):
    """Yield overlapping ~30s chunks so nothing near a boundary gets cut mid-word."""
    step = (CHUNK_SECONDS - OVERLAP_SECONDS) * SAMPLE_RATE
    chunk_len = CHUNK_SECONDS * SAMPLE_RATE
    for start in range(0, waveform.shape[0], step):
        chunk = waveform[start:start + chunk_len]
        if chunk.shape[0] > SAMPLE_RATE:  # skip trailing slivers < 1s
            yield chunk


def get_model():
    """Lazy-load and cache the ASR model so a batch of files only pays load cost once."""
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModel
        print(f"Loading ai4bharat/indic-conformer-600m-multilingual on {_DEVICE} ...")
        _MODEL = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        ).to(_DEVICE).eval()
    return _MODEL


def transcribe_chunk(model, chunk: torch.Tensor, lang: str) -> str:
    with torch.no_grad():
        out = model(chunk.unsqueeze(0).to(_DEVICE), lang, "rnnt")
    text = out[0] if isinstance(out, (list, tuple)) else out
    return str(text).strip()


def transcribe_file(audio_path: Path, transcripts_dir: Path, lang: str) -> Path:
    participant_id = audio_path.stem
    chunks_dir = transcripts_dir / "chunks" / participant_id
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Matching the pattern expected by reflect_and_score.py: audio_*.txt
    transcript_path = transcripts_dir / f"audio_{participant_id}.txt"

    print(f"\n=== Processing: {participant_id} ===")
    waveform = load_audio(str(audio_path))
    duration_min = waveform.shape[0] / SAMPLE_RATE / 60
    print(f"Loaded: {duration_min:.2f} minutes")

    model = get_model()
    chunk_texts = []
    for i, chunk in enumerate(chunk_audio(waveform)):
        chunk_file = chunks_dir / f"chunk_{i:04d}.txt"
        if chunk_file.exists():
            text = chunk_file.read_text(encoding="utf-8")
            print(f"  batch {i}: already done, skipping")
        else:
            text = transcribe_chunk(model, chunk, lang)
            chunk_file.write_text(text, encoding="utf-8")
            print(f"  batch {i}: {text[:80]}{'...' if len(text) > 80 else ''}")
        chunk_texts.append(text)

    merged = " ".join(t for t in chunk_texts if t)
    transcript_path.write_text(merged, encoding="utf-8")
    print(f"Merged transcript -> {transcript_path} ({len(merged)} chars)")
    return transcript_path


def main():
    project_root = Path(__file__).resolve().parent.parent
    default_audio_dir = project_root / "data" / "audio"
    default_out_dir = project_root / "data" / "Transcripts"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(default_audio_dir), help="Audio file or directory")
    parser.add_argument("--out", default=str(default_out_dir), help="Output directory (default: data/Transcripts)")
    parser.add_argument("--lang", default="ta", help="Language code for the ASR model (default: ta)")
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_dir():
        audio_files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not audio_files:
            print(f"No audio files found in {input_path}")
            sys.exit(1)
    elif input_path.is_file():
        audio_files = [input_path]
    else:
        print(f"Not found: {input_path}")
        sys.exit(1)

    print(f"Found {len(audio_files)} audio file(s) to process.")
    for audio_path in audio_files:
        transcribe_file(audio_path, out_dir, args.lang)

    print("\nAll audio files successfully transcribed with IndicConformer!")


if __name__ == "__main__":
    main()