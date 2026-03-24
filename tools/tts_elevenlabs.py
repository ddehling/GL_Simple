"""
ElevenLabs Text-to-Speech utility
Generates speech audio files from text and saves them to media/sounds/

Install: pip install elevenlabs

Usage:
    python tools/tts_elevenlabs.py "Your text here"
    python tools/tts_elevenlabs.py "Your text here" --voice "Rachel" --output my_file.mp3
    python tools/tts_elevenlabs.py --list-voices
    python tools/tts_elevenlabs.py --batch texts.txt

API key priority:
    1. ELEVENLABS_API_KEY environment variable (recommended)
    2. --api-key argument
"""

import os
import sys
import argparse
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "media" / "sounds"


def get_client(api_key=None):
    from elevenlabs.client import ElevenLabs
    key = os.environ.get("ELEVENLABS_API_KEY") or api_key
    if not key:
        print("ERROR: No API key found. Set ELEVENLABS_API_KEY env var or pass --api-key.")
        sys.exit(1)
    return ElevenLabs(api_key=key)


def list_voices(client):
    voices = client.voices.get_all().voices
    print(f"\n{'Name':<30} {'Voice ID':<30} {'Category'}")
    print("-" * 75)
    for v in sorted(voices, key=lambda x: x.name):
        print(f"{v.name:<30} {v.voice_id:<30} {v.category or ''}")
    return voices


def resolve_voice_id(client, name_or_id):
    voices = client.voices.get_all().voices

    for v in voices:
        if v.voice_id == name_or_id:
            return v.voice_id

    name_lower = name_or_id.lower()
    for v in voices:
        if v.name.lower() == name_lower:
            return v.voice_id

    matches = [v for v in voices if name_lower in v.name.lower()]
    if len(matches) == 1:
        return matches[0].voice_id
    if len(matches) > 1:
        print(f"Ambiguous voice '{name_or_id}'. Matches:")
        for v in matches:
            print(f"  {v.name} ({v.voice_id})")
        sys.exit(1)

    print(f"ERROR: Voice '{name_or_id}' not found. Use --list-voices to see options.")
    sys.exit(1)


def generate_and_save(client, text, voice_id, filename,
                      model="eleven_multilingual_v2",
                      stability=0.5, similarity=0.75, style=0.0):
    from elevenlabs import VoiceSettings

    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id=model,
        voice_settings=VoiceSettings(
            stability=stability,
            similarity_boost=similarity,
            style=style,
            use_speaker_boost=True,
        ),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / filename
    with open(out_path, "wb") as f:
        for chunk in audio:
            f.write(chunk)

    print(f"Saved: {out_path}")
    return out_path


def make_filename(text, output=None):
    if output:
        return output if output.endswith(".mp3") else output + ".mp3"
    slug = text[:40].strip().lower()
    slug = "".join(c if c.isalnum() or c in " _-" else "" for c in slug)
    slug = slug.replace(" ", "_").strip("_")
    return f"tts_{slug}.mp3"


def main():
    parser = argparse.ArgumentParser(description="ElevenLabs TTS — generate speech files")
    parser.add_argument("text", nargs="?", help="Text to synthesize")
    parser.add_argument("--voice", default="Rachel", help="Voice name or ID (default: Rachel)")
    parser.add_argument("--output", "-o", help="Output filename (saved to media/sounds/)")
    parser.add_argument("--model", default="eleven_multilingual_v2",
                        help="Model ID (default: eleven_multilingual_v2)")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity", type=float, default=0.75)
    parser.add_argument("--style", type=float, default=0.0)
    parser.add_argument("--list-voices", action="store_true", help="List available voices and exit")
    parser.add_argument("--batch", help="Text file — one line per item to generate")
    parser.add_argument("--api-key", help="ElevenLabs API key (prefer ELEVENLABS_API_KEY env var)")

    args = parser.parse_args()
    client = get_client(args.api_key)

    if args.list_voices:
        list_voices(client)
        return

    # If it looks like a raw voice ID (21+ char alphanumeric), skip the API lookup
    if len(args.voice) >= 20 and args.voice.replace("_", "").isalnum():
        voice_id = args.voice
    else:
        voice_id = resolve_voice_id(client, args.voice)

    if args.batch:
        lines = Path(args.batch).read_text(encoding="utf-8").strip().splitlines()
        lines = [l.strip() for l in lines if l.strip() and not l.startswith("#")]
        print(f"Generating {len(lines)} files...")
        for i, line in enumerate(lines, 1):
            print(f"[{i}/{len(lines)}] {line[:60]}...")
            fname = make_filename(line)
            generate_and_save(client, line, voice_id, fname, args.model,
                              args.stability, args.similarity, args.style)
        return

    if not args.text:
        parser.print_help()
        sys.exit(1)

    fname = make_filename(args.text, args.output)
    generate_and_save(client, args.text, voice_id, fname, args.model,
                      args.stability, args.similarity, args.style)


if __name__ == "__main__":
    main()
