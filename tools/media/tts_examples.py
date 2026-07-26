"""
ElevenLabs TTS — inflection/tone examples
Generates the same sentence multiple ways to demonstrate markup and settings.

Usage:
    python tools/media/tts_examples.py --api-key YOUR_KEY
    python tools/media/tts_examples.py --api-key YOUR_KEY --voice xPqWReEgvzaxqgYCck2k
"""

import argparse
from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).parent))
from tts_elevenlabs import get_client, generate_and_save

EXAMPLES = [
    {
        "name": "neutral",
        "desc": "Flat baseline — plain statement",
        "text": "The storm is coming. You should find shelter before dark.",
        "stability": 0.7, "similarity": 0.75, "style": 0.0,
    },
    {
        "name": "urgent",
        "desc": "Panic — short punchy sentences, caps for stress",
        "text": "The storm is coming! Get inside NOW. Find shelter — before it's too late!",
        "stability": 0.2, "similarity": 0.75, "style": 0.8,
    },
    {
        "name": "whisper_tense",
        "desc": "Whispered warning — fragmented, trailing off",
        "text": "The storm... it's coming. You need to find shelter. Before dark. Please.",
        "stability": 0.15, "similarity": 0.80, "style": 0.7,
    },
    {
        "name": "ominous",
        "desc": "Slow dread — pauses built into the text itself",
        "text": "The storm... is coming. There is nowhere to run. Find shelter... before dark falls.",
        "stability": 0.5, "similarity": 0.75, "style": 0.5,
    },
    {
        "name": "calm_authority",
        "desc": "Measured, authoritative — news anchor delivery",
        "text": "A storm is approaching. Residents are advised to seek shelter before dark. Stay indoors until further notice.",
        "stability": 0.9, "similarity": 0.75, "style": 0.0,
    },
    {
        "name": "storyteller",
        "desc": "Narrative — setting a scene",
        "text": "They say a storm is coming. The kind that rolls in slow and dark, the kind that swallows the light. Find shelter. Before dark.",
        "stability": 0.4, "similarity": 0.75, "style": 0.6,
    },
    {
        "name": "desperate",
        "desc": "Emotional plea — begging",
        "text": "Please, the storm is coming, you have to find shelter, you have to get inside before dark, please!",
        "stability": 0.1, "similarity": 0.75, "style": 0.9,
    },
    {
        "name": "eerie_quiet",
        "desc": "Unsettling calm — something is wrong",
        "text": "The storm is coming. It always comes. You should find shelter before dark. You always should.",
        "stability": 0.6, "similarity": 0.80, "style": 0.3,
    },
]


def main():
    parser = argparse.ArgumentParser(description="TTS inflection examples")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--voice", default="xPqWReEgvzaxqgYCck2k", help="Voice ID")
    parser.add_argument("--model", default="eleven_multilingual_v2")
    args = parser.parse_args()

    client = get_client(args.api_key)

    print(f"\nGenerating {len(EXAMPLES)} examples...\n")
    for ex in EXAMPLES:
        fname = f"example_{ex['name']}.mp3"
        print(f"  [{ex['name']}] {ex['desc']}")
        generate_and_save(
            client, ex["text"], args.voice, fname, args.model,
            ex["stability"], ex["similarity"], ex["style"]
        )
    print("\nDone. Files saved to media/sounds/")


if __name__ == "__main__":
    main()
