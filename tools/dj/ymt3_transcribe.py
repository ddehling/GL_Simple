"""Transcribe a wav with YourMT3+ (YPTF.MoE+Multi, noPS) into instrument-labelled
notes. Runs in ITS OWN interpreter (the WSL venv ~/ymt3venv: torch cu126,
transformers 4.45.1, lightning), never in the app's, because its pins conflict
with this environment's; lib/dj/ymt3.py calls it as a subprocess.

    python tools/dj/ymt3_transcribe.py <in.wav> <out.json> [--cpu]

out.json: {"notes": [{"onset", "offset", "pitch", "program", "is_drum", "velocity"}, ...],
           "seconds": inference time}

The model code is the YourMT3 Hugging Face Space (GPL-3.0 / Apache-2.0 space
files) checked out under models/yourmt3/space, with the checkpoint at
amt/logs/2024/<exp>/checkpoints/last.ckpt; this script only imports it.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPACE = os.path.join(ROOT, "models", "yourmt3", "space")
CHECKPOINT = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops@last.ckpt"
BSZ = 2                        # segments per forward pass: 8 (the Space's) put the MoE model at 6+ GB and, next to the
                               # app's own CUDA context on an 8 GB card, into paging (18 min for 100 s of audio, 2026-09-09)
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5",
        "-nl", "26", "-enc", "perceiver-tf", "-sqr", "1", "-ff", "moe",
        "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope",
        "-rp", "1", "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "16"]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    wav, out = sys.argv[1], sys.argv[2]
    cpu = "--cpu" in sys.argv
    os.chdir(SPACE)                                  # the config's relative "amt/logs"
    sys.path.insert(0, os.path.join(SPACE, "amt", "src"))
    sys.path.insert(0, SPACE)
    import torch
    from collections import Counter
    from model_helper import load_model_checkpoint
    from utils.audio import slice_padded_array
    from utils.note2event import mix_notes
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    import torchaudio
    device = "cpu" if (cpu or not torch.cuda.is_available()) else "cuda"
    args = list(ARGS)
    if device == "cpu":
        args[-1] = "32"
    t0 = time.time()
    model = load_model_checkpoint(args=args, device="cpu")
    model.to(device)
    load_s = time.time() - t0
    import soundfile as sf
    import numpy as np
    data, sr = sf.read(wav, dtype="float32")          # torchaudio.load wants torchcodec on torch >= 2.9
    mono = data.mean(axis=1) if data.ndim == 2 else data
    audio = torch.from_numpy(np.ascontiguousarray(mono)).unsqueeze(0)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    segs = slice_padded_array(audio, model.audio_cfg["input_frames"], model.audio_cfg["input_frames"])
    segs = torch.from_numpy(segs.astype("float32")).to(device).unsqueeze(1)
    t1 = time.time()
    pred_token_arr, _ = model.inference_file(bsz=BSZ, audio_segments=segs)
    n_ch = model.task_manager.num_decoding_channels
    starts = [model.audio_cfg["input_frames"] * i / model.audio_cfg["sample_rate"] for i in range(segs.shape[0])]
    per_ch = []
    for ch in range(n_ch):
        arr_ch = [arr[:, ch, :] for arr in pred_token_arr]
        zipped, _events, _err = model.task_manager.detokenize_list_batches(arr_ch, starts, return_events=True)
        notes_ch, _cnt = merge_zipped_note_events_and_ties_to_notes(zipped)
        per_ch.append(notes_ch)
    notes = mix_notes(per_ch)
    infer_s = time.time() - t1
    rows = [{"onset": round(float(n.onset), 4), "offset": round(float(n.offset), 4), "pitch": int(n.pitch),
             "program": int(n.program), "is_drum": bool(n.is_drum), "velocity": int(n.velocity)} for n in notes]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"notes": rows, "seconds": round(infer_s, 2), "load_seconds": round(load_s, 2), "device": device,
                   "programs": dict(Counter(r["program"] for r in rows if not r["is_drum"]))}, fh)
    print(f"ymt3: {len(rows)} notes, {len(set(r['program'] for r in rows))} programs, inference {infer_s:.1f} s on {device} (load {load_s:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
