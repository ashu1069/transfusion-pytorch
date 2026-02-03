"""
Inference script for trained audio model.
Run: uv run --extra audio inference_audio.py --checkpoint results_audio/final.pt
"""

import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from transfusion_pytorch import Transfusion

# Config (must match training)
SAMPLE_RATE, N_MELS, HOP_LENGTH, LATENT_DIM = 24000, 100, 256, 256
MAX_FRAMES = int(4.0 * SAMPLE_RATE / HOP_LENGTH) + 1


class MelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True,
        )
        self.proj = nn.Sequential(
            nn.Conv1d(N_MELS, LATENT_DIM * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1),
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        m = self.mel(x)
        m = torch.log(m.clamp(min=1e-5))
        m = (m - m.mean(dim=(1, 2), keepdim=True)) / (
            m.std(dim=(1, 2), keepdim=True) + 1e-5
        )
        return self.proj(m).squeeze(0) if x.shape[0] == 1 else self.proj(m)


class MelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, N_MELS, 3, padding=1),
        )

    def forward(self, x):
        return self.proj(x)


def mel_to_audio(mel):
    basis = torchaudio.functional.melscale_fbanks(
        513, 0, SAMPLE_RATE / 2, N_MELS, SAMPLE_RATE
    ).to(mel.device)
    spec = torch.matmul(torch.linalg.pinv(basis.T), torch.exp(mel * 4)).clamp(min=1e-5)
    return torchaudio.transforms.GriffinLim(
        1024, hop_length=HOP_LENGTH, power=1.0, n_iter=64
    ).to(mel.device)(spec)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", required=True)
    parser.add_argument("--output", "-o", default="./generated")
    parser.add_argument("--num", "-n", type=int, default=4)
    parser.add_argument("--duration", "-d", type=float, default=None)
    parser.add_argument("--steps", "-s", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Transfusion(
        num_text_tokens=0,
        dim_latent=LATENT_DIM,
        channel_first_latent=True,
        modality_default_shape=(MAX_FRAMES,),
        modality_encoder=MelEncoder(),
        modality_decoder=MelDecoder(),
        add_pos_emb=True,
        modality_num_dim=1,
        velocity_consistency_loss_weight=0.1,
        model_output_clean=True,
        transformer=dict(dim=384, depth=12, dim_head=64, heads=6, attn_laser=True),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("ema", ckpt.get("model")))
    model.eval()
    print(f"Loaded {args.checkpoint}")

    out = Path(args.output)
    out.mkdir(exist_ok=True, parents=True)

    frames = (
        int(args.duration * SAMPLE_RATE / HOP_LENGTH) + 1 if args.duration else None
    )

    with torch.no_grad():
        kwargs = {"batch_size": args.num, "modality_steps": args.steps}
        if frames:
            kwargs["fixed_modality_shape"] = (frames,)
        mels = model.generate_modality_only(**kwargs)

    for i, mel in enumerate(mels):
        try:
            audio = mel_to_audio(mel)
            audio = audio / audio.abs().max() * 0.95
            torchaudio.save(
                str(out / f"sample_{i}.wav"), audio.cpu().unsqueeze(0), SAMPLE_RATE
            )
            print(f"Saved {out}/sample_{i}.wav")
        except Exception as e:
            print(f"Failed {i}: {e}")


if __name__ == "__main__":
    main()
