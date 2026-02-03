"""
Train Transfusion on audio-only generation using LibriTTS dataset.
Run: uv run train_audio_only.py
"""

import io
from pathlib import Path

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import wandb
from datasets import Audio, load_dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
from transfusion_pytorch import Transfusion
from vocos import Vocos

# === Config ===
SAMPLE_RATE = 24000
N_MELS = 100
HOP_LENGTH = 256
LATENT_DIM = 256
MIN_DURATION = 0.5
MAX_DURATION = 4.0
MAX_FRAMES = int(MAX_DURATION * SAMPLE_RATE / HOP_LENGTH) + 1

# Mel normalization constants (computed from Vocos statistics)
MEL_MEAN = 3.88
MEL_STD = 1.29

# Test mode toggle
TEST_MODE = False

if TEST_MODE:
    BATCH_SIZE, STEPS, LR, ACCUM = 2, 100, 1e-4, 1
    SAMPLE_EVERY, CKPT_EVERY = 50, 100
    DATASET_SPLIT = "dev.clean"
    MAX_SAMPLES = 50
    USE_WANDB = False
else:
    BATCH_SIZE, STEPS, LR, ACCUM = 4, 50_000, 5e-5, 4
    SAMPLE_EVERY, CKPT_EVERY = 1000, 5000
    DATASET_SPLIT = "train.clean.100"
    MAX_SAMPLES = None
    USE_WANDB = True

RESULTS_DIR = Path("./results_audio")
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# === Mel Spectrogram ===
def extract_mel(waveform):
    """Extract Vocos-compatible log mel spectrogram."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, win_length=1024,
        hop_length=HOP_LENGTH, n_mels=N_MELS, power=1, center=True,
    ).to(waveform.device)
    
    if waveform.dim() == 3:
        waveform = waveform.squeeze(1)
    mel = mel_transform(waveform)
    return mel.clamp(min=1e-5).log()


# === Encoder/Decoder ===
class MelEncoder(nn.Module):
    """Encodes waveform to latent space via mel spectrogram."""
    
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(N_MELS, LATENT_DIM, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1),
        )

    def forward(self, wav):
        mel = extract_mel(wav)
        mel = (mel - MEL_MEAN) / MEL_STD  # Normalize
        out = self.proj(mel)
        
        # Pad/trim to fixed length
        if out.shape[-1] < MAX_FRAMES:
            out = torch.nn.functional.pad(out, (0, MAX_FRAMES - out.shape[-1]))
        elif out.shape[-1] > MAX_FRAMES:
            out = out[..., :MAX_FRAMES]
        return out


class MelDecoder(nn.Module):
    """Decodes latent back to log mel spectrogram."""
    
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM, N_MELS, 3, padding=1),
        )

    def forward(self, x):
        out = self.proj(x)
        return out * MEL_STD + MEL_MEAN  # Denormalize


# === Dataset ===
class LibriTTS(Dataset):
    """Lazy-loading LibriTTS dataset."""
    
    def __init__(self, split=DATASET_SPLIT, max_samples=MAX_SAMPLES):
        print(f"Loading LibriTTS {split}...")
        self.target_len = int(MAX_DURATION * SAMPLE_RATE)
        
        ds = load_dataset("mythicinfinity/libritts", "clean", split=split)
        ds = ds.cast_column("audio", Audio(decode=False))
        self.ds = ds
        
        self.indices = list(range(min(len(ds), max_samples or len(ds))))
        print(f"Dataset: {len(self.indices)} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        audio_bytes = self.ds[self.indices[idx]]["audio"]["bytes"]
        wav, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = torch.from_numpy(wav)
        
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        
        # Pad/trim to fixed length
        if wav.shape[0] < self.target_len:
            wav = torch.nn.functional.pad(wav, (0, self.target_len - wav.shape[0]))
        else:
            wav = wav[:self.target_len]
        
        return [wav.clamp(-1.0, 1.0)]


# === Model ===
def create_model():
    return Transfusion(
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
        transformer=dict(dim=768, depth=18, dim_head=64, heads=12, attn_laser=True),
    )


# === Inference Helpers ===
_vocoder = None

def get_vocoder(device):
    global _vocoder
    if _vocoder is None:
        _vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return _vocoder.to(device)


def mel_to_audio(mel):
    """Convert mel spectrogram to audio using Vocos."""
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    with torch.no_grad():
        audio = get_vocoder(mel.device).decode(mel)
    return audio.squeeze(0)


def save_audio(audio, path):
    audio = audio.cpu().numpy()
    if abs(audio).max() > 0:
        audio = audio / abs(audio).max() * 0.95
    sf.write(str(path), audio, SAMPLE_RATE)


def cycle(dl):
    while True:
        yield from dl


# === Training ===
if __name__ == "__main__":
    model = create_model()
    ema = model.create_ema(0.995)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    dataset = LibriTTS()
    dl = cycle(model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True))
    opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ema = model.to(device), ema.to(device)

    # Resume from checkpoint
    start_step = 1
    ckpts = sorted(RESULTS_DIR.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if ckpts:
        print(f"Resuming from {ckpts[-1]}...")
        ckpt = torch.load(ckpts[-1], map_location=device)
        model.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"] + 1

    if USE_WANDB:
        wandb.init(project="transfusion-audio", config={
            "steps": STEPS, "batch": BATCH_SIZE * ACCUM, "lr": LR,
            "params": sum(p.numel() for p in model.parameters())
        }, resume="allow")

    print(f"Training on {device} for {STEPS} steps (starting from {start_step})...")

    for step in range(start_step, STEPS + 1):
        model.train()
        loss_acc = 0
        
        for _ in range(ACCUM):
            batch = [[x.to(device) for x in s] for s in next(dl)]
            loss = model(batch, velocity_consistency_ema_model=ema) / ACCUM
            loss.backward()
            loss_acc += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        sched.step()
        ema.update()

        if step % (10 if TEST_MODE else 100) == 0:
            print(f"[{step}/{STEPS}] loss={loss_acc:.4f} lr={sched.get_last_lr()[0]:.2e}")
            if USE_WANDB:
                wandb.log({"loss": loss_acc, "lr": sched.get_last_lr()[0]}, step=step)

        if step % SAMPLE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                mel = ema.generate_modality_only(batch_size=1, modality_steps=64)[0]
            try:
                save_audio(mel_to_audio(mel), RESULTS_DIR / f"audio_{step}.wav")
                if USE_WANDB:
                    wandb.log({"audio": wandb.Audio(
                        str(RESULTS_DIR / f"audio_{step}.wav"), sample_rate=SAMPLE_RATE
                    )}, step=step)
            except Exception as e:
                print(f"Audio failed: {e}")

        if step % CKPT_EVERY == 0:
            torch.save({
                "step": step, "model": model.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "sched": sched.state_dict()
            }, RESULTS_DIR / f"ckpt_{step}.pt")

    # Save final model
    torch.save({"model": model.state_dict(), "ema": ema.state_dict()}, RESULTS_DIR / "final.pt")
    
    # Generate final samples
    model.eval()
    with torch.no_grad():
        for i, mel in enumerate(ema.generate_modality_only(batch_size=4, modality_steps=128)):
            try:
                save_audio(mel_to_audio(mel), RESULTS_DIR / f"final_{i}.wav")
            except:
                pass

    if USE_WANDB:
        wandb.finish()
    print(f"Done! Results in {RESULTS_DIR}")
