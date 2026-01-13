"""
Train Transfusion on audio-only generation using LibriTTS dataset.
Run: uv run --extra audio train_audio_only.py
"""

from pathlib import Path
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transfusion_pytorch import Transfusion
import wandb

# === Config ===
SAMPLE_RATE, N_MELS, HOP_LENGTH, LATENT_DIM = 16000, 80, 256, 128
MIN_DURATION, MAX_DURATION = 0.5, 4.0
BATCH_SIZE, STEPS, LR, ACCUM = 4, 50_000, 1e-4, 2
SAMPLE_EVERY, CKPT_EVERY = 1000, 5000
DATASET_SPLIT = "train.clean.100"
USE_WANDB = wandb is not None

results = Path('./results_audio')
results.mkdir(exist_ok=True, parents=True)

def get_frames(dur): 
    return int(dur * SAMPLE_RATE / HOP_LENGTH) + 1

MAX_FRAMES = get_frames(MAX_DURATION)


# === Encoder/Decoder ===
class MelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS, normalized=True)
        self.proj = nn.Sequential(
            nn.Conv1d(N_MELS, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        m = self.mel(x)
        m = torch.log(m.clamp(min=1e-5))
        m = (m - m.mean(dim=(1,2), keepdim=True)) / (m.std(dim=(1,2), keepdim=True) + 1e-5)
        return self.proj(m).squeeze(0) if x.shape[0] == 1 else self.proj(m)


class MelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, N_MELS, 3, padding=1))

    def forward(self, x): return self.proj(x)


# === Dataset ===
class LibriTTS(Dataset):
    def __init__(self, split=DATASET_SPLIT):
        self.ds = load_dataset("mythicinfinity/libritts", "clean", split=split)
        self.resample = torchaudio.transforms.Resample(24000, SAMPLE_RATE)
        min_s, max_s = int(MIN_DURATION * 24000), int(MAX_DURATION * 24000)
        self.idx = [i for i, s in enumerate(self.ds) if min_s <= len(s['audio']['array']) <= max_s]
        print(f"Dataset: {len(self.idx)} samples")

    def __len__(self): return len(self.idx)
    
    def __getitem__(self, i):
        wav = torch.tensor(self.ds[self.idx[i]]['audio']['array'], dtype=torch.float32)
        wav = self.resample(wav)
        if wav.abs().max() > 0: wav = wav / wav.abs().max()
        return [wav]


# === Model ===
model = Transfusion(
    num_text_tokens=0, dim_latent=LATENT_DIM, channel_first_latent=True,
    modality_default_shape=(MAX_FRAMES,), modality_encoder=MelEncoder(),
    modality_decoder=MelDecoder(), add_pos_emb=True, modality_num_dim=1,
    velocity_consistency_loss_weight=0.1, model_output_clean=True,
    transformer=dict(dim=384, depth=12, dim_head=64, heads=6, attn_laser=True),
)
ema = model.create_ema(0.995)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


# === Helpers ===
def cycle(dl):
    while True: yield from dl

def mel_to_audio(mel):
    """Griffin-Lim mel to audio."""
    basis = torchaudio.functional.melscale_fbanks(513, 0, SAMPLE_RATE/2, N_MELS, SAMPLE_RATE).to(mel.device)
    spec = torch.matmul(torch.linalg.pinv(basis.T), torch.exp(mel * 4)).clamp(min=1e-5)
    return torchaudio.transforms.GriffinLim(1024, hop_length=HOP_LENGTH, power=1.0, n_iter=64).to(mel.device)(spec)

def save_audio(audio, path):
    audio = audio.cpu()
    if audio.abs().max() > 0: audio = audio / audio.abs().max() * 0.95
    torchaudio.save(str(path), audio.unsqueeze(0) if audio.dim() == 1 else audio, SAMPLE_RATE)


# === Training ===
dataset = LibriTTS()
dl = cycle(model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True))
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, ema = model.to(device), ema.to(device)

if USE_WANDB:
    wandb.init(project="transfusion-audio", config={
        "steps": STEPS, "batch": BATCH_SIZE * ACCUM, "lr": LR, 
        "params": sum(p.numel() for p in model.parameters())})

print(f"Training on {device} for {STEPS} steps...")

for step in range(1, STEPS + 1):
    model.train()
    loss_acc = 0
    for _ in range(ACCUM):
        batch = [[x.to(device) for x in s] for s in next(dl)]
        loss = model(batch, velocity_consistency_ema_model=ema) / ACCUM
        loss.backward()
        loss_acc += loss.item()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(); ema.update()
    
    if step % 100 == 0:
        print(f'[{step}/{STEPS}] loss={loss_acc:.4f} lr={sched.get_last_lr()[0]:.2e}')
        if USE_WANDB: wandb.log({"loss": loss_acc, "lr": sched.get_last_lr()[0]}, step=step)
    
    if step % SAMPLE_EVERY == 0:
        model.eval()
        with torch.no_grad():
            mel = ema.generate_modality_only(batch_size=1, modality_steps=64)[0]
        try:
            save_audio(mel_to_audio(mel), results / f'audio_{step}.wav')
            if USE_WANDB: wandb.log({"audio": wandb.Audio(str(results / f'audio_{step}.wav'), sample_rate=SAMPLE_RATE)}, step=step)
        except Exception as e:
            print(f"Audio failed: {e}")
    
    if step % CKPT_EVERY == 0:
        torch.save({'step': step, 'model': model.state_dict(), 'ema': ema.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict()}, results / f'ckpt_{step}.pt')

# Final
torch.save({'model': model.state_dict(), 'ema': ema.state_dict()}, results / 'final.pt')
model.eval()
with torch.no_grad():
    for i, mel in enumerate(ema.generate_modality_only(batch_size=4, modality_steps=128)):
        try: save_audio(mel_to_audio(mel), results / f'final_{i}.wav')
        except: pass

if USE_WANDB: wandb.finish()
print(f"Done! Results in {results}")
