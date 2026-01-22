"""
Train Transfusion on audio-only generation using LibriTTS dataset.
Run: uv run train_audio_only.py
"""

from pathlib import Path
import torch
import torch.nn as nn
# torchaudio not needed - using Vocos feature extractor for mel
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transfusion_pytorch import Transfusion
from vocos import Vocos
import wandb

# === Config ===
# Config matched to Vocos pretrained vocoder (charactr/vocos-mel-24khz)
SAMPLE_RATE, N_MELS, HOP_LENGTH, LATENT_DIM = 24000, 100, 256, 256
MIN_DURATION, MAX_DURATION = 0.5, 4.0
# === Test mode (set to False for full training on GPU server) ===
TEST_MODE = False  # <-- Set to False for GPU server

if TEST_MODE:
    BATCH_SIZE, STEPS, LR, ACCUM = 2, 100, 1e-4, 1
    SAMPLE_EVERY, CKPT_EVERY = 50, 100
    DATASET_SPLIT = "dev.clean"  # smallest split
    MAX_SAMPLES = 50  # only use 50 samples for quick testing
    USE_WANDB = False
else:
    # === GPU Server Settings (scaled model ~150M params) ===
    BATCH_SIZE = 4          # Reduced for larger model (A100: 8-16, V100: 4-8)
    STEPS = 50_000          # Total training steps
    LR = 5e-5               # Lower LR for larger model
    ACCUM = 4               # Gradient accumulation (effective batch = 16)
    SAMPLE_EVERY = 1000     # Generate sample audio every N steps
    CKPT_EVERY = 5000       # Save checkpoint every N steps
    DATASET_SPLIT = "train.clean.100"  # Full training set (~29k samples)
    MAX_SAMPLES = None      # Use all samples
    USE_WANDB = wandb is not None  # Enable wandb logging

results = Path('./results_audio')
results.mkdir(exist_ok=True, parents=True)

def get_frames(dur): 
    return int(dur * SAMPLE_RATE / HOP_LENGTH) + 1

MAX_FRAMES = get_frames(MAX_DURATION)


# === Encoder/Decoder using Vocos-compatible mel format ===
# Fixed normalization constants (computed from Vocos mel statistics)
MEL_MEAN = 3.88
MEL_STD = 1.29

class MelEncoder(nn.Module):
    """Encodes waveform to latent via Vocos-compatible log mel spectrogram."""
    def __init__(self):
        super().__init__()
        # Use Vocos's feature extractor for consistent mel format
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.fe = self.vocoder.feature_extractor
        # Freeze feature extractor - we just use it for mel extraction
        for p in self.fe.parameters():
            p.requires_grad = False
        # Project mel to latent
        self.proj = nn.Sequential(
            nn.Conv1d(N_MELS, LATENT_DIM, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1))

    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze: x = x.unsqueeze(0)
        # Use Vocos feature extractor (outputs log mel)
        with torch.no_grad():
            m = self.fe(x)  # Shape: (B, n_mels, T)
        # Fixed normalization for stable training (zero mean, unit variance)
        m = (m - MEL_MEAN) / MEL_STD
        out = self.proj(m)
        # Ensure exact output shape (pad/trim to MAX_FRAMES)
        if out.shape[-1] < MAX_FRAMES:
            out = torch.nn.functional.pad(out, (0, MAX_FRAMES - out.shape[-1]))
        elif out.shape[-1] > MAX_FRAMES:
            out = out[..., :MAX_FRAMES]
        return out.squeeze(0) if squeeze else out


class MelDecoder(nn.Module):
    """Decodes latent back to Vocos-compatible log mel spectrogram."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1), nn.GELU(),
            nn.Conv1d(LATENT_DIM, N_MELS, 3, padding=1))

    def forward(self, x):
        # Project to mel space (normalized)
        out = self.proj(x)
        # Denormalize back to Vocos mel range
        out = out * MEL_STD + MEL_MEAN
        return out


# === Dataset ===
import soundfile as sf
import io
from datasets import Audio

class LibriTTS(Dataset):
    def __init__(self, split=DATASET_SPLIT, max_samples=MAX_SAMPLES):
        print(f"Loading LibriTTS {split}...")
        min_s, max_s = int(MIN_DURATION * SAMPLE_RATE), int(MAX_DURATION * SAMPLE_RATE)
        
        # Load dataset and DISABLE audio decoding
        ds = load_dataset("mythicinfinity/libritts", "clean", split=split)
        ds = ds.cast_column("audio", Audio(decode=False))  # Keep as raw bytes
        
        # Filter by duration, decode with soundfile
        self.samples = []
        print("Filtering and decoding audio samples...")
        for i in range(len(ds)):
            try:
                # Get raw audio bytes (not decoded)
                audio_data = ds[i]['audio']
                audio_bytes = audio_data['bytes']
                arr, sr = sf.read(io.BytesIO(audio_bytes))
                if arr.ndim > 1:  # stereo to mono
                    arr = arr.mean(axis=1)
                if min_s <= len(arr) <= max_s:
                    self.samples.append(arr.astype('float32'))
                if max_samples and len(self.samples) >= max_samples:
                    break
            except Exception as e:
                continue  # Skip problematic files
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1} files, kept {len(self.samples)} samples")
        
        print(f"Dataset: {len(self.samples)} samples")

    def __len__(self): 
        return len(self.samples)
    
    def __getitem__(self, i):
        wav = torch.from_numpy(self.samples[i])
        # Normalize audio
        if wav.abs().max() > 0: wav = wav / wav.abs().max()
        
        # Pad/trim to fixed length so all mel spectrograms have MAX_FRAMES
        target_len = int(MAX_DURATION * SAMPLE_RATE)
        if wav.shape[0] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        else:
            wav = wav[:target_len]
        return [wav]


# === Model ===
model = Transfusion(
    num_text_tokens=0, dim_latent=LATENT_DIM, channel_first_latent=True,
    modality_default_shape=(MAX_FRAMES,), modality_encoder=MelEncoder(),
    modality_decoder=MelDecoder(), add_pos_emb=True, modality_num_dim=1,
    velocity_consistency_loss_weight=0.1, model_output_clean=True,
    # Scaled up model (~150M params) for better audio quality
    transformer=dict(dim=768, depth=18, dim_head=64, heads=12, attn_laser=True),
)
ema = model.create_ema(0.995)
print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


# === Helpers ===
def cycle(dl):
    while True: yield from dl

# Global vocoder for inference (loaded once)
_vocoder = None

def get_vocoder(device):
    global _vocoder
    if _vocoder is None:
        _vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    return _vocoder.to(device)

def mel_to_audio(mel):
    """Convert mel spectrogram to audio using Vocos neural vocoder."""
    # Vocos expects (B, C, T) - mel is (C, T) so add batch dim
    if mel.dim() == 2:
        mel = mel.unsqueeze(0)
    vocoder = get_vocoder(mel.device)
    with torch.no_grad():
        audio = vocoder.decode(mel)
    return audio.squeeze(0)  # Remove batch dim

def save_audio(audio, path):
    audio = audio.cpu().numpy()
    if abs(audio).max() > 0: audio = audio / abs(audio).max() * 0.95
    sf.write(str(path), audio, SAMPLE_RATE)


# === Training ===
dataset = LibriTTS()
dl = cycle(model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True))
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, ema = model.to(device), ema.to(device)

# Resume from checkpoint if available
start_step = 1
ckpts = sorted(results.glob('ckpt_*.pt'), key=lambda p: int(p.stem.split('_')[1]))
if ckpts:
    latest = ckpts[-1]
    print(f"Resuming from {latest}...")
    ckpt = torch.load(latest, map_location=device)
    model.load_state_dict(ckpt['model'])
    ema.load_state_dict(ckpt['ema'])
    opt.load_state_dict(ckpt['opt'])
    sched.load_state_dict(ckpt['sched'])
    start_step = ckpt['step'] + 1

if USE_WANDB:
    wandb.init(project="transfusion-audio", config={
        "steps": STEPS, "batch": BATCH_SIZE * ACCUM, "lr": LR, 
        "params": sum(p.numel() for p in model.parameters())}, resume="allow")

print(f"Training on {device} for {STEPS} steps (starting from step {start_step})...")

for step in range(start_step, STEPS + 1):
    model.train()
    loss_acc = 0
    for _ in range(ACCUM):
        batch = [[x.to(device) for x in s] for s in next(dl)]
        loss = model(batch, velocity_consistency_ema_model=ema) / ACCUM
        loss.backward()
        loss_acc += loss.item()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(); ema.update()
    
    if step % (10 if TEST_MODE else 100) == 0:
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
