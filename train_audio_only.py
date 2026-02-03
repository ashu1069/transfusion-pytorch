"""
Train Transfusion on audio-only generation using LibriTTS dataset.
Run: torchrun --nproc_per_node=2 train_audio_only.py
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# torchaudio not needed - using Vocos feature extractor for mel
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transfusion_pytorch import Transfusion
from vocos import Vocos
import wandb
import torchaudio

# === Config ===
# Config matched to Vocos pretrained vocoder (charactr/vocos-mel-24khz)
SAMPLE_RATE, N_MELS, HOP_LENGTH = 24000, 100, 256
LATENT_DIM = N_MELS  # Encoder outputs mel, decoder (Vocos) expects mel
MIN_DURATION, MAX_DURATION = 0.5, 4.0
# === Test mode (set to False for full training on GPU server) ===
TEST_MODE = False  # <-- Set to False for GPU server

if TEST_MODE:
    BATCH_SIZE, STEPS, LR, ACCUM = 32, 100, 1e-4, 1
    SAMPLE_EVERY, CKPT_EVERY = 50, 100
    DATASET_SPLIT = "dev.clean"  # smallest split
    MAX_SAMPLES = 50  # only use 50 samples for quick testing
    USE_WANDB = False
else:
    # === GPU Server Settings (scaled model ~150M params) ===
    BATCH_SIZE = 4  # Reduced for larger model (A100: 8-16, V100: 4-8)
    STEPS = 50_000  # Total training steps
    LR = 5e-5  # Lower LR for larger model
    ACCUM = 4  # Gradient accumulation (effective batch = 16)
    SAMPLE_EVERY = 1000  # Generate sample audio every N steps
    CKPT_EVERY = 5000  # Save checkpoint every N steps
    DATASET_SPLIT = "train.clean.100"  # Full training set (~29k samples)
    MAX_SAMPLES = None  # Use all samples
    USE_WANDB = wandb is not None  # Enable wandb logging

# === Distributed Setup ===
def setup_distributed():
    """Initialize distributed training if available."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    return 0, 1, 0, False

RANK, WORLD_SIZE, LOCAL_RANK, DISTRIBUTED = setup_distributed()
IS_MAIN = RANK == 0

results = Path("./results_audio")
if IS_MAIN:
    results.mkdir(exist_ok=True, parents=True)


def get_frames(dur):
    return int(dur * SAMPLE_RATE / HOP_LENGTH) + 1


MAX_FRAMES = get_frames(MAX_DURATION)


# === Encoder/Decoder using Vocos-compatible mel format ===

def get_vocos_mel_spectrogram(
    waveform,
    n_fft=1024,
    n_mel_channels=100,
    target_sample_rate=24000,
    hop_length=256,
    win_length=1024,
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mel_channels,
        power=1,
        center=True,
        normalized=False,
        norm=None,
    ).to(waveform.device)
    if len(waveform.shape) == 3:
        waveform = waveform.squeeze(1)  # 'b 1 nw -> b nw'

    assert len(waveform.shape) == 2

    mel = mel_stft(waveform)
    mel = mel.clamp(min=1e-5).log()
    return mel


class MelEncoder(nn.Module):
    def __init__(
        self,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=100,
        target_sample_rate=24_000,
        mel_spec_type="vocos",
    ):
        super().__init__()
        assert mel_spec_type in ["vocos", "bigvgan"], print(
            "We only support two extract mel backend: vocos or bigvgan"
        )

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.target_sample_rate = target_sample_rate

        if mel_spec_type == "vocos":
            self.extractor = get_vocos_mel_spectrogram

        self.register_buffer("dummy", torch.tensor(0), persistent=False)

    def forward(self, wav):
        if self.dummy.device != wav.device:
            self.to(wav.device)

        mel = self.extractor(
            waveform=wav,
            n_fft=self.n_fft,
            n_mel_channels=self.n_mel_channels,
            target_sample_rate=self.target_sample_rate,
            hop_length=self.hop_length,
            win_length=self.win_length,
        )

        return mel


class MelDecoder(nn.Module):
    """Decodes mel spectrogram to audio using pretrained Vocos vocoder.
    
    Following F5-TTS approach: https://github.com/SWivid/F5-TTS
    Vocos.decode() converts mel spectrogram directly to waveform.
    """

    def __init__(self):
        super().__init__()
        # Load pretrained Vocos vocoder
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.vocoder.eval()
        # Freeze vocoder weights - we don't train it
        for param in self.vocoder.parameters():
            param.requires_grad = False

    def forward(self, mel):
        # mel shape: (batch, n_mels, time) - already in Vocos format
        # Vocos.decode expects log mel spectrogram
        with torch.no_grad():
            audio = self.vocoder.decode(mel)
        # audio shape: (batch, time)
        return audio


# === Dataset ===
import soundfile as sf
import io
from datasets import Audio


class LibriTTS(Dataset):
    def __init__(self, split=DATASET_SPLIT, max_samples=MAX_SAMPLES):
        if IS_MAIN:
            print(f"Loading LibriTTS {split}...")
        min_s, max_s = int(MIN_DURATION * SAMPLE_RATE), int(MAX_DURATION * SAMPLE_RATE)

        # Load dataset and DISABLE audio decoding
        ds = load_dataset(
            "mythicinfinity/libritts", "clean", split=split, cache_dir="./data_libritts"
        )
        ds = ds.cast_column("audio", Audio(decode=False))  # Keep as raw bytes

        # Filter by duration, decode with soundfile
        self.samples = []
        if IS_MAIN:
            print("Filtering and decoding audio samples...")
        for i in range(len(ds)):
            try:
                # Get raw audio bytes (not decoded)
                audio_data = ds[i]["audio"]
                audio_bytes = audio_data["bytes"]
                arr, sr = sf.read(io.BytesIO(audio_bytes))
                if arr.ndim > 1:  # stereo to mono
                    arr = arr.mean(axis=1)
                if min_s <= len(arr) <= max_s:
                    self.samples.append(arr.astype("float32"))
                if max_samples and len(self.samples) >= max_samples:
                    break
            except Exception as e:
                continue  # Skip problematic files
            if IS_MAIN and (i + 1) % 1000 == 0:
                print(f"  Processed {i+1} files, kept {len(self.samples)} samples")

        if IS_MAIN:
            print(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        wav = torch.from_numpy(self.samples[i])
        # Normalize audio
        # if wav.abs().max() > 0:
        #     wav = wav / wav.abs().max()

        # Pad/trim to fixed length so all mel spectrograms have MAX_FRAMES
        target_len = int(MAX_DURATION * SAMPLE_RATE)
        if wav.shape[0] < target_len:
            wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[0]))
        else:
            wav = wav[:target_len]
        return [wav]


# === Model ===
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
    # Scaled up model (~150M params) for better audio quality
    transformer=dict(dim=768, depth=18, dim_head=64, heads=12, attn_laser=True),
)
ema = model.create_ema(0.995)
if IS_MAIN:
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")


# === Helpers ===
def cycle(dl):
    while True:
        yield from dl


def save_audio(audio, path):
    """Save audio tensor to file."""
    if audio.dim() > 1:
        audio = audio.squeeze()  # Remove batch dim if present
    audio = audio.cpu().numpy()
    if abs(audio).max() > 0:
        audio = audio / abs(audio).max() * 0.95
    sf.write(str(path), audio, SAMPLE_RATE)


# === Training ===
dataset = LibriTTS()

# Setup device and distributed training
if DISTRIBUTED:
    device = torch.device(f"cuda:{LOCAL_RANK}")
    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=RANK, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = None
    dataloader = model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dl = cycle(dataloader)
opt = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
sched = CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR * 0.1)

model, ema = model.to(device), ema.to(device)

# Wrap model with DDP for multi-GPU training
if DISTRIBUTED:
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    # Get the underlying model for EMA updates and generation
    model_module = model.module
else:
    model_module = model

# Resume from checkpoint if available
start_step = 1
ckpts = sorted(results.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
if ckpts:
    latest = ckpts[-1]
    if IS_MAIN:
        print(f"Resuming from {latest}...")
    ckpt = torch.load(latest, map_location=device)
    model_module.load_state_dict(ckpt["model"])
    ema.load_state_dict(ckpt["ema"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    start_step = ckpt["step"] + 1

if USE_WANDB and IS_MAIN:
    wandb.init(
        project="transfusion-audio",
        config={
            "steps": STEPS,
            "batch": BATCH_SIZE * ACCUM * WORLD_SIZE,  # Effective batch size across all GPUs
            "lr": LR,
            "params": sum(p.numel() for p in model_module.parameters()),
            "num_gpus": WORLD_SIZE,
        },
        resume="allow",
    )

if IS_MAIN:
    print(f"Training on {WORLD_SIZE} GPU(s) for {STEPS} steps (starting from step {start_step})...")

if DISTRIBUTED:
    dist.barrier()  # Synchronize all processes before training

for step in range(start_step, STEPS + 1):
    model.train()
    
    # Update sampler epoch for proper shuffling in distributed mode
    if DISTRIBUTED and sampler is not None:
        sampler.set_epoch(step)
    
    loss_acc = 0
    for _ in range(ACCUM):
        batch = [[x.to(device) for x in s] for s in next(dl)]
        loss = model(batch, velocity_consistency_ema_model=ema) / ACCUM
        loss.backward()
        loss_acc += loss.item()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    sched.step()
    opt.zero_grad()
    ema.update()

    if step % 10 == 0 and IS_MAIN:
        epoch = step * BATCH_SIZE * ACCUM * WORLD_SIZE / len(dataset)
        print(f"[Step {step}/{STEPS}] [Epoch {epoch:.2f}] loss={loss_acc:.4f} lr={sched.get_last_lr()[0]:.2e}")
        if USE_WANDB:
            wandb.log({"loss": loss_acc, "lr": sched.get_last_lr()[0], "epoch": epoch}, step=step)

    if step % SAMPLE_EVERY == 0 and IS_MAIN:
        model.eval()
        with torch.no_grad():
            # generate_modality_only returns decoded output (audio from Vocos)
            audio = ema.generate_modality_only(batch_size=1, modality_steps=64)[0]
        try:
            save_audio(audio, results / f"audio_{step}.wav")
            if USE_WANDB:
                wandb.log(
                    {
                        "audio": wandb.Audio(
                            str(results / f"audio_{step}.wav"), sample_rate=SAMPLE_RATE
                        )
                    },
                    step=step,
                )
        except Exception as e:
            print(f"Audio failed: {e}")

    if step % CKPT_EVERY == 0 and IS_MAIN:
        torch.save(
            {
                "step": step,
                "model": model_module.state_dict(),
                "ema": ema.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
            },
            results / f"ckpt_{step}.pt",
        )
    
    # Synchronize after checkpoint saves
    if DISTRIBUTED and step % CKPT_EVERY == 0:
        dist.barrier()

# Final - only main process saves and generates
if IS_MAIN:
    torch.save({"model": model_module.state_dict(), "ema": ema.state_dict()}, results / "final.pt")
    model.eval()
    with torch.no_grad():
        # generate_modality_only returns decoded output (audio from Vocos)
        for i, audio in enumerate(
            ema.generate_modality_only(batch_size=4, modality_steps=128)
        ):
            try:
                save_audio(audio, results / f"final_{i}.wav")
            except:
                pass

    if USE_WANDB:
        wandb.finish()
    print(f"Done! Results in {results}")

# Cleanup distributed
if DISTRIBUTED:
    dist.destroy_process_group()
