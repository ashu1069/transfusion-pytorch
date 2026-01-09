"""
Train Transfusion on audio-only generation using LibriTTS dataset.
Dataset: https://huggingface.co/datasets/mythicinfinity/libritts
"""

from shutil import rmtree
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from datasets import load_dataset

from transfusion_pytorch import Transfusion

# Configuration

# Audio
SAMPLE_RATE = 16000 # 16kHz 
N_MELS = 64 # divide the frequency range into 64 bins; may lose fine-grained frequency information. If we try to invert this back to audio, it might sound slightly robotic or smeary (configurable).
HOP_LENGTH = 256 # overlap between consecutive mel bins, 256/16000 = 16 ms.
LATENT_DIM = 64 # latent dimension
MIN_AUDIO_DURATION = 0.5 # minimum audio duration
MAX_AUDIO_DURATION = 4.0 # maximum audio duration
VARIABLE_LENGTH = True # whether to use variable length audio

'''
Maximum number of audio tokens the model will see for a single clip.
n_frames = samples / hop length = duration * sample_rate / hop_length = (4s * 16000Hz) / 256 = 250 tokens.
We're feeding the model sequences of up to 250 audio tokens at a time. Each audio token represents 16 ms of audio,
and has a feature dimension of 64 (n_mels).
'''

# Training
BATCH_SIZE = 8
NUM_TRAIN_STEPS = 20_000
SAMPLE_EVERY = 500
LEARNING_RATE = 3e-4

# Dataset (configs: "dev", "clean", "other", "all")
# Splits: "dev.clean", "dev.other", "test.clean", "test.other",
#         "train.clean.100", "train.clean.360", "train.other.500"
DATASET_CONFIG = "dev"
DATASET_SPLIT = "dev.clean"
MAX_DATASET_SAMPLES = 1000  # Set to None for full dataset


rmtree('./results_audio', ignore_errors=True)
results_folder = Path('./results_audio')
results_folder.mkdir(exist_ok=True, parents=True)


def _get_mel_frames(duration):
    """Compute mel frames for a given duration."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS, normalized=True
    )
    num_samples = int(SAMPLE_RATE * duration)
    return mel_transform(torch.zeros(num_samples)).shape[-1]


MAX_AUDIO_FRAMES = _get_mel_frames(MAX_AUDIO_DURATION)
MIN_AUDIO_FRAMES = _get_mel_frames(MIN_AUDIO_DURATION)

print(f"Audio config: {N_MELS} mels, {MIN_AUDIO_FRAMES}-{MAX_AUDIO_FRAMES} frames (variable={VARIABLE_LENGTH})")


class MelSpectrogramEncoder(nn.Module):
    """Encodes raw waveform to mel spectrogram latent representation."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=1024, hop_length=HOP_LENGTH, latent_dim=LATENT_DIM):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, normalized=True
        )
        self.proj = nn.Sequential(
            nn.Conv1d(n_mels, latent_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
        )

    def forward(self, waveform):
        squeeze_batch = waveform.dim() == 1
        if squeeze_batch:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-5))
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        latent = self.proj(mel)

        if squeeze_batch:
            latent = latent.squeeze(0)
        return latent


class MelSpectrogramDecoder(nn.Module):
    """Decodes latent representation back to mel spectrogram."""

    def __init__(self, n_mels=N_MELS, latent_dim=LATENT_DIM):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim * 2, n_mels, kernel_size=3, padding=1),
        )

    def forward(self, latent):
        return self.proj(latent)


model = Transfusion(
    num_text_tokens=0, # audio only model, no text tokens
    dim_latent=LATENT_DIM,
    channel_first_latent=True,
    modality_default_shape=(MAX_AUDIO_FRAMES,),
    modality_encoder=MelSpectrogramEncoder(),
    modality_decoder=MelSpectrogramDecoder(),
    add_pos_emb=True,
    modality_num_dim=1,
    velocity_consistency_loss_weight=0.1,
    model_output_clean=True,
    transformer=dict(dim=128, depth=6, dim_head=32, heads=4, attn_laser=False),
)

ema_model = model.create_ema(beta=0.995)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dataset wrapper for LibriTTS dataset from Hugging Face.

class LibriTTSDataset(Dataset):
    """Wrapper for LibriTTS dataset from Hugging Face."""

    def __init__(self, config="dev", split="dev.clean", target_sample_rate=SAMPLE_RATE,
                 min_duration=MIN_AUDIO_DURATION, max_duration=MAX_AUDIO_DURATION,
                 variable_length=VARIABLE_LENGTH, max_samples=None):
        self.target_sample_rate = target_sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.variable_length = variable_length
        self.source_sample_rate = 24000

        print(f"Loading LibriTTS dataset (config='{config}', split='{split}')...")
        self.hf_dataset = load_dataset("mythicinfinity/libritts", config, split=split)

        self.resampler = torchaudio.transforms.Resample(self.source_sample_rate, target_sample_rate) \
            if self.source_sample_rate != target_sample_rate else None

        min_samples = int(min_duration * self.source_sample_rate)
        max_samples_audio = int(max_duration * self.source_sample_rate)

        print(f"Filtering samples by duration ({min_duration}s - {max_duration}s)...")
        self.valid_indices = [
            i for i, sample in enumerate(self.hf_dataset)
            if min_samples <= len(sample['audio']['array']) <= max_samples_audio
        ]

        if max_samples is not None and len(self.valid_indices) > max_samples:
            self.valid_indices = self.valid_indices[:max_samples]

        print(f"Dataset loaded: {len(self.valid_indices)} samples (filtered from {len(self.hf_dataset)} total)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        sample = self.hf_dataset[self.valid_indices[idx]]
        waveform = torch.tensor(sample['audio']['array'], dtype=torch.float32)

        if self.resampler is not None:
            waveform = self.resampler(waveform)

        if waveform.abs().max() > 0:
            waveform = waveform / waveform.abs().max()

        if not self.variable_length:
            max_samples = int(self.max_duration * self.target_sample_rate)
            if len(waveform) > max_samples:
                start = torch.randint(0, len(waveform) - max_samples, (1,)).item()
                waveform = waveform[start:start + max_samples]
            elif len(waveform) < max_samples:
                waveform = torch.nn.functional.pad(waveform, (0, max_samples - len(waveform)))
            return waveform

        return [waveform]


# Helper functions for training and sampling.


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def save_mel_spectrogram(mel, path, title="Generated Mel Spectrogram"):
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.imshow(mel.detach().cpu().numpy(), aspect='auto', origin='lower', cmap='magma')
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Bin')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()
    except ImportError:
        print("matplotlib not installed, skipping visualization")

# training setup

print("\n" + "=" * 50)
print("Setting up LibriTTS dataset...")
print("=" * 50)

dataset = LibriTTSDataset(
    config=DATASET_CONFIG,
    split=DATASET_SPLIT,
    target_sample_rate=SAMPLE_RATE,
    min_duration=MIN_AUDIO_DURATION,
    max_duration=MAX_AUDIO_DURATION,
    variable_length=VARIABLE_LENGTH,
    max_samples=MAX_DATASET_SAMPLES,
)

if VARIABLE_LENGTH:
    dataloader = model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
else:
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

iter_dl = cycle(dataloader)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ema_model.to(device)

print(f"Training on: {device}")
print(f"Audio frames: {MIN_AUDIO_FRAMES}-{MAX_AUDIO_FRAMES} (variable={VARIABLE_LENGTH})")
print(f"Starting training for {NUM_TRAIN_STEPS} steps...")

for step in range(1, NUM_TRAIN_STEPS + 1):
    model.train()
    batch = next(iter_dl)

    if VARIABLE_LENGTH:
        batch = [[item.to(device) for item in sample] for sample in batch]
    else:
        batch = batch.to(device)

    loss = model(batch, velocity_consistency_ema_model=ema_model)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    optimizer.zero_grad()
    ema_model.update()

    if step % 50 == 0:
        print(f'Step {step}/{NUM_TRAIN_STEPS}: loss={loss.item():.4f}')

    if step % SAMPLE_EVERY == 0:
        print(f"\n--- Generating sample at step {step} ---")
        with torch.no_grad():
            generated_mel = ema_model.generate_modality_only(batch_size=4, modality_steps=32)

        save_mel_spectrogram(generated_mel[0], results_folder / f'mel_step_{step}.png',
                             title=f'Generated Mel Spectrogram - Step {step}')

        if generated_mel.shape[0] >= 4:
            grid = torch.cat([generated_mel[i] for i in range(4)], dim=1)
            save_mel_spectrogram(grid, results_folder / f'mel_grid_step_{step}.png',
                                 title=f'Generated Samples Grid - Step {step}')

        print(f"Saved samples to {results_folder}\n")

print("\n" + "=" * 50)
print("Training complete!")
print(f"Results saved to: {results_folder}")
print("=" * 50)

# Final generation
print("\nGenerating final high-quality samples...")

with torch.no_grad():
    final_samples = ema_model.generate_modality_only(batch_size=4, modality_steps=64)

for i, mel in enumerate(final_samples):
    save_mel_spectrogram(mel, results_folder / f'final_sample_{i}_max_length.png',
                         title=f'Final Sample {i} ({MAX_AUDIO_FRAMES} frames)')

print(f"Saved {len(final_samples)} samples at max length ({MAX_AUDIO_FRAMES} frames)")

if VARIABLE_LENGTH:
    print("\nGenerating samples at different durations...")
    for duration in [0.5, 1.0, 2.0, 3.0]:
        target_frames = max(MIN_AUDIO_FRAMES, min(MAX_AUDIO_FRAMES, _get_mel_frames(duration)))
        with torch.no_grad():
            sample = ema_model.generate_modality_only(batch_size=1, fixed_modality_shape=(target_frames,), modality_steps=64)
        save_mel_spectrogram(sample[0], results_folder / f'final_sample_{duration:.1f}s.png',
                             title=f'Generated Audio ({duration:.1f}s, {target_frames} frames)')
        print(f"  Generated {duration:.1f}s audio ({target_frames} frames)")

print(f"\nAll results saved to: {results_folder}")
