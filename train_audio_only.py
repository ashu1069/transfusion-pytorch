"""
Train Transfusion on audio-only generation (e.g., music, speech)

This demonstrates how to add audio modality to Transfusion.
Uses mel spectrogram representation for simplicity.
"""

from shutil import rmtree
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from einops import rearrange

from transfusion_pytorch import Transfusion

# Clean and create results folder
rmtree('./results_audio', ignore_errors=True)
results_folder = Path('./results_audio')
results_folder.mkdir(exist_ok=True, parents=True)

# Sample configuration

SAMPLE_RATE = 16000       # 16kHz audio
N_MELS = 64               # Number of mel frequency bins
HOP_LENGTH = 256          # ~62.5 frames per second
LATENT_DIM = 64           # Latent dimension for transformer
AUDIO_DURATION = 2.0      # Seconds of audio to generate
BATCH_SIZE = 16
NUM_TRAIN_STEPS = 20_000
SAMPLE_EVERY = 500

# Calculate audio shape dynamically to match actual MelSpectrogram output
def _get_mel_frames():
    """Compute actual mel frames by running transform on dummy input."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=HOP_LENGTH, n_mels=N_MELS, normalized=True
    )
    num_samples = int(SAMPLE_RATE * AUDIO_DURATION)
    dummy = torch.zeros(num_samples)
    return mel_transform(dummy).shape[-1]

AUDIO_FRAMES = _get_mel_frames()

print(f"Audio config: {N_MELS} mels x {AUDIO_FRAMES} frames = {N_MELS * AUDIO_FRAMES} tokens")

class MelSpectrogramEncoder(nn.Module):
    """
    Encodes raw waveform to mel spectrogram latent representation.
    
    Input:  (batch, samples) or (samples,) - raw audio waveform
    Output: (batch, latent_dim, time_frames) - latent representation
    """
    
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
        n_fft: int = 1024,
        hop_length: int = HOP_LENGTH,
        latent_dim: int = LATENT_DIM
    ):
        super().__init__()
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            normalized=True
        )
        
        # Simple projection from mel bins to latent dimension
        # Could be replaced with more complex encoder (conv layers, etc.)
        self.proj = nn.Sequential(
            nn.Conv1d(n_mels, latent_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
        )
        
    def forward(self, waveform):
        # Handle single sample (no batch dim)
        squeeze_batch = waveform.dim() == 1
        if squeeze_batch:
            waveform = waveform.unsqueeze(0)
        
        # Compute mel spectrogram: (batch, n_mels, time_frames)
        mel = self.mel_transform(waveform)
        
        # Log-scale and normalize
        mel = torch.log(mel.clamp(min=1e-5))
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        
        # Project to latent: (batch, latent_dim, time_frames)
        latent = self.proj(mel)
        
        if squeeze_batch:
            latent = latent.squeeze(0)
            
        return latent


class MelSpectrogramDecoder(nn.Module):
    """
    Decodes latent representation back to mel spectrogram.
    
    Input:  (batch, latent_dim, time_frames) - latent representation  
    Output: (batch, n_mels, time_frames) - mel spectrogram
    
    Note: For actual audio playback, you'd need a vocoder (HiFi-GAN, etc.)
    to convert mel spectrogram back to waveform.
    """
    
    def __init__(
        self,
        n_mels: int = N_MELS,
        latent_dim: int = LATENT_DIM
    ):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(latent_dim * 2, n_mels, kernel_size=3, padding=1),
        )
        
    def forward(self, latent):
        # latent: (batch, latent_dim, time_frames)
        mel = self.proj(latent)
        return mel

# Model Definition

model = Transfusion(
    num_text_tokens=0,                    # No text tokens for audio-only
    dim_latent=LATENT_DIM,                # Latent dimension
    channel_first_latent=True,            # (batch, channels, time) format
    modality_default_shape=(AUDIO_FRAMES,),  # Default audio length
    modality_encoder=MelSpectrogramEncoder(),
    modality_decoder=MelSpectrogramDecoder(),
    add_pos_emb=True,                     # Add positional embeddings
    modality_num_dim=1,                   # 1D modality (time axis)
    velocity_consistency_loss_weight=0.1, # For straighter flow trajectories
    model_output_clean=True,              # Predict clean data instead of flow
    transformer=dict(
        dim=128,                          # Model dimension
        depth=6,                          # Number of transformer layers
        dim_head=32,                      # Attention head dimension
        heads=4,                          # Number of attention heads
        attn_laser=False,                 # LASER attention (optional)
    )
)

# Create EMA model for better sampling
ema_model = model.create_ema(beta=0.995)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Demo dataset - synthetic audio (pure tones with harmonics)

class SyntheticToneDataset(Dataset):
    """
    Generates synthetic audio (pure tones with harmonics) for testing.
    
    In practice, replace this with your actual audio dataset:
    - LibriSpeech for speech
    - MusicCaps for music
    - Custom dataset for your use case
    """
    
    def __init__(
        self,
        size: int = 10000,
        sample_rate: int = SAMPLE_RATE,
        duration: float = AUDIO_DURATION
    ):
        self.size = size
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate time axis
        t = torch.linspace(0, self.duration, self.num_samples)
        
        # Random base frequency (musical notes A2 to A5)
        base_freq = 110 * (2 ** (torch.rand(1).item() * 3))  # 110Hz to 880Hz
        
        # Generate tone with harmonics
        waveform = torch.zeros(self.num_samples)
        for harmonic in range(1, 5):
            amplitude = 1.0 / harmonic
            waveform += amplitude * torch.sin(2 * torch.pi * base_freq * harmonic * t)
        
        # Normalize
        waveform = waveform / waveform.abs().max()
        
        # Add envelope (attack-decay)
        attack = torch.linspace(0, 1, self.num_samples // 10)
        decay = torch.linspace(1, 0, self.num_samples - len(attack))
        envelope = torch.cat([attack, decay])
        waveform = waveform * envelope
        
        # Add slight noise for realism
        waveform = waveform + 0.01 * torch.randn_like(waveform)
        
        return waveform.float()


# Training loop

def divisible_by(num, den):
    return (num % den) == 0


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def save_mel_spectrogram(mel, path, title="Generated Mel Spectrogram"):
    """Save mel spectrogram as an image."""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        plt.imshow(
            mel.detach().cpu().numpy(), 
            aspect='auto', 
            origin='lower',
            cmap='magma'
        )
        plt.colorbar(label='Amplitude')
        plt.xlabel('Time Frame')
        plt.ylabel('Mel Bin')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=100)
        plt.close()
    except ImportError:
        print("matplotlib not installed, skipping visualization")


# Create dataset and dataloader
dataset = SyntheticToneDataset()
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
iter_dl = cycle(dataloader)

# Optimizer
optimizer = Adam(model.parameters(), lr=3e-4)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ema_model.to(device)

print(f"Training on: {device}")
print(f"Audio frames per sample: {AUDIO_FRAMES}")
print(f"Starting training for {NUM_TRAIN_STEPS} steps...")

# Training loop
for step in range(1, NUM_TRAIN_STEPS + 1):
    model.train()
    
    # Get batch of audio waveforms
    batch = next(iter_dl).to(device)
    
    # Forward pass - model handles encoding internally
    loss = model(batch, velocity_consistency_ema_model=ema_model)
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Update EMA model
    ema_model.update()
    
    # Logging
    if step % 50 == 0:
        print(f'Step {step}/{NUM_TRAIN_STEPS}: loss={loss.item():.4f}')
    
    # Sampling
    if divisible_by(step, SAMPLE_EVERY):
        print(f"\n--- Generating sample at step {step} ---")
        
        # Generate mel spectrogram using EMA model
        with torch.no_grad():
            # generate_modality_only returns decoded output (mel spectrogram)
            generated_mel = ema_model.generate_modality_only(
                batch_size=4,
                modality_steps=32  # More steps = higher quality
            )
        
        # Save first sample as image
        mel = generated_mel[0]  # (n_mels, time_frames)
        save_mel_spectrogram(
            mel,
            results_folder / f'mel_step_{step}.png',
            title=f'Generated Mel Spectrogram - Step {step}'
        )
        
        # Also save a grid of all 4 samples
        if generated_mel.shape[0] >= 4:
            grid = torch.cat([generated_mel[i] for i in range(4)], dim=1)
            save_mel_spectrogram(
                grid,
                results_folder / f'mel_grid_step_{step}.png',
                title=f'Generated Samples Grid - Step {step}'
            )
        
        print(f"Saved samples to {results_folder}")
        print()

print("\n" + "="*50)
print("Training complete!")
print(f"Results saved to: {results_folder}")
print("="*50)

# Final generation with more steps for better quality
print("\nGenerating final high-quality samples...")
with torch.no_grad():
    final_samples = ema_model.generate_modality_only(
        batch_size=8,
        modality_steps=64  # More steps for final samples
    )
    
for i, mel in enumerate(final_samples):
    save_mel_spectrogram(
        mel,
        results_folder / f'final_sample_{i}.png',
        title=f'Final Sample {i}'
    )

print(f"Saved {len(final_samples)} final samples to {results_folder}")

