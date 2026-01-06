"""
Train Transfusion for text-to-audio generation

This demonstrates multimodal training with text conditioning audio generation.
The model learns to generate audio conditioned on text descriptions.

"""

from shutil import rmtree
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.optim import Adam

from transfusion_pytorch import Transfusion, print_modality_sample

# Clean and create results folder
rmtree('./results_text_to_audio', ignore_errors=True)
results_folder = Path('./results_text_to_audio')
results_folder.mkdir(exist_ok=True, parents=True)

# Sample configuration

SAMPLE_RATE = 16000
N_MELS = 64
HOP_LENGTH = 256
LATENT_DIM = 64
AUDIO_DURATION = 2.0
BATCH_SIZE = 8
NUM_TRAIN_STEPS = 30_000
SAMPLE_EVERY = 1000

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

print(f"Audio config: {N_MELS} mels x {AUDIO_FRAMES} frames")

# Audio encoder/decoder

class MelEncoder(nn.Module):
    """Encodes waveform to mel spectrogram latent."""
    
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        )
        self.proj = nn.Sequential(
            nn.Conv1d(N_MELS, LATENT_DIM * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, LATENT_DIM, 3, padding=1),
        )
        
    def forward(self, x):
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
            
        mel = self.mel(x)
        mel = torch.log(mel.clamp(min=1e-5))
        mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-5)
        latent = self.proj(mel)
        
        if squeeze:
            latent = latent.squeeze(0)
        return latent


class MelDecoder(nn.Module):
    """Decodes latent to mel spectrogram."""
    
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(LATENT_DIM, LATENT_DIM * 2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(LATENT_DIM * 2, N_MELS, 3, padding=1),
        )
        
    def forward(self, x):
        return self.proj(x)


# Model Definition

model = Transfusion(
    # Text configuration
    num_text_tokens=256,  # ASCII characters
    
    # Audio modality configuration
    dim_latent=LATENT_DIM,
    channel_first_latent=True,
    modality_default_shape=(AUDIO_FRAMES,),
    modality_encoder=MelEncoder(),
    modality_decoder=MelDecoder(),
    
    # Position embeddings for audio
    add_pos_emb=True,
    modality_num_dim=1,
    
    # Loss weights
    text_loss_weight=1.0,
    flow_loss_weight=1.0,
    velocity_consistency_loss_weight=0.1,
    
    # Model predicts clean data
    model_output_clean=True,
    
    # Transformer configuration
    transformer=dict(
        dim=192,
        depth=8,
        dim_head=48,
        heads=4,
    )
)

ema_model = model.create_ema(beta=0.995)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Dataset - Synthetic text-audio pairs

class TextAudioDataset(Dataset):
    """
    Synthetic dataset generating audio with text descriptions.
    
    Generates simple tones with descriptions like:
    - "low pitch tone" → 200Hz
    - "high pitch tone" → 800Hz
    - "medium pitch tone" → 400Hz
    """
    
    DESCRIPTIONS = [
        ("very low tone", 110),
        ("low tone", 220),
        ("medium low tone", 330),
        ("medium tone", 440),
        ("medium high tone", 550),
        ("high tone", 660),
        ("very high tone", 880),
        ("deep bass", 100),
        ("bright tone", 1000),
        ("warm tone", 300),
    ]
    
    def __init__(self, size: int = 10000):
        self.size = size
        self.num_samples = int(SAMPLE_RATE * AUDIO_DURATION)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Select description and frequency
        desc, freq = self.DESCRIPTIONS[idx % len(self.DESCRIPTIONS)]
        
        # Add some randomness to frequency
        freq = freq * (0.9 + 0.2 * torch.rand(1).item())
        
        # Generate tone
        t = torch.linspace(0, AUDIO_DURATION, self.num_samples)
        waveform = torch.sin(2 * torch.pi * freq * t)
        
        # Add harmonics for richer sound
        for h in [2, 3, 4]:
            waveform += (0.5 / h) * torch.sin(2 * torch.pi * freq * h * t)
        
        # Normalize
        waveform = waveform / waveform.abs().max()
        
        # Add envelope
        attack_len = self.num_samples // 20
        decay_len = self.num_samples - attack_len
        envelope = torch.cat([
            torch.linspace(0, 1, attack_len),
            torch.linspace(1, 0, decay_len)
        ])
        waveform = waveform * envelope
        
        # Add slight noise
        waveform = waveform + 0.02 * torch.randn_like(waveform)
        
        # Convert text to tensor
        text_tensor = torch.tensor([ord(c) for c in desc], dtype=torch.long)
        
        # Return [text, audio] - audio will be encoded by modality_encoder
        return text_tensor, waveform.float()


# Training loop

def divisible_by(num, den):
    return (num % den) == 0


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def decode_tokens(tokens):
    """Convert token tensor to string."""
    chars = []
    for t in tokens.tolist():
        if 0 <= t < 256:
            chars.append(chr(t))
    return ''.join(chars)


def save_mel(mel, path, title=""):
    """Save mel spectrogram visualization."""
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
        pass


# Create dataset and dataloader
dataset = TextAudioDataset(size=10000)

# Use Transfusion's custom dataloader for variable-length sequences
dataloader = model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)
iter_dl = cycle(dataloader)

# Optimizer
optimizer = Adam(model.parameters(), lr=2e-4)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
ema_model.to(device)

print(f"Training on: {device}")
print(f"Starting training for {NUM_TRAIN_STEPS} steps...")
print()

# Training loop
for step in range(1, NUM_TRAIN_STEPS + 1):
    model.train()
    
    # Get batch: list of [text_tensor, audio_tensor] pairs
    batch = next(iter_dl)
    
    # Move to device
    batch = [[item.to(device) if torch.is_tensor(item) else item for item in sample] for sample in batch]
    
    # Forward pass
    loss = model(batch, velocity_consistency_ema_model=ema_model)
    
    # Backward pass
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    optimizer.step()
    optimizer.zero_grad()
    
    ema_model.update()
    
    # Logging
    if step % 100 == 0:
        print(f'Step {step}/{NUM_TRAIN_STEPS}: loss={loss.item():.4f}')
    
    # Sampling
    if divisible_by(step, SAMPLE_EVERY):
        print(f"\n{'='*50}")
        print(f"Generating samples at step {step}")
        print('='*50)
        
        # Test different prompts
        test_prompts = ["low tone", "high tone", "medium tone"]
        
        for prompt_text in test_prompts:
            print(f"\nPrompt: '{prompt_text}'")
            
            # Encode prompt
            prompt = torch.tensor([ord(c) for c in prompt_text], dtype=torch.long).to(device)
            
            # Generate with prompt
            try:
                sample = ema_model.sample(
                    prompt=prompt,
                    max_length=512,
                    modality_steps=32,
                    text_temperature=0.8
                )
                
                # Print sample structure
                print_modality_sample(sample)
                
                # Extract and save audio if present
                audio_found = False
                for i, item in enumerate(sample):
                    if isinstance(item, tuple):
                        modality_type, mel = item
                        if mel.dim() >= 2:
                            safe_prompt = prompt_text.replace(' ', '_')
                            save_mel(
                                mel,
                                results_folder / f'step_{step}_{safe_prompt}.png',
                                title=f"'{prompt_text}' - Step {step}"
                            )
                            print(f"  Saved mel spectrogram")
                            audio_found = True
                            break
                
                if not audio_found:
                    print(f"  Warning: No audio generated (model may need more training)")
                            
            except Exception as e:
                print(f"  Sampling failed: {e}")
        
        # Also try unconditional generation
        print("\nUnconditional generation:")
        try:
            sample = ema_model.sample(max_length=512, modality_steps=32)
            print_modality_sample(sample)
            
            # Extract text and audio
            text_parts = []
            for item in sample:
                if torch.is_tensor(item) and item.dtype in (torch.int, torch.long):
                    text = decode_tokens(item)
                    text_parts.append(text)
                elif isinstance(item, tuple):
                    modality_type, mel = item
                    if mel.dim() >= 2:
                        save_mel(
                            mel,
                            results_folder / f'step_{step}_unconditional.png',
                            title=f"Unconditional - Step {step}"
                        )
            
            if text_parts:
                print(f"  Generated text: {''.join(text_parts)[:50]}...")
                
        except Exception as e:
            print(f"  Sampling failed: {e}")
        
        print()

print("\n" + "="*50)
print("Training complete!")
print(f"Results saved to: {results_folder}")
print("="*50)

# Final evaluation

print("\nFinal evaluation with all prompts:")

all_prompts = [desc for desc, _ in TextAudioDataset.DESCRIPTIONS]

for prompt_text in all_prompts:
    prompt = torch.tensor([ord(c) for c in prompt_text], dtype=torch.long).to(device)
    
    try:
        sample = ema_model.sample(
            prompt=prompt,
            max_length=512,
            modality_steps=64  # More steps for final samples
        )
        
        for item in sample:
            if isinstance(item, tuple):
                modality_type, mel = item
                if mel.dim() >= 2:
                    safe_prompt = prompt_text.replace(' ', '_')
                    save_mel(
                        mel,
                        results_folder / f'final_{safe_prompt}.png',
                        title=f"Final: '{prompt_text}'"
                    )
                    print(f"  ✓ Generated: {prompt_text}")
                    break
                    
    except Exception as e:
        print(f"  ✗ Failed: {prompt_text} - {e}")

print(f"\nAll results saved to: {results_folder}")

