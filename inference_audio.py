"""
Inference script for trained audio-only Transfusion model.
Generates mel spectrograms and converts them to listenable audio.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio

from transfusion_pytorch import Transfusion

# Must match training config
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
LATENT_DIM = 128
MAX_AUDIO_DURATION = 4.0
N_FFT = 1024


def _get_mel_frames(duration):
    """Compute mel frames for a given duration."""
    num_samples = int(SAMPLE_RATE * duration)
    # Approximate: (num_samples / hop_length) + 1
    return num_samples // HOP_LENGTH + 1


MAX_AUDIO_FRAMES = _get_mel_frames(MAX_AUDIO_DURATION)


class MelSpectrogramEncoder(nn.Module):
    """Encodes raw waveform to mel spectrogram latent representation."""

    def __init__(self, sample_rate=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, latent_dim=LATENT_DIM):
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


def mel_to_audio_griffin_lim(mel_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH, 
                              n_mels=N_MELS, sample_rate=SAMPLE_RATE, n_iter=64):
    """
    Convert mel spectrogram to audio using Griffin-Lim algorithm.
    
    Args:
        mel_spectrogram: Tensor of shape (n_mels, time) or (batch, n_mels, time)
        n_iter: Number of Griffin-Lim iterations (more = better quality)
    
    Returns:
        Audio waveform tensor
    """
    device = mel_spectrogram.device
    
    # Handle batch dimension
    if mel_spectrogram.dim() == 2:
        mel_spectrogram = mel_spectrogram.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    # Create inverse mel scale
    mel_basis = torchaudio.functional.melscale_fbanks(
        n_freqs=n_fft // 2 + 1,
        f_min=0.0,
        f_max=sample_rate / 2,
        n_mels=n_mels,
        sample_rate=sample_rate,
    ).to(device)  # (n_freqs, n_mels)
    
    # Pseudo-inverse for mel to linear spectrogram
    mel_basis_pinv = torch.linalg.pinv(mel_basis.T)  # (n_freqs, n_mels)
    
    audios = []
    for mel in mel_spectrogram:
        # Denormalize (approximate - we lost the exact mean/std during encoding)
        mel_denorm = mel * 4.0  # Rough scaling
        
        # Convert from log mel to linear mel
        mel_linear = torch.exp(mel_denorm)
        
        # Convert mel to linear spectrogram
        linear_spec = torch.matmul(mel_basis_pinv, mel_linear)  # (n_freqs, time)
        linear_spec = torch.clamp(linear_spec, min=1e-5)
        
        # Griffin-Lim
        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
            power=1.0,
            n_iter=n_iter,
        ).to(device)
        
        audio = griffin_lim(linear_spec)
        audios.append(audio)
    
    result = torch.stack(audios)
    if squeeze:
        result = result.squeeze(0)
    
    return result


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    
    # Create model with same architecture as training
    model = Transfusion(
        num_text_tokens=0,
        dim_latent=LATENT_DIM,
        channel_first_latent=True,
        modality_default_shape=(MAX_AUDIO_FRAMES,),
        modality_encoder=MelSpectrogramEncoder(n_mels=N_MELS, latent_dim=LATENT_DIM),
        modality_decoder=MelSpectrogramDecoder(n_mels=N_MELS, latent_dim=LATENT_DIM),
        add_pos_emb=True,
        modality_num_dim=1,
        velocity_consistency_loss_weight=0.1,
        model_output_clean=True,
        transformer=dict(
            dim=384,
            depth=12,
            dim_head=64,
            heads=6,
            attn_laser=True,
        ),
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try loading EMA model first (better quality), fall back to regular model
    if 'ema_model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_model_state_dict'])
        print("Loaded EMA model weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights")
    else:
        raise ValueError("Checkpoint doesn't contain model weights")
    
    model = model.to(device)
    model.eval()
    
    step = checkpoint.get('step', 'unknown')
    print(f"Loaded checkpoint from step {step}")
    
    return model


def generate_audio(model, num_samples=4, duration=None, steps=128, device='cuda'):
    """
    Generate audio samples from the model.
    
    Args:
        model: Trained Transfusion model
        num_samples: Number of samples to generate
        duration: Duration in seconds (None = max duration)
        steps: Number of diffusion steps (more = better quality)
        device: Device to run on
    
    Returns:
        Tuple of (mel_spectrograms, audio_waveforms)
    """
    if duration is not None:
        target_frames = _get_mel_frames(duration)
        target_frames = max(32, min(MAX_AUDIO_FRAMES, target_frames))
        fixed_shape = (target_frames,)
    else:
        fixed_shape = None
    
    with torch.no_grad():
        if fixed_shape:
            mel = model.generate_modality_only(
                batch_size=num_samples,
                fixed_modality_shape=fixed_shape,
                modality_steps=steps
            )
        else:
            mel = model.generate_modality_only(
                batch_size=num_samples,
                modality_steps=steps
            )
    
    # Convert mel to audio
    audio = mel_to_audio_griffin_lim(mel)
    
    return mel, audio


def save_mel_spectrogram(mel, path, title="Generated Mel Spectrogram"):
    """Save mel spectrogram as image."""
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


def main():
    parser = argparse.ArgumentParser(description='Generate audio from trained Transfusion model')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to checkpoint file')
    parser.add_argument('--output_dir', type=str, default='./generated_audio',
                        help='Output directory for generated audio')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='Number of samples to generate')
    parser.add_argument('--duration', type=float, default=None,
                        help='Duration in seconds (default: max trained duration)')
    parser.add_argument('--steps', type=int, default=128,
                        help='Number of diffusion steps (more = better quality)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Set random seed: {args.seed}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device)
    
    # Generate
    duration_str = f"{args.duration}s" if args.duration else "max"
    print(f"\nGenerating {args.num_samples} samples (duration={duration_str}, steps={args.steps})...")
    
    mel_specs, audios = generate_audio(
        model, 
        num_samples=args.num_samples,
        duration=args.duration,
        steps=args.steps,
        device=device
    )
    
    # Save outputs
    print(f"\nSaving outputs to: {output_dir}")
    
    for i in range(len(audios)):
        # Save audio
        audio_path = output_dir / f'sample_{i}.wav'
        audio = audios[i].cpu()
        
        # Normalize audio
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max() * 0.95
        
        torchaudio.save(str(audio_path), audio.unsqueeze(0), SAMPLE_RATE)
        print(f"  Saved: {audio_path}")
        
        # Save mel spectrogram image
        mel_path = output_dir / f'sample_{i}_mel.png'
        save_mel_spectrogram(mel_specs[i], mel_path, title=f'Generated Sample {i}')
    
    print(f"\n✓ Generated {len(audios)} audio samples!")
    print(f"  Audio files: {output_dir}/*.wav")
    print(f"  Mel spectrograms: {output_dir}/*_mel.png")
    
    # Print playback instructions
    print("\n" + "=" * 50)
    print("To listen to the audio:")
    print("=" * 50)
    print(f"  # macOS:")
    print(f"  afplay {output_dir}/sample_0.wav")
    print(f"  ")
    print(f"  # Linux:")
    print(f"  aplay {output_dir}/sample_0.wav")
    print(f"  ")
    print(f"  # Or open in any audio player / Jupyter notebook")


if __name__ == '__main__':
    main()

