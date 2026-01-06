# Transfusion-PyTorch: Complete Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture Deep Dive](#architecture-deep-dive)
3. [Training Mechanism](#training-mechanism)
4. [Inference and Sampling](#inference-and-sampling)
5. [Adding Audio Modality](#adding-audio-modality)

---

## 1. Overview

### What is Transfusion?

Transfusion is a unified architecture from Meta AI that combines:
- **Autoregressive Language Modeling** (for text generation) 
- **Flow Matching** (for continuous modality generation like images/audio)

The key innovation is that **one single Transformer** can handle both discrete tokens (text) and continuous data (images, audio, etc.) by using different training objectives for each modality type:
- **Cross-entropy loss** for text (next-token prediction)
- **Flow matching loss** for continuous modalities (denoising/rectified flow)

### Repository Modifications

This implementation replaces standard diffusion with **Flow Matching** (rectified flow) inspired by Flux from Black Forest Labs. This provides:
- Simpler training (no noise schedule needed)
- Straight line trajectories in latent space
- Faster sampling via ODE solvers

---

## 2. Architecture Deep Dive

### 2.1 Core Components

#### Main Transfusion Class

```python
Transfusion(
    num_text_tokens,        # Size of text vocabulary
    dim_latent,             # Latent dimension for modalities (or tuple for multiple)
    transformer,            # Transformer configuration dict or Transformer instance
    modality_default_shape, # Default shape for each modality type
    modality_encoder,       # Optional encoder (e.g., VAE encoder)
    modality_decoder,       # Optional decoder (e.g., VAE decoder)
    ...
)
```

#### Transformer Architecture

The internal `Transformer` class (`transfusion.py:1035-1253`) features:

1. **Adaptive LayerNorm (AdaLN)**: Different normalization for text vs modalities
2. **Rotary Position Embeddings (RoPE)**: For handling variable-length sequences
3. **Hyper-Connections**: Multi-stream residual connections
4. **U-Net Style Skip Connections**: First-half features connected to second-half
5. **LASER Attention**: Optional exponential value transformation
6. **Flex Attention**: PyTorch 2.5+ optimized attention with custom masking

#### Key Architectural Decisions

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRANSFUSION ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   Input: [text_tokens] [SOM] [modality_tokens] [EOM] [text...]   │
│                          ↓                                       │
│   ┌───────────────────────────────────────────────────────┐     │
│   │           Text Embedding Layer                         │     │
│   │    (Embeds discrete text tokens to dim dimensions)     │     │
│   └───────────────────────────────────────────────────────┘     │
│                          ↓                                       │
│   ┌───────────────────────────────────────────────────────┐     │
│   │           Modality Encoder/Projection                  │     │
│   │    (Projects continuous modality to dim dimensions)    │     │
│   │    Optional: VAE encoder, Conv layers, etc.            │     │
│   └───────────────────────────────────────────────────────┘     │
│                          ↓                                       │
│   ┌───────────────────────────────────────────────────────┐     │
│   │              Combined Sequence                         │     │
│   │    Text tokens + Modality tokens interleaved           │     │
│   └───────────────────────────────────────────────────────┘     │
│                          ↓                                       │
│   ┌───────────────────────────────────────────────────────┐     │
│   │           TRANSFORMER LAYERS (depth)                   │     │
│   │                                                        │     │
│   │   For each layer:                                      │     │
│   │   ┌────────────────────────────────────────────────┐  │     │
│   │   │  Skip Connection (if second half of layers)    │  │     │
│   │   └────────────────────────────────────────────────┘  │     │
│   │                      ↓                                │     │
│   │   ┌────────────────────────────────────────────────┐  │     │
│   │   │  Adaptive LayerNorm + Multi-Head Attention     │  │     │
│   │   │  - Causal for text                             │  │     │
│   │   │  - Bidirectional for modality positions        │  │     │
│   │   │  - RoPE for positional encoding                │  │     │
│   │   │  - Value gating + Learned value residual       │  │     │
│   │   └────────────────────────────────────────────────┘  │     │
│   │                      ↓                                │     │
│   │   ┌────────────────────────────────────────────────┐  │     │
│   │   │  Adaptive LayerNorm + FeedForward (GEGLU)      │  │     │
│   │   │  - Different scaling for text vs modality      │  │     │
│   │   └────────────────────────────────────────────────┘  │     │
│   │                                                        │     │
│   └───────────────────────────────────────────────────────┘     │
│                          ↓                                       │
│   ┌───────────────────────────────────────────────────────┐     │
│   │           Output Projection                            │     │
│   │   - Text: to_text_logits → vocabulary logits           │     │
│   │   - Modality: model_to_latent → predicted flow/clean   │     │
│   └───────────────────────────────────────────────────────┘     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Attention Masking Strategy

The transfusion attention mask is crucial:

```python
def transfusion_attn_mask(modalities):
    # Text tokens: CAUSAL attention (can only attend to past)
    # Modality tokens: BIDIRECTIONAL attention (all modality tokens attend to each other)
    # Cross-modality: Modalities can attend to preceding text
```

This is implemented in `transfusion.py:343-356`:
- Text uses standard causal masking
- Each modality region gets bidirectional (full) attention within itself
- The combination allows autoregressive text with diffusion-style modality denoising

### 2.3 Time Conditioning

For flow matching, time (noise level) conditioning is injected via:

1. **Random Fourier Embedding** (`RandomFourierEmbed`): Converts scalar time to rich embedding
2. **AdaptiveWrapper**: Uses FiLM conditioning (Feature-wise Linear Modulation)
   - `gamma, beta = self.to_film(time_cond)` 
   - `modality_tokens = x * (gamma + 1.) + beta`
3. **AdaLN-Zero**: Output scaling based on time for modalities

---

## 3. Training Mechanism

### 3.1 Training Modes

The model supports three training modes:

#### Mode 1: Text-Only Training
```python
text = torch.randint(0, 256, (batch, seq_len))
loss = model(text)  # Returns cross-entropy loss
```

#### Mode 2: Modality-Only Training  
```python
images = torch.randn(batch, channels, height, width)
loss = model(images, modality_type=0)  # Returns flow matching loss
```

#### Mode 3: Multimodal Training
```python
# List of samples, each containing interleaved text and modalities
data = [
    [text_tensor, image_tensor, text_tensor, image_tensor],
    [text_tensor, image_tensor, text_tensor]
]
loss = model(data)  # Returns combined loss
```

### 3.2 Flow Matching Training

Flow matching trains the model to predict the "flow" (velocity field) that transforms noise to data:

```
┌─────────────────────────────────────────────────────────────────┐
│                    FLOW MATCHING TRAINING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Sample time t ~ Uniform(0, 1)                               │
│                                                                  │
│   2. Create interpolated sample:                                 │
│      noised = t * data + (1-t) * noise                           │
│                                                                  │
│   3. Ground truth flow:                                          │
│      flow = data - noise                                         │
│                                                                  │
│   4. Model predicts either:                                      │
│      - Flow directly (model_output_clean=False)                  │
│      - Clean data (model_output_clean=True) → convert to flow    │
│                                                                  │
│   5. Loss = MSE(predicted_flow, ground_truth_flow)               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

From `transfusion.py:2024-2090`:
```python
# Sample random time
times = torch.rand((batch,), device=device)

# Linear interpolation (flow matching)
padded_times = append_dims(times, tokens.ndim - 1)
noise = torch.randn_like(tokens)
noised_tokens = padded_times * tokens + (1. - padded_times) * noise

# Ground truth flow
flow = tokens - noise

# ... model forward pass ...

# Flow loss
flow_loss = F.mse_loss(pred_flow, flow)
```

### 3.3 Combined Loss Computation

```python
total_loss = (
    text_loss * text_loss_weight * self.text_loss_weight +
    flow_loss * modality_loss_weight * self.flow_loss_weight +
    velocity_loss * self.velocity_consistency_loss_weight +
    recon_loss * self.reconstruction_loss_weight
)
```

Components:
- **Text Loss**: Cross-entropy for next token prediction
- **Flow Loss**: MSE between predicted and true flow
- **Velocity Consistency Loss**: Optional, for straightening flow trajectories
- **Reconstruction Loss**: Optional, for decoder quality

### 3.4 Training Example: MNIST with Text Labels

```python
from transfusion_pytorch import Transfusion

model = Transfusion(
    num_text_tokens=10,           # Digits 0-9
    dim_latent=4,                 # Latent dimension after encoding
    modality_default_shape=(14, 14),  # 28x28 → 14x14 patches
    modality_encoder=Encoder(),   # Patches image to latent
    modality_decoder=Decoder(),   # Reconstructs from latent
    add_pos_emb=True,
    modality_num_dim=2,           # 2D modality (image)
    transformer=dict(
        dim=64,
        depth=4,
        dim_head=32,
        heads=8
    )
)

# Training loop
for step in range(num_steps):
    batch = next(dataloader)  # Returns [[label, image], [label, image], ...]
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

---

## 4. Inference and Sampling

### 4.1 Text-Only Generation

```python
prompt = torch.randint(0, 256, (1, 10))
generated = model.generate_text_only(
    prompt, 
    seq_len=256,
    temperature=1.5,
    min_p=0.1
)
```

Uses standard autoregressive sampling with min-p filtering.

### 4.2 Modality-Only Generation

```python
samples = model.generate_modality_only(
    batch_size=8,
    modality_type=0,  # Which modality
    modality_steps=16  # ODE solver steps
)
```

Uses ODE integration (via `torchdiffeq.odeint`) to sample from noise to data.

### 4.3 Multimodal Sampling

```
┌─────────────────────────────────────────────────────────────────┐
│                    MULTIMODAL SAMPLING                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. Start with [SOS] token                                      │
│                                                                  │
│   2. WHILE not done:                                             │
│      │                                                           │
│      ├─► IF decoding_text:                                       │
│      │      Sample next token autoregressively                   │
│      │      IF token == [SOM]:                                   │
│      │         Parse modality shape from meta tokens             │
│      │         Switch to modality decoding                       │
│      │      IF token == [EOS]:                                   │
│      │         STOP                                              │
│      │                                                           │
│      └─► IF decoding_modality:                                   │
│             Initialize noise of modality_shape                   │
│             FOR t in [0, 1] over modality_steps:                 │
│                 pred_flow = model(context + noised_modality, t)  │
│                 noised_modality = ODE_step(noised_modality)      │
│             Append decoded modality                              │
│             Append [EOM] token                                   │
│             Switch to text decoding                              │
│                                                                  │
│   3. Return full multimodal sample                               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

From `transfusion.py:1618-1890`:
```python
def sample(
    self,
    prompt=None,
    max_length=2048,
    text_temperature=1.5,
    modality_steps=16,
    cache_kv=False,
    ...
):
    # Alternates between:
    # 1. Autoregressive text decoding
    # 2. ODE-based modality generation when [SOM] detected
```

### 4.4 ODE Sampling for Modalities

```python
def ode_step_fn(step_times, denoised):
    # Get embeddings from transformer
    (embeds, get_pred_flows), _ = self.forward(
        [[*context, (modality_type, denoised)]],
        times=step_times,
        return_embed=True
    )
    # Extract flow prediction
    flow = model_to_latent(parse_embed(embeds))
    return flow

# Integrate from t=0 (noise) to t=1 (data)
times = torch.linspace(0, 1, modality_steps)
trajectory = odeint(ode_step_fn, noise, times)
sampled_modality = trajectory[-1]
```

---

## 5. Adding Audio Modality

### 5.1 Architecture Compatibility Analysis

**Good News**: The Transfusion architecture is **designed for multiple modalities**! The codebase already supports:

- Multiple latent dimensions: `dim_latent = (384, 192, 256)` 
- Per-modality encoders/decoders
- Per-modality shapes: `modality_default_shape = ((16, 16), (1000,))`
- Per-modality number of dimensions: `modality_num_dim = (2, 1)`

### 5.2 Audio Representation Options

#### Option A: Raw Waveform with Patches
```python
# Audio: 16kHz, 1 second = 16000 samples
# Patch: groups of 256 samples → 62 tokens
# Latent dim: 256 (patch encoding dimension)
```

#### Option B: Mel Spectrogram (Recommended)
```python
# Audio → Mel Spectrogram → 2D image-like representation
# Shape: (n_mels, time_frames) e.g., (80, 200)
# Can use 2D convolutions similar to images
```

#### Option C: Neural Audio Codec (Best for High Quality)
```python
# Use pre-trained audio codec (EnCodec, DAC, etc.)
# Maps audio → discrete/continuous latent codes
# Similar to VAE for images
```

### 5.3 Step-by-Step Implementation Guide

#### Step 1: Create Audio Encoder/Decoder

```python
import torch
import torch.nn as nn
import torchaudio

class AudioMelEncoder(nn.Module):
    """Encodes raw waveform to mel spectrogram latent."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        latent_dim: int = 256
    ):
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Project mel bins to latent dimension
        self.proj = nn.Sequential(
            nn.Conv1d(n_mels, latent_dim, 1),
            nn.GELU(),
            nn.Conv1d(latent_dim, latent_dim, 1)
        )
        
    def forward(self, waveform):
        # waveform: (batch, samples) or (samples,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # (batch, n_mels, time_frames)
        mel = self.mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-5))
        
        # Normalize
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        
        # Project: (batch, latent_dim, time_frames)
        latent = self.proj(mel)
        
        return latent  # channel_first format


class AudioMelDecoder(nn.Module):
    """Decodes latent back to mel spectrogram (or uses vocoder for waveform)."""
    
    def __init__(
        self,
        n_mels: int = 80,
        latent_dim: int = 256,
        use_vocoder: bool = False
    ):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(latent_dim, latent_dim, 1),
            nn.GELU(),
            nn.Conv1d(latent_dim, n_mels, 1)
        )
        
        self.use_vocoder = use_vocoder
        if use_vocoder:
            # Could integrate HiFi-GAN, Vocos, etc.
            self.vocoder = None  # Load pretrained vocoder
        
    def forward(self, latent):
        # latent: (batch, latent_dim, time_frames)
        mel = self.proj(latent)
        
        if self.use_vocoder and self.vocoder is not None:
            waveform = self.vocoder(mel)
            return waveform
            
        return mel
```

#### Step 2: Using Neural Audio Codec (Alternative - Higher Quality)

```python
# pip install encodec  # Meta's neural audio codec

from encodec import EncodecModel
from encodec.utils import convert_audio

class EncodecEncoder(nn.Module):
    """Wraps EnCodec for audio encoding."""
    
    def __init__(self, bandwidth: float = 6.0):
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, waveform):
        # waveform: (batch, channels, samples) at 24kHz
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
            
        # Get continuous embeddings before quantization
        encoded = self.model.encoder(waveform)
        return encoded  # (batch, latent_dim, time_frames)


class EncodecDecoder(nn.Module):
    """Wraps EnCodec for audio decoding."""
    
    def __init__(self, bandwidth: float = 6.0):
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(bandwidth)
        self.model.eval()
        
    @torch.no_grad()
    def forward(self, latent):
        # latent: (batch, latent_dim, time_frames)
        waveform = self.model.decoder(latent)
        return waveform.squeeze(1)  # (batch, samples)
```

#### Step 3: Configure Transfusion with Audio

```python
from transfusion_pytorch import Transfusion

# Option 1: Image + Audio (two modalities)
model = Transfusion(
    num_text_tokens=256,  # Character-level or BPE
    
    # Multiple modalities: (image, audio)
    dim_latent=(384, 256),  # Image latent dim, Audio latent dim
    
    # Channel-first for both (typical for conv-based encoders)
    channel_first_latent=(True, True),
    
    # Default shapes
    # Image: 16x16 latent patches
    # Audio: 100 time frames (adjust based on your audio length)
    modality_default_shape=((16, 16), (100,)),
    
    # Number of dimensions per modality
    modality_num_dim=(2, 1),  # Image is 2D, Audio is 1D
    
    # Encoders/Decoders for each modality
    modality_encoder=(ImageEncoder(), AudioMelEncoder()),
    modality_decoder=(ImageDecoder(), AudioMelDecoder()),
    
    # Axial positional embeddings
    add_pos_emb=(True, True),
    
    # Optional: Pre/post transformer processing (like U-Net blocks)
    pre_post_transformer_enc_dec=(
        # Image: downsample then upsample
        (nn.Conv2d(384, 512, 3, 2, 1), nn.ConvTranspose2d(512, 384, 3, 2, 1, 1)),
        # Audio: 1D convolutions
        (nn.Conv1d(256, 512, 3, 2, 1), nn.ConvTranspose1d(512, 256, 3, 2, 1, 1)),
    ),
    
    transformer=dict(
        dim=512,
        depth=12,
        dim_head=64,
        heads=8
    )
)
```

#### Step 4: Create Audio Dataset

```python
import torchaudio
from torch.utils.data import Dataset

class TextAudioDataset(Dataset):
    """Dataset for text-audio pairs."""
    
    def __init__(
        self,
        audio_dir: str,
        transcripts_file: str,
        sample_rate: int = 16000,
        max_audio_len: int = 16000 * 5  # 5 seconds
    ):
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.max_audio_len = max_audio_len
        
        # Load transcripts: {audio_file: text}
        self.items = self._load_transcripts(transcripts_file)
        
    def _load_transcripts(self, path):
        items = []
        with open(path) as f:
            for line in f:
                audio_file, text = line.strip().split('\t')
                items.append((audio_file, text))
        return items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        audio_file, text = self.items[idx]
        
        # Load audio
        waveform, sr = torchaudio.load(self.audio_dir / audio_file)
        
        # Resample if needed
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        
        # Convert to mono and truncate/pad
        waveform = waveform.mean(dim=0)
        if waveform.shape[0] > self.max_audio_len:
            waveform = waveform[:self.max_audio_len]
        
        # Encode text to tensor
        text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long)
        
        # Return as [text, (modality_type, audio)]
        # modality_type=1 for audio (assuming images are type 0)
        return text_tensor, (1, waveform)


class TextImageAudioDataset(Dataset):
    """Dataset with text, images, and audio."""
    
    def __getitem__(self, idx):
        text, image, audio = self._load_item(idx)
        
        # Structure: [text, image, text, audio] or any combination
        return [
            text_tensor,      # torch.long
            image_tensor,     # torch.float (type 0 implicit)
            caption_tensor,   # torch.long
            (1, audio_tensor) # Explicit type 1 for audio
        ]
```

#### Step 5: Training Script

```python
# train_audio.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from transfusion_pytorch import Transfusion, create_dataloader

# Define model with audio modality
model = Transfusion(
    num_text_tokens=256,
    dim_latent=(256,),  # Audio only for now
    channel_first_latent=(True,),
    modality_default_shape=((100,),),  # 100 time frames
    modality_num_dim=(1,),
    modality_encoder=(AudioMelEncoder(latent_dim=256),),
    modality_decoder=(AudioMelDecoder(latent_dim=256),),
    add_pos_emb=(True,),
    transformer=dict(
        dim=256,
        depth=8,
        dim_head=64,
        heads=8
    )
).cuda()

# Create EMA model for better sampling
ema_model = model.create_ema(beta=0.995)

# Dataset and dataloader
dataset = TextAudioDataset(
    audio_dir='./data/audio',
    transcripts_file='./data/transcripts.txt'
)

# Use Transfusion's custom collate function for variable-length modalities
dataloader = model.create_dataloader(dataset, batch_size=8, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
for epoch in range(100):
    for batch in dataloader:
        optimizer.zero_grad()
        
        loss = model(batch)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        ema_model.update()
        
        print(f'Loss: {loss.item():.4f}')
    
    # Sample every epoch
    if epoch % 10 == 0:
        sample = ema_model.sample(max_length=256, modality_steps=32)
        # Process and save sample
```

#### Step 6: Sampling Audio

```python
# Generate audio from text prompt
text_prompt = "Hello, this is a test."
prompt_tensor = torch.tensor([ord(c) for c in text_prompt], dtype=torch.long)

sample = model.sample(
    prompt=prompt_tensor,
    max_length=512,
    modality_steps=32,  # More steps = higher quality
    text_temperature=0.7
)

# Extract audio from sample
for item in sample:
    if isinstance(item, tuple):
        modality_type, audio = item
        if modality_type == 0:  # Audio is modality 0 in single-modality setup
            # audio is mel spectrogram or waveform depending on decoder
            save_audio(audio, 'generated.wav')

# Or generate audio only
audio = model.generate_modality_only(
    batch_size=4,
    modality_type=0,
    modality_steps=50
)
```

### 5.4 Audio-Specific Considerations

#### 1. Handling Variable-Length Audio

```python
# The model handles variable lengths automatically via:
# - Padding in collate function
# - Attention masking for different modality positions
# - Dynamic shape parsing from meta tokens during generation
```

#### 2. Time Resolution Trade-offs

| Approach | Time Frames/Sec | Quality | Speed |
|----------|-----------------|---------|-------|
| Raw waveform patches | 62.5 (16kHz/256) | Best | Slowest |
| Mel spectrogram | 62.5 (hop=256) | Good | Fast |
| Neural codec | 75 (EnCodec) | Excellent | Medium |

#### 3. Positional Embeddings for Audio

```python
# 1D continuous axial positional embeddings work well for audio
# The model uses ContinuousAxialPositionalEmbedding for this
add_pos_emb=(True,),
modality_num_dim=(1,),  # 1D for audio time axis
```

#### 4. Conditioning Options

```python
# Text-to-Audio: Use text as prompt
sample = model.sample(prompt=text_tensor, max_length=1024)

# Audio Continuation: Provide audio prefix as prompt
sample = model.sample(prompt=(0, audio_prefix), max_length=1024)

# Unconditional: Sample from scratch
sample = model.generate_modality_only(batch_size=1, modality_type=0)
```

### 5.5 Complete Training Example with Audio

```python
#!/usr/bin/env python3
"""train_audio.py - Train Transfusion on text-to-audio generation."""

from pathlib import Path
from shutil import rmtree

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torch.optim import AdamW

from transfusion_pytorch import Transfusion

# Config
SAMPLE_RATE = 16000
N_MELS = 80
HOP_LENGTH = 256
LATENT_DIM = 256
AUDIO_SECONDS = 4
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_STEPS = 50000
SAMPLE_EVERY = 1000

# Results directory
rmtree('./results_audio', ignore_errors=True)
results_folder = Path('./results_audio')
results_folder.mkdir(exist_ok=True, parents=True)


class MelEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.proj = nn.Conv1d(N_MELS, LATENT_DIM, 1)
        
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        mel = torch.log(self.mel(x).clamp(min=1e-5))
        mel = (mel - mel.mean()) / (mel.std() + 1e-5)
        return self.proj(mel)


class MelDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv1d(LATENT_DIM, N_MELS, 1)
        
    def forward(self, x):
        return self.proj(x)


# Calculate default audio shape
audio_frames = (SAMPLE_RATE * AUDIO_SECONDS) // HOP_LENGTH + 1

# Model
model = Transfusion(
    num_text_tokens=256,
    dim_latent=LATENT_DIM,
    channel_first_latent=True,
    modality_default_shape=(audio_frames,),
    modality_num_dim=1,
    modality_encoder=MelEncoder(),
    modality_decoder=MelDecoder(),
    add_pos_emb=True,
    transformer=dict(
        dim=256,
        depth=8,
        dim_head=64,
        heads=8
    )
).cuda()

ema_model = model.create_ema(beta=0.995)


# Simple synthetic dataset for testing
class SyntheticAudioDataset(Dataset):
    """Generates synthetic audio (sine waves) with text labels."""
    
    def __init__(self, size=10000):
        self.size = size
        self.sample_rate = SAMPLE_RATE
        self.duration = AUDIO_SECONDS
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Random frequency
        freq = 220 + (idx % 10) * 110  # A3 to ~A5
        note_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'A+', 'B+', 'C+']
        note = note_names[idx % 10]
        
        # Generate sine wave
        t = torch.linspace(0, self.duration, self.sample_rate * self.duration)
        waveform = torch.sin(2 * torch.pi * freq * t)
        
        # Add some noise
        waveform = waveform + 0.01 * torch.randn_like(waveform)
        
        # Text label
        text = f"Note {note}"
        text_tensor = torch.tensor([ord(c) for c in text], dtype=torch.long)
        
        return text_tensor, waveform.float()


dataset = SyntheticAudioDataset()
dataloader = model.create_dataloader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def cycle(loader):
    while True:
        for batch in loader:
            yield batch

iter_dl = cycle(dataloader)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for step in range(1, NUM_STEPS + 1):
    batch = next(iter_dl)
    
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    ema_model.update()
    
    if step % 100 == 0:
        print(f'Step {step}: loss={loss.item():.4f}')
    
    if step % SAMPLE_EVERY == 0:
        # Generate sample
        mel = ema_model.generate_modality_only(batch_size=1, modality_steps=32)
        
        # Save mel spectrogram as image
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 4))
        plt.imshow(mel[0].cpu().numpy(), aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(f'Generated Mel Spectrogram - Step {step}')
        plt.savefig(results_folder / f'mel_{step}.png')
        plt.close()

print("Training complete!")
```

---

## Summary

### Key Takeaways

1. **Transfusion unifies autoregressive (text) and flow-matching (continuous) generation** in one transformer
2. **Training uses combined loss**: cross-entropy for text + MSE for flow matching
3. **Sampling alternates between** autoregressive text decoding and ODE-based modality generation
4. **Adding audio is straightforward** because the architecture already supports multiple modalities with different:
   - Latent dimensions
   - Shapes (1D, 2D, etc.)
   - Encoders/decoders

### Recommended Next Steps for Audio

1. Start with mel spectrogram representation (simpler, proven)
2. Train on a small dataset first (e.g., LJSpeech for TTS)
3. Add a vocoder (HiFi-GAN, Vocos) for waveform reconstruction
4. Experiment with EnCodec for higher quality
5. Scale up to text-to-speech or music generation

### Architecture Extensions to Consider

- **Cross-modal attention**: Currently all modalities share the same attention. Could add modality-specific attention patterns
- **Classifier-free guidance**: Add unconditional training probability for guidance during sampling
- **Multi-resolution**: Generate audio at multiple temporal resolutions
- **Streaming**: Adapt for real-time audio generation

