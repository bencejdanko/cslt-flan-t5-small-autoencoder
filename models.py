import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticEncoder(nn.Module):
    def __init__(self, input_dim=540, latent_dim=512, temporal_compress=4):
        super().__init__()
        # Input: [B, T, F] -> [B, F, T] for 1D Conv
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1) # T -> T/2
        self.conv3 = nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1) # T/2 -> T/4
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(latent_dim)
        
    def forward(self, x):
        # x: [B, T, F]
        x = x.transpose(1, 2) # [B, F, T]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x # [B, D, T'] -> should be [B, 15, 512] if T=60

class DiffusionDecoder(nn.Module):
    """
    A simple 1D UNet for Diffusion, conditioned on Z.
    Z: [B, D, T']
    """
    def __init__(self, input_dim=540, latent_dim=512):
        super().__init__()
        self.input_dim = input_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.GELU(),
            nn.Linear(128, 128)
        )
        
        # Simple projection of Z to match temporal resolution if needed, 
        # but here we can just use cross-attention or upsampling.
        # Let's use upsampling + concatenation for simplicity in this baseline.
        self.upsample_z = nn.ConvTranspose1d(latent_dim, latent_dim, kernel_size=4, stride=4)
        
        self.net = nn.Sequential(
            nn.Conv1d(input_dim + latent_dim + 128, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(512, input_dim, kernel_size=3, padding=1)
        )

    def forward(self, x, z, t):
        # x: [B, T, F] -> [B, F, T]
        # z: [B, D, T']
        # t: [B, 1]
        
        x = x.transpose(1, 2) # [B, F, T]
        
        t_emb = self.time_mlp(t.float()).unsqueeze(-1).repeat(1, 1, x.shape[-1]) # [B, 128, T]
        z_upsampled = self.upsample_z(z) # [B, D, T]
        
        # Concatenate x, z_upsampled, and t_emb
        feat = torch.cat([x, z_upsampled, t_emb], dim=1)
        out = self.net(feat)
        
        return out.transpose(1, 2) # [B, T, F]

class TranslationModel(nn.Module):
    """
    Phase 2: Translates Z [B, 15, 512] to text using FLAN-T5.
    """
    def __init__(self, latent_dim=512):
        super().__init__()
        from transformers import T5ForConditionalGeneration
        # Load FLAN-T5-small
        self.t5 = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
        
        # We need to project Z to the T5 encoder hidden dimension if they don't match.
        # FLAN-T5-small d_model is 512, which matches our D=512.
        # If D was different, we'd need:
        # self.project = nn.Linear(latent_dim, self.t5.config.d_model)
        
    def forward(self, z, labels=None):
        # z: [B, 15, 512]
        # In T5, we can pass encoder_outputs directly or use the encoder.
        # However, it's easier to just pass z as the inputs_embeds to the encoder.
        
        if labels is not None:
            return self.t5(inputs_embeds=z, labels=labels)
        else:
            return self.t5.generate(inputs_embeds=z)
