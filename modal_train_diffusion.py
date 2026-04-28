import modal
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import HfApi, create_repo, upload_folder
import os

# Define the Modal App
app = modal.App("continuous-sign-language-diffusion")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "huggingface_hub",
    "datasets",
    "tqdm"
).add_local_dir(
    "/home/bence/continuous-sign-language-translation", 
    remote_path="/root/project"
)

@app.cls(image=image, gpu="A10G", secrets=[modal.Secret.from_name("my-huggingface-secret")])
class Phase1Trainer:
    def __init__(self, batch_size=32, lr=1e-4, epochs=50):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @modal.method()
    def train(self):
        import sys
        sys.path.append("/root/project")
        from models import SemanticEncoder, DiffusionDecoder
        from data import RealSignLanguageDataset
        from torch.utils.data import DataLoader
        
        print(f"Starting REAL Phase 1 training on {self.device}...")
        
        # Initialize models
        encoder = SemanticEncoder().to(self.device)
        decoder = DiffusionDecoder().to(self.device)
        
        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=self.lr)
        criterion = nn.MSELoss()
        
        # Real Dataset (Streaming)
        # Note: We use a small batch size for the stream
        dataset = RealSignLanguageDataset(split="train")
        
        for epoch in range(self.epochs):
            total_loss = 0
            count = 0
            
            # Manual batching for streaming dataset if not using a buffered loader
            batch_features = []
            for features, _ in dataset:
                batch_features.append(features)
                
                if len(batch_features) == self.batch_size:
                    batch = torch.stack(batch_features).to(self.device)
                    batch_features = []
                    
                    # Forward Phase 1: Encode
                    z = encoder(batch) # [B, D, T']
                    
                    # Diffusion Step
                    noise = torch.randn_like(batch)
                    t = torch.randint(0, 1000, (batch.shape[0], 1)).to(self.device)
                    alpha_t = (1 - t / 1000).view(-1, 1, 1)
                    noisy_batch = alpha_t * batch + (1 - alpha_t) * noise
                    
                    # Decode (Denoise)
                    pred_noise = decoder(noisy_batch, z, t)
                    loss = criterion(pred_noise, batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    count += 1
                    
                    if count % 100 == 0:
                        print(f"Epoch {epoch+1}, Batch {count}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1}/{self.epochs} Complete. Avg Loss: {total_loss/count:.4f}")
            
            # Save checkpoints periodically
            if (epoch + 1) % 10 == 0:
                self.save_and_upload(encoder, decoder)

    def save_and_upload(self, encoder, decoder):
        os.makedirs("/tmp/model", exist_ok=True)
        torch.save(encoder.state_dict(), "/tmp/model/semantic_encoder.pth")
        torch.save(decoder.state_dict(), "/tmp/model/diffusion_decoder.pth")
        
        repo_id = "bdanko/continuous-sign-language-translation"
        token = os.environ.get("HF_TOKEN")
        if token:
            print(f"Uploading checkpoints to {repo_id}")
            api = HfApi()
            create_repo(repo_id, token=token, exist_ok=True)
            upload_folder(
                folder_path="/tmp/model",
                repo_id=repo_id,
                token=token,
                commit_message="Training Checkpoint"
            )

@app.local_entrypoint()
def main():
    trainer = Phase1Trainer()
    trainer.train.remote()
