import modal
import torch
import torch.nn as nn
import torch.optim as optim
from huggingface_hub import hf_hub_download, HfApi, upload_folder
from transformers import T5Tokenizer
import os

# Define the Modal App
app = modal.App("continuous-sign-language-translation")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install(
    "torch",
    "numpy",
    "huggingface_hub",
    "transformers",
    "sentencepiece",
    "datasets",
    "tqdm"
).add_local_dir(
    "/home/bence/continuous-sign-language-translation", 
    remote_path="/root/project"
)

@app.cls(image=image, gpu="A10G", secrets=[modal.Secret.from_name("my-huggingface-secret")])
class Phase2Trainer:
    def __init__(self, batch_size=8, lr=5e-5, epochs=20):
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @modal.method()
    def train(self):
        import sys
        sys.path.append("/root/project")
        from models import SemanticEncoder, TranslationModel
        from data import RealSignLanguageDataset
        
        print(f"Starting REAL Phase 2 translation training on {self.device}...")
        
        # 1. Load and Freeze Semantic Encoder
        repo_id = "bdanko/continuous-sign-language-translation"
        encoder = SemanticEncoder().to(self.device)
        
        try:
            weights_path = hf_hub_download(repo_id=repo_id, filename="semantic_encoder.pth")
            encoder.load_state_dict(torch.load(weights_path))
            print("Successfully loaded Semantic Encoder weights.")
        except Exception as e:
            print(f"FAILED to load encoder weights: {e}. Cannot proceed with REAL training.")
            return
            
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False
            
        # 2. Initialize Translation Model
        translation_model = TranslationModel().to(self.device)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
        
        optimizer = optim.AdamW(translation_model.parameters(), lr=self.lr)
        
        # 3. Real Dataset (Streaming)
        dataset = RealSignLanguageDataset(split="train")
        
        for epoch in range(self.epochs):
            total_loss = 0
            count = 0
            
            batch_z = []
            batch_labels = []
            
            for features, sentence in dataset:
                # Encode landmarks to Z
                with torch.no_grad():
                    # features is [T, F] -> [1, T, F]
                    z = encoder(features.unsqueeze(0).to(self.device)) # [1, D, T']
                
                batch_z.append(z.squeeze(0).transpose(0, 1)) # [15, 512]
                batch_labels.append(sentence)
                
                if len(batch_z) == self.batch_size:
                    z_tensor = torch.stack(batch_z) # [B, 15, 512]
                    
                    # Tokenize labels
                    labels = tokenizer(batch_labels, return_tensors="pt", padding=True, truncation=True).input_ids.to(self.device)
                    
                    # Forward Translation
                    outputs = translation_model(z_tensor, labels=labels)
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    count += 1
                    
                    batch_z = []
                    batch_labels = []
                    
                    if count % 100 == 0:
                        print(f"Epoch {epoch+1}, Batch {count}, Loss: {loss.item():.4f}")
            
            print(f"Epoch {epoch+1}/{self.epochs} Complete. Avg Loss: {total_loss/count:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(translation_model)

    def save_checkpoint(self, model):
        os.makedirs("/tmp/model_phase2", exist_ok=True)
        torch.save(model.state_dict(), "/tmp/model_phase2/translation_model.pth")
        
        repo_id = "bdanko/continuous-sign-language-translation"
        token = os.environ.get("HF_TOKEN")
        if token:
            api = HfApi()
            upload_folder(
                folder_path="/tmp/model_phase2",
                repo_id=repo_id,
                token=token,
                commit_message="Translation Checkpoint"
            )

@app.local_entrypoint()
def main():
    trainer = Phase2Trainer()
    trainer.train.remote()
