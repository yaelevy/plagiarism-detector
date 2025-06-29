#!/usr/bin/env python3
"""
Plagiarism Detection CLI - Siamese BERT Pipeline
Usage: python plagiarism_detector.py --corpus_path /path/to/corpus --csv_file mappings.csv [options]
"""

import argparse
import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Monkey patch for PyTorch compatibility
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.get_default_device = get_default_device

class PlagiarismDataset(Dataset):
    """Dataset class for plagiarism detection pairs"""
    
    def __init__(self, pairs: List[Dict], tokenizer, max_length: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Tokenize both texts
        encoding1 = self.tokenizer(
            pair['text1'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        encoding2 = self.tokenizer(
            pair['text2'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': encoding1['input_ids'].flatten(),
            'attention_mask_1': encoding1['attention_mask'].flatten(),
            'input_ids_2': encoding2['input_ids'].flatten(),
            'attention_mask_2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(pair['label'], dtype=torch.float)
        }

class SiameseBERT(nn.Module):
    """Siamese Network with BERT encoder and similarity classifier"""
    
    def __init__(self, model_name: str = 'bert-base-uncased', dropout: float = 0.3):
        super(SiameseBERT, self).__init__()
        
        # Shared BERT encoder
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Similarity classifier
        bert_dim = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(bert_dim * 3, 512),  # concatenated + absolute difference
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Get embeddings from both texts using shared BERT
        outputs_1 = self.bert(input_ids_1, attention_mask_1)
        outputs_2 = self.bert(input_ids_2, attention_mask_2)
        
        # Use [CLS] pooled output
        emb_1 = outputs_1.pooler_output
        emb_2 = outputs_2.pooler_output
        
        # Apply dropout
        emb_1 = self.dropout(emb_1)
        emb_2 = self.dropout(emb_2)
        
        # Create feature vector: [emb1, emb2, |emb1-emb2|]
        diff = torch.abs(emb_1 - emb_2)
        combined = torch.cat([emb_1, emb_2, diff], dim=1)
        
        # Predict similarity
        similarity = self.classifier(combined)
        return similarity.squeeze()

class PlagiarismDataLoader:
    """Data loader for plagiarism detection dataset"""
    
    def __init__(self, csv_file: str, corpus_path: str):
        self.csv_file = csv_file
        self.corpus_path = corpus_path
    
    def extract_passages_python(self, source_id: str, suspicious_id: str, 
                               source_offset: int, source_length: int,
                               suspicious_offset: int, suspicious_length: int) -> Tuple[str, str]:
        """Extract text passages from corpus files"""
        try:
            # Calculate part numbers
            source_num = int(source_id.split('document')[1].split('.')[0])
            source_part = (source_num - 1) // 500 + 1
            
            suspicious_num = int(suspicious_id.split('document')[1])
            suspicious_part = (suspicious_num - 1) // 500 + 1
            
            # Build file paths
            source_file = f"{self.corpus_path}/source-document/part{source_part}/{source_id}"
            suspicious_file = f"{self.corpus_path}/suspicious-document/part{suspicious_part}/{suspicious_id}.txt"
            
            # Read and extract passages
            with open(source_file, 'r', encoding='utf-8', errors='ignore') as f:
                source_text = f.read()
                source_passage = source_text[source_offset:source_offset + source_length]
            
            with open(suspicious_file, 'r', encoding='utf-8', errors='ignore') as f:
                suspicious_text = f.read()
                suspicious_passage = suspicious_text[suspicious_offset:suspicious_offset + suspicious_length]
            
            return source_passage.strip(), suspicious_passage.strip()
        
        except Exception as e:
            print(f"Error extracting passages: {e}")
            return None, None
    
    def load_plagiarism_data(self, sample_size: int = None) -> List[Dict]:
        """Load positive plagiarism pairs from CSV"""
        df = pd.read_csv(self.csv_file)
        
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=42)
        
        positive_pairs = []
        
        print(f"Loading {len(df)} plagiarism instances...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            source_passage, suspicious_passage = self.extract_passages_python(
                row['source_id'], row['suspicious_id'],
                int(row['source_offset']), int(row['source_length']),
                int(row['suspicious_offset']), int(row['suspicious_length'])
            )
            
            if source_passage and suspicious_passage and len(source_passage) > 50 and len(suspicious_passage) > 50:
                positive_pairs.append({
                    'text1': source_passage,
                    'text2': suspicious_passage,
                    'label': 1,
                    'obfuscation': row['obfuscation']
                })
        
        print(f"Loaded {len(positive_pairs)} valid positive pairs")
        return positive_pairs
    
    def create_negative_pairs(self, positive_pairs: List[Dict], ratio: float = 1.0) -> List[Dict]:
        """Create negative pairs by randomly pairing unrelated passages"""
        negative_pairs = []
        n_negatives = int(len(positive_pairs) * ratio)
        
        # Get all unique passages
        all_passages = []
        for pair in positive_pairs:
            all_passages.extend([pair['text1'], pair['text2']])
        
        print(f"Creating {n_negatives} negative pairs...")
        for _ in tqdm(range(n_negatives)):
            text1, text2 = random.sample(all_passages, 2)
            negative_pairs.append({
                'text1': text1,
                'text2': text2,
                'label': 0,
                'obfuscation': 'none'
            })
        
        return negative_pairs

class PlagiarismTrainer:
    """Trainer class for Siamese BERT model"""
    
    def __init__(self, model: SiameseBERT, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
    
    def train_epoch(self, dataloader: DataLoader, optimizer, scheduler=None) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Training"):
            optimizer.zero_grad()
            
            # Move batch to device
            input_ids_1 = batch['input_ids_1'].to(self.device)
            attention_mask_1 = batch['attention_mask_1'].to(self.device)
            input_ids_2 = batch['input_ids_2'].to(self.device)
            attention_mask_2 = batch['attention_mask_2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = F.binary_cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids_1 = batch['input_ids_1'].to(self.device)
                attention_mask_1 = batch['attention_mask_1'].to(self.device)
                input_ids_2 = batch['input_ids_2'].to(self.device)
                attention_mask_2 = batch['attention_mask_2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = F.binary_cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.cpu().detach().tolist())
                true_labels.extend(labels.cpu().detach().tolist())
        
        # Calculate metrics
        pred_labels = (np.array(predictions) > 0.5).astype(int)
        accuracy = accuracy_score(true_labels, pred_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        auc = roc_auc_score(true_labels, predictions)
        
        return {
            'loss': total_loss / len(dataloader),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 3, lr: float = 2e-5, output_dir: str = ".") -> None:
        """Full training loop"""
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        best_f1 = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            self.train_losses.append(train_loss)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_accuracies.append(val_metrics['accuracy'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Val F1: {val_metrics['f1']:.4f}")
            print(f"Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                model_path = os.path.join(output_dir, 'best_siamese_bert.pth')
                torch.save(self.model.state_dict(), model_path)
                print(f"Saved best model to {model_path}!")
        
        # Always save final model even if F1 didn't improve
        final_model_path = os.path.join(output_dir, 'best_siamese_bert.pth')
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")
        
        # Save training plot
        self.save_training_plot(output_dir)
    
    def save_training_plot(self, output_dir: str):
        """Save training history plot"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Plot accuracy
        axes[1].plot(self.val_accuracies, label='Val Accuracy')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'training_history.png')
        plt.savefig(plot_path)
        print(f"Training plot saved to {plot_path}")

def predict_similarity(model: SiameseBERT, tokenizer, text1: str, text2: str, 
                      device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """Predict similarity between two texts"""
    model.eval()
    
    # Tokenize texts
    encoding1 = tokenizer(text1, truncation=True, padding='max_length', 
                         max_length=512, return_tensors='pt')
    encoding2 = tokenizer(text2, truncation=True, padding='max_length', 
                         max_length=512, return_tensors='pt')
    
    # Move to device
    input_ids_1 = encoding1['input_ids'].to(device)
    attention_mask_1 = encoding1['attention_mask'].to(device)
    input_ids_2 = encoding2['input_ids'].to(device)
    attention_mask_2 = encoding2['attention_mask'].to(device)
    
    with torch.no_grad():
        similarity = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
    
    return similarity.item()

def main():
    parser = argparse.ArgumentParser(description='Plagiarism Detection with Siamese BERT')
    parser.add_argument('--corpus_path', required=True, help='Path to corpus directory')
    parser.add_argument('--csv_file', required=True, help='Path to plagiarism mappings CSV')
    parser.add_argument('--model_name', default='bert-base-uncased', help='BERT model name')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size for training')
    parser.add_argument('--output_dir', default='.', help='Output directory for models and plots')
    parser.add_argument('--test_text1', help='First text for similarity test')
    parser.add_argument('--test_text2', help='Second text for similarity test')
    parser.add_argument('--load_model', help='Path to pre-trained model for inference only')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.corpus_path):
        print(f"Error: Corpus path {args.corpus_path} does not exist")
        sys.exit(1)
    
    if not os.path.exists(args.csv_file):
        print(f"Error: CSV file {args.csv_file} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # If inference only
    if args.load_model and args.test_text1 and args.test_text2:
        model = SiameseBERT(args.model_name)
        model.load_state_dict(torch.load(args.load_model))
        similarity = predict_similarity(model, tokenizer, args.test_text1, args.test_text2)
        print(f"Similarity score: {similarity:.4f}")
        sys.exit(0)
    
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Corpus path: {args.corpus_path}")
    print(f"CSV file: {args.csv_file}")
    print(f"Sample size: {args.sample_size}")
    
    # Load data
    print("Loading data...")
    data_loader = PlagiarismDataLoader(args.csv_file, args.corpus_path)
    
    # Load positive pairs
    positive_pairs = data_loader.load_plagiarism_data(sample_size=args.sample_size)
    
    if len(positive_pairs) == 0:
        print("Error: No valid positive pairs found!")
        sys.exit(1)
    
    # Create negative pairs
    negative_pairs = data_loader.create_negative_pairs(positive_pairs, ratio=1.0)
    
    # Combine all pairs
    all_pairs = positive_pairs + negative_pairs
    random.shuffle(all_pairs)
    
    print(f"Total dataset: {len(all_pairs)} pairs")
    print(f"Positive: {len(positive_pairs)}, Negative: {len(negative_pairs)}")
    
    # Split data
    train_pairs, val_pairs = train_test_split(all_pairs, test_size=0.2, random_state=42, 
                                             stratify=[p['label'] for p in all_pairs])
    
    # Initialize datasets
    train_dataset = PlagiarismDataset(train_pairs, tokenizer, args.max_length)
    val_dataset = PlagiarismDataset(val_pairs, tokenizer, args.max_length)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model and trainer
    model = SiameseBERT(args.model_name)
    trainer = PlagiarismTrainer(model)
    
    # Train model
    print("Starting training...")
    trainer.train(train_loader, val_loader, epochs=args.epochs, 
                 lr=args.learning_rate, output_dir=args.output_dir)
    
    # Test prediction
    if len(positive_pairs) > 0:
        print("\nTesting prediction...")
        test_text1 = positive_pairs[0]['text1']
        test_text2 = positive_pairs[0]['text2']
        similarity = predict_similarity(model, tokenizer, test_text1, test_text2)
        print(f"Sample similarity score: {similarity:.4f}")
    
    print(f"\nTraining completed! Model saved in {args.output_dir}")

if __name__ == "__main__":
    main()