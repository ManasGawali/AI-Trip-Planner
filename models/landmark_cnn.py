"""
Lightweight CNN for Landmark Recognition
Uses MobileNetV3 - perfect for small datasets and fast inference
Model size: ~15MB, Training time: 30-60 mins on CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

class LightweightLandmarkCNN(nn.Module):
    """Lightweight CNN using MobileNetV3-Small (5.4M parameters)"""
    def __init__(self, num_classes, pretrained=True):
        super(LightweightLandmarkCNN, self).__init__()
        
        # Use MobileNetV3-Small - very lightweight and fast
        self.backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Replace classifier
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


class EasyLandmarkTrainer:
    """Easy-to-use trainer for landmark recognition"""
    
    def __init__(self, data_dir, model_save_dir='models/saved_models'):
        self.data_dir = Path(data_dir)
        self.model_save_dir = Path(model_save_dir)
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Data augmentation for training
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Simple transform for validation
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    def load_data(self, batch_size=16):
        """Load training and validation datasets"""
        print("Loading datasets...")
        
        train_dataset = datasets.ImageFolder(
            self.data_dir / 'train',
            transform=self.train_transform
        )
        
        val_dataset = datasets.ImageFolder(
            self.data_dir / 'val',
            transform=self.val_transform
        )
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        self.num_classes = len(train_dataset.classes)
        print(f"✓ Loaded {len(train_dataset)} training images")
        print(f"✓ Loaded {len(val_dataset)} validation images")
        print(f"✓ Number of landmark classes: {self.num_classes}")
        
        return self.num_classes
    
    def train(self, num_epochs=15, learning_rate=0.001, batch_size=16):
        """Train the model"""
        
        # Load data
        num_classes = self.load_data(batch_size)
        
        # Create model
        print("\nInitializing model...")
        self.model = LightweightLandmarkCNN(num_classes=num_classes).to(self.device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        best_val_acc = 0.0
        
        print(f"\nStarting training for {num_epochs} epochs...")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                                 'acc': f'{100.*train_correct/train_total:.1f}%'})
            
            train_loss = train_loss / len(self.train_loader)
            train_acc = 100. * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]  ")
                for images, labels in pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    pbar.set_postfix({'loss': f'{loss.item():.3f}', 
                                     'acc': f'{100.*val_correct/val_total:.1f}%'})
            
            val_loss = val_loss / len(self.val_loader)
            val_acc = 100. * val_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print("-" * 60)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model('landmark_model_best.pth', epoch, val_acc)
                print(f"  ✓ Saved new best model! (Val Acc: {val_acc:.2f}%)")
        
        print("\n" + "=" * 60)
        print(f"Training complete! Best validation accuracy: {best_val_acc:.2f}%")
        print("=" * 60)
        
        # Plot training history
        self.plot_history()
        
        return best_val_acc
    
    def save_model(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'val_acc': val_acc,
            'num_classes': self.num_classes
        }, self.model_save_dir / filename)
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.model_save_dir / 'training_history.png')
        print(f"\n✓ Training history plot saved to {self.model_save_dir / 'training_history.png'}")


class LandmarkPredictor:
    """Easy-to-use predictor for landmark recognition"""
    
    def __init__(self, model_path, landmark_info_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load landmark information
        with open(landmark_info_path, 'r') as f:
            self.landmark_info = json.load(f)
        
        # Convert string keys to integers
        self.landmark_info = {int(k): v for k, v in self.landmark_info.items()}
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        self.model = LightweightLandmarkCNN(num_classes=num_classes, pretrained=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded successfully!")
        print(f"  Can recognize {num_classes} landmarks")
        print(f"  Model accuracy: {checkpoint['val_acc']:.2f}%")
    
    def predict(self, image_path, top_k=3):
        """
        Predict landmark from image
        
        Args:
            image_path: Path to image file
            top_k: Return top K predictions
            
        Returns:
            List of predictions with landmark info
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, min(top_k, len(probabilities[0])))
        
        # Format results
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            class_idx = idx.item()
            if class_idx in self.landmark_info:
                info = self.landmark_info[class_idx]
                results.append({
                    'landmark': info['name'],
                    'city': info['city'],
                    'country': info['country'],
                    'latitude': info['lat'],
                    'longitude': info['lon'],
                    'confidence': prob.item() * 100
                })
        
        return results
    
    def predict_and_display(self, image_path):
        """Predict and display results with image"""
        from PIL import Image
        import matplotlib.pyplot as plt
        
        # Get predictions
        results = self.predict(image_path, top_k=3)
        
        # Display image
        image = Image.open(image_path)
        plt.figure(figsize=(10, 6))
        plt.imshow(image)
        plt.axis('off')
        
        # Add predictions as text
        result_text = "Top Predictions:\n\n"
        for i, result in enumerate(results, 1):
            result_text += f"{i}. {result['landmark']}\n"
            result_text += f"   {result['city']}, {result['country']}\n"
            result_text += f"   Confidence: {result['confidence']:.1f}%\n\n"
        
        plt.title(result_text, fontsize=10, loc='left')
        plt.tight_layout()
        plt.show()
        
        return results


# Example usage
if __name__ == "__main__":
    # Training
    print("="*60)
    print("LANDMARK RECOGNITION - TRAINING")
    print("="*60)
    
    trainer = EasyLandmarkTrainer(data_dir='data/landmarks_compact')
    best_acc = trainer.train(
        num_epochs=15,
        learning_rate=0.001,
        batch_size=16  # Use 8 if running on low-memory device
    )
    
    print("\n" + "="*60)
    print("LANDMARK RECOGNITION - TESTING")
    print("="*60)
    
    # Testing
    predictor = LandmarkPredictor(
        model_path='models/saved_models/landmark_model_best.pth',
        landmark_info_path='data/landmarks_compact/landmark_info.json'
    )
    
    # Example prediction
    # predictor.predict_and_display('path/to/test/image.jpg')