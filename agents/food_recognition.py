"""
Food recognition using transfer learning.

Uses a pretrained ResNet18 model fine-tuned on Food-101 subset to classify
food images and lookup nutritional information.
"""
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# Paths
MODEL_PATH = Path("models/food_classifier.pt")
NUTRITION_MAPPING_PATH = Path("data/food_class_to_nutrition.json")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FoodDataset(Dataset):
    """Dataset for food images."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_model(num_classes: int) -> nn.Module:
    """
    Create ResNet18 model with frozen early layers.

    Args:
        num_classes: Number of food classes to predict

    Returns:
        Model ready for training
    """
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers (keep first 3 layer blocks frozen)
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False

    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model.to(device)


def get_transforms(train: bool = True):
    """Get image transforms for training or inference."""
    if train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def train_model(
    data_dir: Path,
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001
) -> Tuple[nn.Module, List[str]]:
    """
    Train food classifier on Food-101 subset.

    Args:
        data_dir: Directory containing food class subdirectories
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate

    Returns:
        Tuple of (trained model, class names list)
    """
    print("=" * 70)
    print("Training Food Classifier")
    print("=" * 70 + "\n")

    # Get class directories
    class_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    num_classes = len(class_names)

    print(f"Found {num_classes} food classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    print()

    # Collect image paths and labels
    image_paths = []
    labels = []

    for class_idx, class_dir in enumerate(class_dirs):
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        image_paths.extend(images)
        labels.extend([class_idx] * len(images))
        print(f"  {class_dir.name}: {len(images)} images")

    print(f"\nTotal images: {len(image_paths)}\n")

    # Split into train/val (80/20)
    from sklearn.model_selection import train_test_split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Create datasets
    train_dataset = FoodDataset(train_paths, train_labels, get_transforms(train=True))
    val_dataset = FoodDataset(val_paths, val_labels, get_transforms(train=False))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train set: {len(train_dataset)} images")
    print(f"Val set: {len(val_dataset)} images\n")

    # Create model
    model = create_model(num_classes)
    print(f"Model: ResNet18 with {num_classes} output classes")
    print(f"Device: {device}\n")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=learning_rate)

    # Training loop
    print("Starting training...\n")
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0

        for images, lbls in train_loader:
            images, lbls = images.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == lbls).sum().item()

        train_acc = 100 * train_correct / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.no_grad():
            for images, lbls in val_loader:
                images, lbls = images.to(device), lbls.to(device)
                outputs = model(images)
                loss = criterion(outputs, lbls)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == lbls).sum().item()

        val_acc = 100 * val_correct / len(val_dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss/len(train_loader):.4f} "
              f"Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss/len(val_loader):.4f} "
              f"Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': class_names,
                'num_classes': num_classes
            }, MODEL_PATH)
            print(f"  → Best model saved (val acc: {val_acc:.2f}%)")

    print(f"\n[TRAINING COMPLETE] Best validation accuracy: {best_val_acc:.2f}%")
    print(f"[SAVE] Model saved to: {MODEL_PATH}\n")
    print("=" * 70)

    return model, class_names


def load_model() -> Tuple[nn.Module, List[str]]:
    """Load trained food classifier model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Train the model first."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = checkpoint['num_classes']

    model = create_model(num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, class_names


def predict_food(image_path: str, threshold: float = 0.6) -> Dict[str, any]:
    """
    Predict food class from image with confidence scores.

    Args:
        image_path: Path to food image
        threshold: Confidence threshold for direct answer (default 0.6)

    Returns:
        Dict with:
            - predictions: List of (class_name, confidence) tuples (top-3)
            - needs_confirmation: Boolean (True if top confidence < threshold)
            - message: Optional message for low confidence
    """
    model, class_names = load_model()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(train=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]

    # Get top-3 predictions
    top_probs, top_indices = torch.topk(probabilities, min(3, len(class_names)))

    predictions = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

    # Check if confidence is high enough
    top_confidence = predictions[0][1]
    needs_confirmation = top_confidence < threshold

    message = None
    if needs_confirmation:
        message = (f"I'm only {top_confidence*100:.1f}% confident about this prediction. "
                  f"Could you confirm if this is {predictions[0][0]}?")

    return {
        "predictions": predictions,
        "needs_confirmation": needs_confirmation,
        "message": message
    }


def load_nutrition_mapping() -> Dict[str, Dict[str, str]]:
    """Load food class to nutrition information mapping."""
    if not NUTRITION_MAPPING_PATH.exists():
        return {}

    with open(NUTRITION_MAPPING_PATH, 'r') as f:
        return json.load(f)


def save_nutrition_mapping(mapping: Dict[str, Dict[str, str]]) -> None:
    """Save food class to nutrition information mapping."""
    NUTRITION_MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NUTRITION_MAPPING_PATH, 'w') as f:
        json.dump(mapping, indent=2, fp=f)
    print(f"[SAVE] Nutrition mapping saved to: {NUTRITION_MAPPING_PATH}")


def lookup_nutrition(predicted_class: str) -> Optional[Dict[str, str]]:
    """
    Look up nutrition information for a predicted food class.

    Args:
        predicted_class: Predicted food class name

    Returns:
        Dict with nutrition info or None if not found
    """
    mapping = load_nutrition_mapping()
    return mapping.get(predicted_class)


def create_sample_data():
    """Create sample food images for testing (synthetic placeholder)."""
    print("Creating sample food dataset structure...\n")

    # Sample food classes (subset of Food-101)
    sample_classes = [
        "apple_pie",
        "pizza",
        "hamburger",
        "sushi",
        "ice_cream"
    ]

    data_dir = Path("data/raw/food101_subset")

    for food_class in sample_classes:
        class_dir = data_dir / food_class
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {class_dir}")

    print(f"\n[INFO] Please add training images to these directories.")
    print(f"[INFO] Recommended: 50-100 images per class for good results.\n")

    # Create sample nutrition mapping
    sample_mapping = {
        "apple_pie": {
            "category": "dessert",
            "main_ingredients": "apples, flour, sugar, butter",
            "nutrition_query": "What is the nutritional content of apple pie?",
            "notes": "High in carbohydrates and sugar. Contains fiber from apples."
        },
        "pizza": {
            "category": "main_dish",
            "main_ingredients": "dough, tomato sauce, cheese, toppings",
            "nutrition_query": "What are the nutritional facts for pizza?",
            "notes": "Varies greatly by toppings. Source of protein and calcium from cheese."
        },
        "hamburger": {
            "category": "main_dish",
            "main_ingredients": "beef patty, bun, lettuce, tomato, condiments",
            "nutrition_query": "What is the nutritional value of a hamburger?",
            "notes": "High in protein and iron. Choose lean beef for healthier option."
        },
        "sushi": {
            "category": "main_dish",
            "main_ingredients": "rice, fish, seaweed, vegetables",
            "nutrition_query": "What are the nutritional benefits of sushi?",
            "notes": "Good source of omega-3 fatty acids from fish. Rice provides carbohydrates."
        },
        "ice_cream": {
            "category": "dessert",
            "main_ingredients": "milk, cream, sugar, flavorings",
            "nutrition_query": "What is the nutritional information for ice cream?",
            "notes": "High in sugar and fat. Source of calcium. Consume in moderation."
        }
    }

    save_nutrition_mapping(sample_mapping)
    print()


def main():
    """Train model and test predictions."""
    print("=" * 70)
    print("Food Recognition System")
    print("=" * 70 + "\n")

    data_dir = Path("data/raw/food101_subset")

    # Check if data exists
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("[INFO] Training data not found. Creating sample structure...\n")
        create_sample_data()
        print("[SKIP] Cannot train without actual images.")
        print("[INFO] To train the model:")
        print("       1. Add food images to data/raw/food101_subset/<class_name>/")
        print("       2. Run this script again\n")
        return

    # Check if we have images
    total_images = sum(
        len(list(d.glob("*.jpg")) + list(d.glob("*.png")))
        for d in data_dir.iterdir() if d.is_dir()
    )

    if total_images == 0:
        print("[ERROR] No images found in training directories.")
        print("[INFO] Please add .jpg or .png images to:")
        for d in data_dir.iterdir():
            if d.is_dir():
                print(f"        {d}/")
        return

    # Train model if not exists
    if not MODEL_PATH.exists():
        print("[TRAIN] Starting model training...\n")
        model, class_names = train_model(data_dir, num_epochs=10, batch_size=16)
    else:
        print(f"[LOAD] Model already trained. Loading from {MODEL_PATH}\n")
        model, class_names = load_model()
        print(f"Loaded model with {len(class_names)} classes: {class_names}\n")

    # Test predictions on sample images
    test_images_dir = Path("data/samples/food_images")
    if test_images_dir.exists():
        test_images = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))

        if test_images:
            print("=" * 70)
            print("TESTING PREDICTIONS")
            print("=" * 70 + "\n")

            for img_path in test_images[:3]:  # Test on first 3 images
                print(f"Image: {img_path.name}")
                print("-" * 70)

                result = predict_food(str(img_path))

                print(f"Top-3 Predictions:")
                for i, (class_name, confidence) in enumerate(result["predictions"], 1):
                    print(f"  {i}. {class_name}: {confidence*100:.1f}%")

                if result["needs_confirmation"]:
                    print(f"\n⚠️  {result['message']}")
                else:
                    print(f"\n✓ High confidence prediction")

                # Lookup nutrition info
                top_class = result["predictions"][0][0]
                nutrition_info = lookup_nutrition(top_class)

                print(f"\nNutrition Lookup for '{top_class}':")
                if nutrition_info:
                    print(f"  Category: {nutrition_info.get('category', 'N/A')}")
                    print(f"  Ingredients: {nutrition_info.get('main_ingredients', 'N/A')}")
                    print(f"  Notes: {nutrition_info.get('notes', 'N/A')}")
                    print(f"  Query for details: {nutrition_info.get('nutrition_query', 'N/A')}")
                else:
                    print(f"  No nutrition mapping found for '{top_class}'")

                print("\n")
        else:
            print("[INFO] No test images found in data/samples/food_images/")

    print("=" * 70)


if __name__ == "__main__":
    main()
