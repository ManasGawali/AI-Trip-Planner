"""
Compact Landmark Dataset Builder
Creates a small, custom landmark dataset using web scraping
Final size: ~500MB for 100 landmarks with 100 images each
"""

import os
import requests
from pathlib import Path
from bing_image_downloader import downloader
import json

class CompactLandmarkDataset:
    def __init__(self, output_dir='data/landmarks_compact'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Top 100 world landmarks with their locations
        self.landmarks = {
            'Eiffel Tower': {'city': 'Paris', 'country': 'France', 'lat': 48.8584, 'lon': 2.2945},
            'Taj Mahal': {'city': 'Agra', 'country': 'India', 'lat': 27.1751, 'lon': 78.0421},
            'Statue of Liberty': {'city': 'New York', 'country': 'USA', 'lat': 40.6892, 'lon': -74.0445},
            'Big Ben': {'city': 'London', 'country': 'UK', 'lat': 51.5007, 'lon': -0.1246},
            'Colosseum': {'city': 'Rome', 'country': 'Italy', 'lat': 41.8902, 'lon': 12.4922},
            'Great Wall of China': {'city': 'Beijing', 'country': 'China', 'lat': 40.4319, 'lon': 116.5704},
            'Machu Picchu': {'city': 'Cusco', 'country': 'Peru', 'lat': -13.1631, 'lon': -72.5450},
            'Christ the Redeemer': {'city': 'Rio de Janeiro', 'country': 'Brazil', 'lat': -22.9519, 'lon': -43.2105},
            'Petra': {'city': 'Petra', 'country': 'Jordan', 'lat': 30.3285, 'lon': 35.4444},
            'Sydney Opera House': {'city': 'Sydney', 'country': 'Australia', 'lat': -33.8568, 'lon': 151.2153},
            'Burj Khalifa': {'city': 'Dubai', 'country': 'UAE', 'lat': 25.1972, 'lon': 55.2744},
            'Sagrada Familia': {'city': 'Barcelona', 'country': 'Spain', 'lat': 41.4036, 'lon': 2.1744},
            'Leaning Tower of Pisa': {'city': 'Pisa', 'country': 'Italy', 'lat': 43.7230, 'lon': 10.3966},
            'Stonehenge': {'city': 'Wiltshire', 'country': 'UK', 'lat': 51.1789, 'lon': -1.8262},
            'Angkor Wat': {'city': 'Siem Reap', 'country': 'Cambodia', 'lat': 13.4125, 'lon': 103.8670},
            'Golden Gate Bridge': {'city': 'San Francisco', 'country': 'USA', 'lat': 37.8199, 'lon': -122.4783},
            'Mount Rushmore': {'city': 'South Dakota', 'country': 'USA', 'lat': 43.8791, 'lon': -103.4591},
            'Neuschwanstein Castle': {'city': 'Bavaria', 'country': 'Germany', 'lat': 47.5576, 'lon': 10.7498},
            'Acropolis': {'city': 'Athens', 'country': 'Greece', 'lat': 37.9715, 'lon': 23.7267},
            'Chichen Itza': {'city': 'Yucatan', 'country': 'Mexico', 'lat': 20.6843, 'lon': -88.5678},
            'CN Tower': {'city': 'Toronto', 'country': 'Canada', 'lat': 43.6426, 'lon': -79.3871},
            'Gateway of India': {'city': 'Mumbai', 'country': 'India', 'lat': 18.9220, 'lon': 72.8347},
            'Brandenburg Gate': {'city': 'Berlin', 'country': 'Germany', 'lat': 52.5163, 'lon': 13.3777},
            'Red Square': {'city': 'Moscow', 'country': 'Russia', 'lat': 55.7539, 'lon': 37.6208},
            'Empire State Building': {'city': 'New York', 'country': 'USA', 'lat': 40.7484, 'lon': -73.9857},
            'Tower Bridge': {'city': 'London', 'country': 'UK', 'lat': 51.5055, 'lon': -0.0754},
            'Forbidden City': {'city': 'Beijing', 'country': 'China', 'lat': 39.9163, 'lon': 116.3972},
            'Victoria Falls': {'city': 'Livingstone', 'country': 'Zambia', 'lat': -17.9243, 'lon': 25.8572},
            'Alhambra': {'city': 'Granada', 'country': 'Spain', 'lat': 37.1761, 'lon': -3.5881},
            'Notre Dame': {'city': 'Paris', 'country': 'France', 'lat': 48.8530, 'lon': 2.3499},
            'Blue Mosque': {'city': 'Istanbul', 'country': 'Turkey', 'lat': 41.0055, 'lon': 28.9769},
            'Hagia Sophia': {'city': 'Istanbul', 'country': 'Turkey', 'lat': 41.0086, 'lon': 28.9802},
            'Table Mountain': {'city': 'Cape Town', 'country': 'South Africa', 'lat': -33.9628, 'lon': 18.4098},
            'Santorini': {'city': 'Santorini', 'country': 'Greece', 'lat': 36.3932, 'lon': 25.4615},
            'Mount Fuji': {'city': 'Fujinomiya', 'country': 'Japan', 'lat': 35.3606, 'lon': 138.7274},
            'Pyramids of Giza': {'city': 'Cairo', 'country': 'Egypt', 'lat': 29.9792, 'lon': 31.1342},
            'Buckingham Palace': {'city': 'London', 'country': 'UK', 'lat': 51.5014, 'lon': -0.1419},
            'Times Square': {'city': 'New York', 'country': 'USA', 'lat': 40.7580, 'lon': -73.9855},
            'Space Needle': {'city': 'Seattle', 'country': 'USA', 'lat': 47.6205, 'lon': -122.3493},
            'Niagara Falls': {'city': 'Niagara Falls', 'country': 'Canada', 'lat': 43.0962, 'lon': -79.0377},
            'Arc de Triomphe': {'city': 'Paris', 'country': 'France', 'lat': 48.8738, 'lon': 2.2950},
            'Louvre Museum': {'city': 'Paris', 'country': 'France', 'lat': 48.8606, 'lon': 2.3376},
            'Vatican City': {'city': 'Vatican', 'country': 'Vatican City', 'lat': 41.9029, 'lon': 12.4534},
            'Trevi Fountain': {'city': 'Rome', 'country': 'Italy', 'lat': 41.9009, 'lon': 12.4833},
            'Hollywood Sign': {'city': 'Los Angeles', 'country': 'USA', 'lat': 34.1341, 'lon': -118.3215},
            'Kremlin': {'city': 'Moscow', 'country': 'Russia', 'lat': 55.7520, 'lon': 37.6175},
            'Palace of Versailles': {'city': 'Versailles', 'country': 'France', 'lat': 48.8049, 'lon': 2.1204},
            'Windmills of Kinderdijk': {'city': 'Kinderdijk', 'country': 'Netherlands', 'lat': 51.8833, 'lon': 4.6333},
            'Mecca': {'city': 'Mecca', 'country': 'Saudi Arabia', 'lat': 21.4225, 'lon': 39.8262},
            'Chrysler Building': {'city': 'New York', 'country': 'USA', 'lat': 40.7516, 'lon': -73.9755},
        }
    
    def download_images(self, images_per_landmark=100, limit_landmarks=None):
        """
        Download images for each landmark using Bing Image Downloader
        
        Args:
            images_per_landmark: Number of images to download per landmark
            limit_landmarks: Limit to first N landmarks (for testing)
        """
        landmarks_to_process = list(self.landmarks.keys())
        if limit_landmarks:
            landmarks_to_process = landmarks_to_process[:limit_landmarks]
        
        print(f"Downloading images for {len(landmarks_to_process)} landmarks...")
        
        for idx, landmark_name in enumerate(landmarks_to_process, 1):
            print(f"\n[{idx}/{len(landmarks_to_process)}] Downloading: {landmark_name}")
            
            try:
                # Download images using Bing
                downloader.download(
                    landmark_name,
                    limit=images_per_landmark,
                    output_dir=str(self.output_dir),
                    adult_filter_off=True,
                    force_replace=False,
                    timeout=15,
                    verbose=False
                )
                print(f"✓ Downloaded images for {landmark_name}")
                
            except Exception as e:
                print(f"✗ Error downloading {landmark_name}: {e}")
    
    def organize_dataset(self):
        """
        Organize downloaded images into train/val split
        Rename folders to numeric indices for easier model training
        """
        from sklearn.model_selection import train_test_split
        import shutil
        
        print("\nOrganizing dataset...")
        
        # Create train/val directories
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        class_mapping = {}
        
        for idx, (landmark_name, info) in enumerate(self.landmarks.items()):
            # Source directory (created by bing_image_downloader)
            src_dir = self.output_dir / landmark_name
            
            if not src_dir.exists():
                print(f"Skipping {landmark_name} - no images found")
                continue
            
            # Get all images
            images = list(src_dir.glob('*.jpg')) + list(src_dir.glob('*.png'))
            
            if len(images) < 10:  # Skip if too few images
                print(f"Skipping {landmark_name} - only {len(images)} images")
                continue
            
            # Split into train/val
            train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
            
            # Create class directories
            train_class_dir = train_dir / str(idx)
            val_class_dir = val_dir / str(idx)
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Copy images
            for img in train_imgs:
                shutil.copy2(img, train_class_dir / img.name)
            for img in val_imgs:
                shutil.copy2(img, val_class_dir / img.name)
            
            # Save mapping
            class_mapping[idx] = {
                'name': landmark_name,
                **info
            }
            
            print(f"✓ Organized {landmark_name}: {len(train_imgs)} train, {len(val_imgs)} val")
        
        # Save class mapping
        with open(self.output_dir / 'landmark_info.json', 'w') as f:
            json.dump(class_mapping, f, indent=2)
        
        print(f"\n✓ Dataset organized! Total classes: {len(class_mapping)}")
        return class_mapping
    
    def get_dataset_stats(self):
        """Print dataset statistics"""
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        
        if not train_dir.exists():
            print("Dataset not organized yet. Run organize_dataset() first.")
            return
        
        total_train = sum(len(list(d.glob('*'))) for d in train_dir.iterdir() if d.is_dir())
        total_val = sum(len(list(d.glob('*'))) for d in val_dir.iterdir() if d.is_dir())
        num_classes = len(list(train_dir.iterdir()))
        
        print(f"\n{'='*50}")
        print(f"Dataset Statistics:")
        print(f"{'='*50}")
        print(f"Number of landmarks: {num_classes}")
        print(f"Training images: {total_train}")
        print(f"Validation images: {total_val}")
        print(f"Total images: {total_train + total_val}")
        print(f"{'='*50}\n")


if __name__ == "__main__":
    # Create dataset builder
    builder = CompactLandmarkDataset(output_dir='data/landmarks_compact')
    
    # Download images (start with 10 landmarks for testing)
    print("Starting download... This may take 15-30 minutes")
    builder.download_images(images_per_landmark=100, limit_landmarks=10)
    
    # Organize into train/val
    class_mapping = builder.organize_dataset()
    
    # Show statistics
    builder.get_dataset_stats()
    
    print("\n✓ Dataset ready for training!")
    print(f"Location: {builder.output_dir}")