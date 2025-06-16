"""
CIFAR-10 Data Preprocessing and Loading Module
Implements data loading, preprocessing, data augmentation, feature engineering functions
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage import feature, filters
from sklearn.feature_extraction import image
import cv2
import os

class CIFAR10DataLoader:
    """CIFAR-10 Data Loader"""
    
    def __init__(self, data_dir='./data', batch_size=32, val_split=0.1, 
                 random_state=42, use_feature_engineering=False):
        """
        Initialize data loader
        
        Args:
            data_dir: Data storage directory
            batch_size: Batch size
            val_split: Validation split ratio
            random_state: Random seed
            use_feature_engineering: Whether to use feature engineering
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.random_state = random_state
        self.use_feature_engineering = use_feature_engineering
        
        # CIFAR-10 class names
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
        # Data normalization parameters (CIFAR-10 mean and std)
        self.normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        
        # Feature engineering components
        self.pca = None
        self.scaler = StandardScaler()
        
        self._load_data()
    
    def _get_transforms(self):
        """Get data transformations"""
        # Data augmentation for training
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  # Random crop
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.ToTensor(),
            self.normalize
        ])
        
        # Data preprocessing for testing
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize
        ])
        
        return train_transform, test_transform
    
    def _load_data(self):
        """Load CIFAR-10 dataset"""
        print("Loading CIFAR-10 dataset...")
        
        train_transform, test_transform = self._get_transforms()
        
        # Load training and test sets
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=True, 
            transform=train_transform
        )
        
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=False, 
            download=True, 
            transform=test_transform
        )
        
        # Split training set into train and validation
        train_size = len(train_dataset)
        indices = list(range(train_size))
        
        # Set random seed for reproducibility
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        split_idx = int(train_size * (1 - self.val_split))
        train_indices, val_indices = indices[:split_idx], indices[split_idx:]
        
        # Create subsets
        train_subset = Subset(train_dataset, train_indices)
        val_subset = Subset(train_dataset, val_indices)
        
        # Create validation dataset without data augmentation
        val_dataset_no_aug = torchvision.datasets.CIFAR10(
            root=self.data_dir, 
            train=True, 
            download=False, 
            transform=test_transform
        )
        val_subset_no_aug = Subset(val_dataset_no_aug, val_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_subset_no_aug, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        print(f"Training set size: {len(train_subset)}")
        print(f"Validation set size: {len(val_subset)}")
        print(f"Test set size: {len(test_dataset)}")
    
    def _extract_hog_features(self, image):
        """Extract HOG features"""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # HOG parameters
        hog_features = feature.hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            visualize=False,
            feature_vector=True
        )
        return hog_features
    
    def _extract_lbp_features(self, image):
        """Extract LBP features"""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # LBP parameters
        radius = 3
        n_points = 8 * radius
        
        lbp = feature.local_binary_pattern(image, n_points, radius, method='uniform')
        
        # Calculate LBP histogram
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        
        return hist
    
    def _extract_edge_features(self, image):
        """Extract edge features"""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Convert to float for calculation
        image = image.astype(np.float64) / 255.0
        
        # Sobel edge detection
        sobel_x = filters.sobel_h(image)
        sobel_y = filters.sobel_v(image)
        
        # Edge magnitude and direction
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_direction = np.arctan2(sobel_y, sobel_x)
        
        # Statistical features
        features = [
            np.mean(edge_magnitude),
            np.std(edge_magnitude),
            np.mean(edge_direction),
            np.std(edge_direction),
            np.sum(edge_magnitude > np.percentile(edge_magnitude, 90))  # Strong edge pixel count
        ]
        
        return np.array(features)
    
    def _extract_color_features(self, image):
        """Extract color features"""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        features = []
        
        # Ensure 3-channel image
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB channel statistics
            for channel in range(3):
                ch_data = image[:, :, channel]
                features.extend([
                    np.mean(ch_data),
                    np.std(ch_data),
                    np.var(ch_data),
                    np.percentile(ch_data, 25),
                    np.percentile(ch_data, 75)
                ])
            
            # Overall image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray)
            ])
        else:
            # If grayscale image, use directly
            if len(image.shape) == 2:
                gray = image
            else:
                gray = image[:, :, 0]  # Take first channel
            
            # To maintain feature dimension consistency, replicate grayscale statistics
            gray_stats = [
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.percentile(gray, 25),
                np.percentile(gray, 75)
            ]
            features.extend(gray_stats * 3)  # Repeat 3 times to simulate RGB
            features.extend(gray_stats[:3])  # Add overall statistics
        
        return np.array(features)
    
    def _extract_entropy_features(self, image):
        """Extract entropy features"""
        # Ensure image is numpy array
        if not isinstance(image, np.ndarray):
            image = np.array(image)
        
        # Ensure correct data type
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif len(image.shape) == 3:
            image = image[:, :, 0]  # Take first channel
        
        # Calculate image entropy
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        hist = hist[hist > 0]  # Avoid log(0)
        
        if len(hist) > 0:
            entropy = -np.sum(hist * np.log2(hist))
        else:
            entropy = 0.0
        
        # Local entropy (sliding window)
        local_entropies = []
        window_size = 8
        for i in range(0, image.shape[0] - window_size + 1, window_size):
            for j in range(0, image.shape[1] - window_size + 1, window_size):
                window = image[i:i+window_size, j:j+window_size]
                window_hist, _ = np.histogram(window, bins=16, range=(0, 256))
                if window_hist.sum() > 0:
                    window_hist = window_hist / window_hist.sum()
                    window_hist = window_hist[window_hist > 0]
                    if len(window_hist) > 0:
                        local_entropy = -np.sum(window_hist * np.log2(window_hist))
                        local_entropies.append(local_entropy)
        
        if len(local_entropies) > 0:
            return np.array([entropy, np.mean(local_entropies), np.std(local_entropies)])
        else:
            return np.array([entropy, 0.0, 0.0])
    
    def _extract_advanced_features(self, image):
        """Extract all advanced features"""
        features = []
        
        try:
            # Ensure image is numpy array and format is correct
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            
            # Ensure image is in correct range
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            
            # Ensure correct image shape
            if len(image.shape) == 3 and image.shape[2] > 3:
                image = image[:, :, :3]  # Only take first three channels
            
            # HOG features
            hog_feat = self._extract_hog_features(image.copy())
            features.extend(hog_feat)
            
            # LBP features
            lbp_feat = self._extract_lbp_features(image.copy())
            features.extend(lbp_feat)
            
            # Edge features
            edge_feat = self._extract_edge_features(image.copy())
            features.extend(edge_feat)
            
            # Color features
            color_feat = self._extract_color_features(image.copy())
            features.extend(color_feat)
            
            # Entropy features
            entropy_feat = self._extract_entropy_features(image.copy())
            features.extend(entropy_feat)
            
        except Exception as e:
            print(f"Feature extraction warning: {e}")
            # If feature extraction fails, return fixed-size zero vector
            # Calculate expected feature dimensions: HOG(324) + LBP(26) + Edge(5) + Color(18) + Entropy(3)
            expected_features = 324 + 26 + 5 + 18 + 3  # 376
            features = [0.0] * expected_features
        
        return np.array(features)
    
    def get_data_for_boosting(self):
        """
        Prepare data for Boosting models
        Convert image data to flattened feature vectors with optional feature engineering
        """
        print("Preparing data for Boosting models...")
        
        def extract_features(loader, is_training=False):
            """Extract features and labels from DataLoader"""
            features = []
            labels = []
            advanced_features = []
            
            for batch_idx, (data, target) in enumerate(loader):
                batch_size = data.size(0)
                
                # Basic features: flattened pixel values
                flattened = data.view(batch_size, -1).numpy()
                features.append(flattened)
                labels.append(target.numpy())
                
                # Advanced feature engineering
                if self.use_feature_engineering:
                    batch_advanced_features = []
                    for i in range(batch_size):
                        # Denormalize image for feature extraction
                        img = data[i].clone()
                        for t, m, s in zip(img, self.normalize.mean, self.normalize.std):
                            t.mul_(s).add_(m)
                        img = torch.clamp(img, 0, 1)
                        img_np = img.permute(1, 2, 0).numpy()
                        
                        # Extract advanced features
                        adv_feat = self._extract_advanced_features(img_np)
                        batch_advanced_features.append(adv_feat)
                    
                    advanced_features.append(np.array(batch_advanced_features))
                
                # Show progress
                if batch_idx % 20 == 0:
                    print(f"Processing batch: {batch_idx}/{len(loader)}", end='\r')
            
            # Combine features from all batches
            all_features = np.vstack(features)
            all_labels = np.hstack(labels)
            
            if self.use_feature_engineering and advanced_features:
                all_advanced_features = np.vstack(advanced_features)
                # Combine basic and advanced features
                print(f"\nBasic features shape: {all_features.shape}")
                print(f"Advanced features shape: {all_advanced_features.shape}")
                all_features = np.hstack([all_features, all_advanced_features])
                print(f"Combined features shape: {all_features.shape}")
            
            return all_features, all_labels
        
        # Extract training, validation and test data
        print("Extracting training data...")
        X_train, y_train = extract_features(self.train_loader, is_training=True)
        
        print("\nExtracting validation data...")
        X_val, y_val = extract_features(self.val_loader)
        
        print("\nExtracting test data...")
        X_test, y_test = extract_features(self.test_loader)
        
        # Apply feature engineering post-processing
        if self.use_feature_engineering:
            print("\nApplying feature scaling...")
            # Standardize features
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
            
            # PCA dimensionality reduction
            print("Applying PCA dimensionality reduction...")
            n_components = min(1000, X_train.shape[1], X_train.shape[0] - 1)
            self.pca = PCA(n_components=n_components, random_state=self.random_state)
            X_train = self.pca.fit_transform(X_train)
            X_val = self.pca.transform(X_val)
            X_test = self.pca.transform(X_test)
            
            print(f"Feature dimensions after PCA: {X_train.shape[1]}")
            print(f"Cumulative explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        print(f"\nFinal data shapes:")
        print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
        print(f"Validation set: X_val {X_val.shape}, y_val {y_val.shape}")
        print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def get_sample_images(self, num_samples=10):
        """Get sample images for visualization"""
        samples = []
        labels = []
        
        for i, (data, target) in enumerate(self.test_loader):
            if len(samples) >= num_samples:
                break
            
            for j in range(min(num_samples - len(samples), data.size(0))):
                # Denormalize image
                img = data[j].clone()
                for t, m, s in zip(img, self.normalize.mean, self.normalize.std):
                    t.mul_(s).add_(m)
                img = torch.clamp(img, 0, 1)
                
                samples.append(img.permute(1, 2, 0).numpy())
                labels.append(target[j].item())
        
        return samples, labels

def test_data_loader():
    """Test data loader"""
    print("Testing data loader...")
    
    # Create data loader
    data_loader = CIFAR10DataLoader(batch_size=64)
    
    # Get data for Boosting models
    X_train, y_train, X_val, y_val, X_test, y_test = data_loader.get_data_for_boosting()
    
    # Verify data
    print(f"Feature dimensions: {X_train.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Training set label distribution: {np.bincount(y_train)}")
    
    # Get sample images
    samples, sample_labels = data_loader.get_sample_images(5)
    print(f"Sample image count: {len(samples)}")
    print(f"Sample labels: {[data_loader.classes[label] for label in sample_labels]}")
    
    print("Data loader test completed!")

if __name__ == "__main__":
    test_data_loader() 