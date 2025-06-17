import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
import time
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_cifar_data_optimized():
    """Load CIFAR-10 data using pickle files (same as your reference code)"""
    print("Loading CIFAR-10 dataset...")
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # Load training data
    X_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(f'cifar-10-batches-py/data_batch_{i}')
        X_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    
    # Load test data
    test_batch = unpickle('cifar-10-batches-py/test_batch')
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # Reshape data
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Load class names
    meta = unpickle('cifar-10-batches-py/batches.meta')
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    return X_train, y_train, X_test, y_test, class_names

class CIFAR10_KernelSVM:
    def __init__(self):
        self.class_names = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=50)
        self.svm = SVC(kernel='rbf', C=100.0, gamma='scale', probability=True, random_state=42)
        
    def load_and_preprocess_data(self):
        """Load and preprocess CIFAR-10 data"""
        # Load CIFAR-10 data using the same method as your reference code
        X_train, y_train, X_test, y_test, self.class_names = load_cifar_data_optimized()
        
        print(f"Original data shape: Train {X_train.shape}, Test {X_test.shape}")
        print(f"Class names: {self.class_names}")
        
        # Flatten images
        X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
        X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0
        
        # Combine all data for preprocessing
        X_all = np.vstack([X_train_flat, X_test_flat])
        y_all = np.hstack([y_train, y_test])
        
        # Standardization
        print("Applying standardization...")
        X_all_scaled = self.scaler.fit_transform(X_all)
        
        # PCA
        print("Applying PCA with 50 components...")
        X_all_pca = self.pca.fit_transform(X_all_scaled)
        
        # Split back
        X_train_processed = X_all_pca[:len(X_train)]
        X_test_processed = X_all_pca[len(X_train):]
        
        print(f"Processed data shape: Train {X_train_processed.shape}, Test {X_test_processed.shape}")
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
        
        # Store for visualization
        self.X_all_processed = X_all_pca
        self.y_all = y_all
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def plot_data_distribution(self, y_train, y_test):
        """Plot class distribution"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Training set distribution
        train_counts = np.bincount(y_train)
        ax1.bar(range(len(self.class_names)), train_counts, color='skyblue', alpha=0.7)
        ax1.set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes', fontsize=12)
        ax1.set_ylabel('Number of Samples', fontsize=12)
        ax1.set_xticks(range(len(self.class_names)))
        ax1.set_xticklabels(self.class_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(train_counts):
            ax1.text(i, v + 50, str(v), ha='center', va='bottom')
        
        # Test set distribution
        test_counts = np.bincount(y_test)
        ax2.bar(range(len(self.class_names)), test_counts, color='lightcoral', alpha=0.7)
        ax2.set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Classes', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_xticks(range(len(self.class_names)))
        ax2.set_xticklabels(self.class_names, rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(test_counts):
            ax2.text(i, v + 20, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('cifar10_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_sample_images(self, X_train, y_train):
        """Plot sample images from each class"""
        fig, axes = plt.subplots(2, 5, figsize=(15, 8))
        axes = axes.ravel()
        
        for i, class_name in enumerate(self.class_names):
            # Find first image of this class
            class_indices = np.where(y_train == i)[0]
            if len(class_indices) > 0:
                img = X_train[class_indices[0]]
                axes[i].imshow(img)
                axes[i].set_title(f'{class_name}', fontsize=12, fontweight='bold')
                axes[i].axis('off')
        
        plt.suptitle('Sample Images from Each Class', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('cifar10_sample_images.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pca_analysis(self):
        """Plot PCA analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Explained variance ratio
        ax1.plot(range(1, len(self.pca.explained_variance_ratio_) + 1), 
                self.pca.explained_variance_ratio_, 'bo-', markersize=4)
        ax1.set_title('PCA Explained Variance Ratio', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Principal Component', fontsize=12)
        ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        cumsum = np.cumsum(self.pca.explained_variance_ratio_)
        ax2.plot(range(1, len(cumsum) + 1), cumsum, 'ro-', markersize=4)
        ax2.axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
        ax2.set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Principal Component', fontsize=12)
        ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # First two principal components scatter
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        for i, class_name in enumerate(self.class_names):
            mask = self.y_all == i
            # Use a sample of points for better visualization
            sample_size = min(1000, np.sum(mask))
            indices = np.where(mask)[0]
            sample_indices = np.random.choice(indices, sample_size, replace=False)
            
            ax3.scatter(self.X_all_processed[sample_indices, 0], 
                       self.X_all_processed[sample_indices, 1], 
                       c=[colors[i]], label=class_name, alpha=0.6, s=10)
        
        ax3.set_title('First Two Principal Components', fontsize=14, fontweight='bold')
        ax3.set_xlabel('PC1', fontsize=12)
        ax3.set_ylabel('PC2', fontsize=12)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Component importance heatmap
        components_df = self.pca.components_[:10, :100]  # First 10 components, first 100 features
        im = ax4.imshow(components_df, cmap='RdBu_r', aspect='auto')
        ax4.set_title('PCA Components Heatmap (Top 10)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Original Features (First 100)', fontsize=12)
        ax4.set_ylabel('Principal Components', fontsize=12)
        plt.colorbar(im, ax=ax4)
        
        plt.tight_layout()
        plt.savefig('cifar10_pca_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, X_train, y_train):
        """Plot learning curves"""
        print("Generating learning curves...")
        
        # Use a subset for learning curves due to computational constraints
        subset_size = min(5000, len(X_train))
        indices = np.random.choice(len(X_train), subset_size, replace=False)
        X_subset = X_train[indices]
        y_subset = y_train[indices]
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.svm, X_subset, y_subset, cv=3, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
        )
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Accuracy')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        
        ax.set_title('Learning Curves - Kernel SVM on CIFAR-10', fontsize=14, fontweight='bold')
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cifar10_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def train_model(self, X_train, y_train):
        """Train the SVM model"""
        print("Training Kernel SVM model...")
        print(f"Parameters: kernel=rbf, C=100.0, gamma=scale")
        
        start_time = time.time()
        self.svm.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        print(f"Training completed in {training_time:.2f} seconds")
        print(f"Number of support vectors: {self.svm.n_support_}")
        
        return training_time
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the model and generate predictions"""
        print("Evaluating model...")
        
        start_time = time.time()
        y_pred = self.svm.predict(X_test)
        y_pred_proba = self.svm.predict_proba(X_test)
        prediction_time = time.time() - start_time
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Prediction time: {prediction_time:.2f} seconds")
        
        return y_pred, y_pred_proba, accuracy
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.class_names, yticklabels=self.class_names, ax=ax)
        ax.set_title('Confusion Matrix - Kernel SVM on CIFAR-10', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Labels', fontsize=12)
        ax.set_ylabel('True Labels', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('cifar10_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, y_test, y_pred):
        """Plot classification report as heatmap"""
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
        
        metrics_data = np.array([precision, recall, f1]).T
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 10))
        sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlOrRd',
                    xticklabels=['Precision', 'Recall', 'F1-Score'],
                    yticklabels=self.class_names, ax=ax)
        ax.set_title('Classification Metrics by Class', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('cifar10_classification_metrics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curves(self, y_test, y_pred_proba):
        """Plot ROC curves for each class"""
        # Binarize the output
        y_test_bin = label_binarize(y_test, classes=list(range(10)))
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('cifar10_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_summary(self, training_time, accuracy):
        """Plot training summary"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Model parameters
        params_text = f"""
        Preprocessing Parameters:
        • PCA Components: 50
        • Standardization: StandardScaler
        • Explained Variance: {self.pca.explained_variance_ratio_.sum():.3f}
        
        SVM Parameters:
        • Kernel: RBF
        • C: 100.0
        • Gamma: scale
        • Support Vectors: {sum(self.svm.n_support_)}
        
        Performance Results:
        • Training Time: {training_time:.2f}s
        • Test Accuracy: {accuracy:.4f}
        • Training Samples: 50,000
        • Test Samples: 10,000
        """
        ax1.text(0.05, 0.5, params_text, fontsize=11, transform=ax1.transAxes,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.set_title('Model Configuration & Results', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Explained variance
        ax2.bar(['Explained Variance'], [self.pca.explained_variance_ratio_.sum()], 
                color='green', alpha=0.7)
        ax2.set_title('PCA Explained Variance', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Ratio', fontsize=12)
        ax2.set_ylim([0, 1])
        for i, v in enumerate([self.pca.explained_variance_ratio_.sum()]):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Accuracy visualization
        ax3.bar(['Test Accuracy'], [accuracy], color='orange', alpha=0.7)
        ax3.set_title('Model Performance', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_ylim([0, 1])
        for i, v in enumerate([accuracy]):
            ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Support vectors distribution
        ax4.bar(self.class_names, self.svm.n_support_, color='purple', alpha=0.7)
        ax4.set_title('Support Vectors by Class', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Number of Support Vectors', fontsize=12)
        ax4.set_xticklabels(self.class_names, rotation=45)
        
        plt.tight_layout()
        plt.savefig('cifar10_training_summary.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("="*60)
        print("CIFAR-10 Kernel SVM Analysis")
        print("="*60)
        
        # Load and preprocess data
        X_train, X_test, y_train, y_test = self.load_and_preprocess_data()
        
        # Load original data for sample visualization
        X_train_orig, _, _, _, _ = load_cifar_data_optimized()
        
        # Generate visualizations
        print("\n1. Plotting sample images...")
        self.plot_sample_images(X_train_orig, y_train)
        
        print("\n2. Plotting data distribution...")
        self.plot_data_distribution(y_train, y_test)
        
        print("\n3. Plotting PCA analysis...")
        self.plot_pca_analysis()
        
        print("\n4. Generating learning curves...")
        self.plot_learning_curves(X_train, y_train)
        
        # Train model
        print("\n5. Training model...")
        training_time = self.train_model(X_train, y_train)
        
        # Evaluate model
        print("\n6. Evaluating model...")
        y_pred, y_pred_proba, accuracy = self.evaluate_model(X_test, y_test)
        
        # Generate evaluation plots
        print("\n7. Plotting confusion matrix...")
        self.plot_confusion_matrix(y_test, y_pred)
        
        print("\n8. Plotting classification metrics...")
        self.plot_classification_report(y_test, y_pred)
        
        print("\n9. Plotting ROC curves...")
        self.plot_roc_curves(y_test, y_pred_proba)
        
        print("\n10. Plotting training summary...")
        self.plot_training_summary(training_time, accuracy)
        
        # Print final classification report
        print("\n" + "="*60)
        print("FINAL CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        print(f"\nAll visualization plots have been saved as PNG files.")
        print("Analysis completed successfully!")

# Run the analysis
if __name__ == "__main__":
    # Change to the directory containing CIFAR-10 data
    # Make sure 'cifar-10-batches-py' folder is in the current directory
    
    analyzer = CIFAR10_KernelSVM()
    analyzer.run_complete_analysis()