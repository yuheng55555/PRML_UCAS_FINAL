"""
Visualization Module
Contains visualization functions for training curves, confusion matrices, performance comparisons, etc.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import os

# Set font and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

class ModelVisualizer:
    """Model Visualizer"""
    
    def __init__(self, save_dir: str = "./results/plots"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set color theme
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        self.model_colors = {
            'XGBoost': '#1f77b4',
            'LightGBM': '#ff7f0e', 
            'CatBoost': '#2ca02c'
        }
    
    def plot_confusion_matrices(self, results: Dict[str, Any], class_names: List[str]):
        """Plot confusion matrix heatmaps"""
        print("Generating confusion matrices...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            cm = result['confusion_matrix']
            
            # Calculate percentages
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            # Plot heatmap
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                ax=axes[idx],
                xticklabels=class_names,
                yticklabels=class_names,
                cbar_kws={'label': 'Percentage (%)'}
            )
            
            axes[idx].set_title(f'{model_name} Confusion Matrix', fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Predicted Label', fontsize=12)
            axes[idx].set_ylabel('True Label', fontsize=12)
            
            # Rotate x-axis labels
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'confusion_matrices.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved to: {save_path}")
        
        plt.show()
    
    def plot_performance_comparison(self, results: Dict[str, Any]):
        """Plot performance metrics comparison"""
        print("Generating performance comparison...")
        
        # Prepare data
        metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        model_names = list(results.keys())
        metric_values = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            result = results[model_name]
            for metric in metrics:
                metric_values[metric].append(result[metric])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            # Plot bar chart
            bars = ax.bar(
                model_names, 
                metric_values[metric],
                color=[self.model_colors[name] for name in model_names],
                alpha=0.8,
                edgecolor='black',
                linewidth=1
            )
            
            ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, value in zip(bars, metric_values[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Performance comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_training_time_comparison(self, results: Dict[str, Any]):
        """Plot training time comparison"""
        print("Generating training time comparison...")
        
        model_names = list(results.keys())
        training_times = [results[name]['training_time'] for name in model_names]
        prediction_times = [results[name]['prediction_time'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training time comparison
        bars1 = ax1.bar(
            model_names, 
            training_times,
            color=[self.model_colors[name] for name in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        
        ax1.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars1, training_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + max(training_times)*0.01,
                    f'{value:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # Prediction time comparison
        bars2 = ax2.bar(
            model_names, 
            prediction_times,
            color=[self.model_colors[name] for name in model_names],
            alpha=0.8,
            edgecolor='black',
            linewidth=1
        )
        
        ax2.set_title('Prediction Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, value in zip(bars2, prediction_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(prediction_times)*0.01,
                    f'{value:.4f}s', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'time_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Time comparison saved to: {save_path}")
        
        plt.show()
    
    def plot_per_class_performance(self, results: Dict[str, Any], class_names: List[str]):
        """Plot per-class performance comparison"""
        print("Generating per-class performance...")
        
        # Prepare data
        metrics = ['per_class_precision', 'per_class_recall', 'per_class_f1']
        metric_names = ['Precision', 'Recall', 'F1 Score']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            # Create DataFrame for plotting
            data = {}
            for model_name, result in results.items():
                data[model_name] = result[metric]
            
            df = pd.DataFrame(data, index=class_names)
            
            # Plot grouped bar chart
            df.plot(kind='bar', ax=ax, 
                   color=[self.model_colors[name] for name in results.keys()],
                   alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_title(f'Per-Class {metric_name} Comparison', fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_xlabel('Class', fontsize=12)
            ax.legend(title='Model', loc='upper right')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'per_class_performance.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Per-class performance saved to: {save_path}")
        
        plt.show()
    
    def plot_misclassified_samples(self, misclassified_images: List[np.ndarray], 
                                 true_labels: List[int], pred_labels: List[int],
                                 class_names: List[str], model_name: str,
                                 max_samples: int = 16):
        """Plot misclassified samples"""
        print(f"Generating {model_name} misclassified samples...")
        
        n_samples = min(len(misclassified_images), max_samples)
        if n_samples == 0:
            print(f"{model_name} has no misclassified samples")
            return
        
        # Calculate subplot layout
        n_cols = 4
        n_rows = (n_samples + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
        
        for i in range(n_samples):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            
            # Display image
            ax.imshow(misclassified_images[i])
            ax.axis('off')
            
            # Set title
            true_class = class_names[true_labels[i]]
            pred_class = class_names[pred_labels[i]]
            ax.set_title(f'True: {true_class}\nPred: {pred_class}', 
                        fontsize=10, color='red')
        
        # Hide extra subplots
        for i in range(n_samples, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row][col] if n_rows > 1 else axes[col]
            ax.axis('off')
        
        plt.suptitle(f'{model_name} Misclassified Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, f'{model_name.lower()}_misclassified.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"{model_name} misclassified samples saved to: {save_path}")
        
        plt.show()
    
    def plot_comprehensive_summary(self, results: Dict[str, Any], class_names: List[str]):
        """Plot comprehensive summary"""
        print("Generating comprehensive summary...")
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Overall performance comparison (top left)
        ax1 = plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        
        bars = ax1.bar(model_names, accuracies, 
                      color=[self.model_colors[name] for name in model_names],
                      alpha=0.8, edgecolor='black')
        ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_ylim(0, 1)
        
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Training time comparison (top right)
        ax2 = plt.subplot(2, 3, 2)
        training_times = [results[name]['training_time'] for name in model_names]
        
        bars = ax2.bar(model_names, training_times,
                      color=[self.model_colors[name] for name in model_names],
                      alpha=0.8, edgecolor='black')
        ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (seconds)', fontsize=12)
        
        for bar, time in zip(bars, training_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + max(training_times)*0.01,
                    f'{time:.1f}s', ha='center', va='bottom')
        
        # 3. F1 score comparison (top center)
        ax3 = plt.subplot(2, 3, 3)
        f1_scores = [results[name]['macro_f1'] for name in model_names]
        
        bars = ax3.bar(model_names, f1_scores,
                      color=[self.model_colors[name] for name in model_names],
                      alpha=0.8, edgecolor='black')
        ax3.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('F1 Score', fontsize=12)
        ax3.set_ylim(0, 1)
        
        for bar, f1 in zip(bars, f1_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 4. Best model confusion matrix (bottom, span 3 columns)
        best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
        best_model_name, best_result = best_model
        
        ax4 = plt.subplot(2, 1, 2)
        cm = best_result['confusion_matrix']
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=ax4,
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Percentage (%)'})
        
        ax4.set_title(f'Best Model ({best_model_name}) Confusion Matrix', 
                     fontsize=14, fontweight='bold')
        ax4.set_xlabel('Predicted Label', fontsize=12)
        ax4.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        # Save plot
        save_path = os.path.join(self.save_dir, 'comprehensive_summary.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comprehensive summary saved to: {save_path}")
        
        plt.show()


def test_visualizer():
    """Test visualization functions"""
    print("Testing visualization functions...")
    
    # Mock test data
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Mock results data
    results = {
        'XGBoost': {
            'accuracy': 0.85,
            'macro_precision': 0.83,
            'macro_recall': 0.82,
            'macro_f1': 0.83,
            'training_time': 120.5,
            'prediction_time': 0.45,
            'confusion_matrix': np.random.randint(5, 50, (10, 10)),
            'per_class_precision': np.random.uniform(0.7, 0.9, 10),
            'per_class_recall': np.random.uniform(0.7, 0.9, 10),
            'per_class_f1': np.random.uniform(0.7, 0.9, 10),
        },
        'LightGBM': {
            'accuracy': 0.82,
            'macro_precision': 0.81,
            'macro_recall': 0.80,
            'macro_f1': 0.80,
            'training_time': 95.2,
            'prediction_time': 0.38,
            'confusion_matrix': np.random.randint(5, 50, (10, 10)),
            'per_class_precision': np.random.uniform(0.7, 0.9, 10),
            'per_class_recall': np.random.uniform(0.7, 0.9, 10),
            'per_class_f1': np.random.uniform(0.7, 0.9, 10),
        },
        'CatBoost': {
            'accuracy': 0.83,
            'macro_precision': 0.82,
            'macro_recall': 0.81,
            'macro_f1': 0.81,
            'training_time': 180.7,
            'prediction_time': 0.52,
            'confusion_matrix': np.random.randint(5, 50, (10, 10)),
            'per_class_precision': np.random.uniform(0.7, 0.9, 10),
            'per_class_recall': np.random.uniform(0.7, 0.9, 10),
            'per_class_f1': np.random.uniform(0.7, 0.9, 10),
        }
    }
    
    # Create visualizer
    visualizer = ModelVisualizer("./test_plots")
    
    # Test various visualization functions
    visualizer.plot_performance_comparison(results)
    visualizer.plot_training_time_comparison(results)
    visualizer.plot_confusion_matrices(results, class_names)
    visualizer.plot_per_class_performance(results, class_names)
    visualizer.plot_comprehensive_summary(results, class_names)
    
    print("Visualization test completed!")


if __name__ == "__main__":
    test_visualizer() 