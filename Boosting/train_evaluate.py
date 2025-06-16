"""
Training and Evaluation Module
Contains model training, hyperparameter tuning, performance evaluation functions
"""

import os
import time
import json
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

from models import XGBoostModel, LightGBMModel, CatBoostModel, ModelEvaluator
from data_preprocessing import CIFAR10DataLoader


class TrainingPipeline:
    """Training Pipeline"""
    
    def __init__(self, data_loader: CIFAR10DataLoader, save_dir: str = "./results"):
        """
        Initialize training pipeline
        
        Args:
            data_loader: Data loader
            save_dir: Directory to save results
        """
        self.data_loader = data_loader
        self.save_dir = save_dir
        self.models = {}
        self.results = {}
        self.training_history = {}
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare data
        self._prepare_data()
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.data_loader.classes)
    
    def _prepare_data(self):
        """Prepare training data"""
        print("Preparing training data...")
        
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = \
            self.data_loader.get_data_for_boosting()
        
        print(f"Data preparation completed:")
        print(f"Training set: {self.X_train.shape}")
        print(f"Validation set: {self.X_val.shape}")
        print(f"Test set: {self.X_test.shape}")
    
    def initialize_models(self):
        """Initialize all models"""
        print("Initializing models...")
        
        self.models = {
            'XGBoost': XGBoostModel(),
            'LightGBM': LightGBMModel(),
            'CatBoost': CatBoostModel()
        }
        
        print(f"Initialized {len(self.models)} models")
    
    def train_with_default_params(self):
        """Train all models with default parameters"""
        print("\n=== Training models with default parameters ===")
        
        for model_name, model in self.models.items():
            print(f"\nTraining {model_name} model...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Evaluate on validation set
            val_predictions = model.predict(self.X_val)
            val_accuracy = self.evaluator.evaluate_model(model, self.X_val, self.y_val)['accuracy']
            
            print(f"{model_name} validation accuracy: {val_accuracy:.4f}")
            
            # Save training history
            self.training_history[model_name] = {
                'default_params': model.get_default_params(),
                'validation_accuracy': val_accuracy,
                'training_time': model.training_time,
                'prediction_time': model.prediction_time
            }
    
    def hyperparameter_tuning(self, n_iter: int = 20):
        """Perform hyperparameter tuning"""
        print(f"\n=== Hyperparameter tuning (n_iter={n_iter}) ===")
        
        for model_name, model in self.models.items():
            print(f"\nTuning {model_name} model...")
            
            try:
                # Perform hyperparameter tuning
                best_params = model.hyperparameter_tuning(
                    self.X_train, self.y_train,
                    self.X_val, self.y_val,
                    n_iter=n_iter
                )
                
                # Retrain with best parameters
                model.fit(self.X_train, self.y_train, best_params)
                
                # Evaluate on validation set
                val_predictions = model.predict(self.X_val)
                val_accuracy = self.evaluator.evaluate_model(model, self.X_val, self.y_val)['accuracy']
                
                print(f"{model_name} validation accuracy after tuning: {val_accuracy:.4f}")
                
                # Save tuning history
                self.training_history[model_name].update({
                    'best_params': best_params,
                    'tuned_validation_accuracy': val_accuracy,
                    'tuned_training_time': model.training_time,
                    'tuned_prediction_time': model.prediction_time
                })
                
            except Exception as e:
                print(f"Error during {model_name} hyperparameter tuning: {str(e)}")
    
    def evaluate_all_models(self):
        """Evaluate all models on test set"""
        print("\n=== Test set performance evaluation ===")
        
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name} model...")
            
            # Evaluate model
            result = self.evaluator.evaluate_model(model, self.X_test, self.y_test)
            
            # Save results
            self.results[model_name] = result
            
            # Print results
            self.evaluator.print_evaluation_results(result)
    
    def get_misclassified_samples(self, model_name: str, num_samples: int = 10) -> Tuple[List, List, List]:
        """Get misclassified samples"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} does not exist")
        
        model = self.models[model_name]
        
        # Get predictions
        predictions = model.predict(self.X_test)
        
        # Find misclassified samples
        misclassified_indices = np.where(predictions != self.y_test)[0]
        
        if len(misclassified_indices) == 0:
            print(f"{model_name} has no misclassified samples")
            return [], [], []
        
        # Randomly select samples
        if len(misclassified_indices) > num_samples:
            selected_indices = np.random.choice(
                misclassified_indices, 
                size=num_samples, 
                replace=False
            )
        else:
            selected_indices = misclassified_indices
        
        # Get original image data
        sample_images = []
        true_labels = []
        pred_labels = []
        
        for idx in selected_indices:
            # Note: Here we need original image data, not processed features
            # This needs to be implemented based on actual requirements
            sample_images.append(self.X_test[idx])
            true_labels.append(self.y_test[idx])
            pred_labels.append(predictions[idx])
        
        return sample_images, true_labels, pred_labels
    
    def save_results(self):
        """Save training results"""
        print("\nSaving training results...")
        
        # Save models
        models_dir = os.path.join(self.save_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        for model_name, model in self.models.items():
            model_path = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Saved {model_name} model to {model_path}")
        
        # Save evaluation results
        results_path = os.path.join(self.save_dir, "evaluation_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_results = {}
        for model_name, result in self.results.items():
            json_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    json_result[key] = value.tolist()
                elif isinstance(value, np.integer):
                    json_result[key] = int(value)
                elif isinstance(value, np.floating):
                    json_result[key] = float(value)
                else:
                    json_result[key] = value
            json_results[model_name] = json_result
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation results to {results_path}")
        
        # Save training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        print(f"Saved training history to {history_path}")
    
    def load_results(self):
        """Load training results"""
        print("Loading training results...")
        
        # Load models
        models_dir = os.path.join(self.save_dir, "models")
        for model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            model_path = os.path.join(models_dir, f"{model_name.lower()}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} model")
        
        # Load evaluation results
        results_path = os.path.join(self.save_dir, "evaluation_results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r', encoding='utf-8') as f:
                json_results = json.load(f)
            
            # Convert lists back to numpy arrays
            for model_name, result in json_results.items():
                processed_result = {}
                for key, value in result.items():
                    if key == 'confusion_matrix' and isinstance(value, list):
                        processed_result[key] = np.array(value)
                    elif key in ['per_class_precision', 'per_class_recall', 'per_class_f1'] and isinstance(value, list):
                        processed_result[key] = np.array(value)
                    else:
                        processed_result[key] = value
                self.results[model_name] = processed_result
            print("Loaded evaluation results")
        
        # Load training history
        history_path = os.path.join(self.save_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r', encoding='utf-8') as f:
                self.training_history = json.load(f)
            print("Loaded training history")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.results:
            print("No evaluation results available")
            return {}
        
        summary = {}
        for model_name, result in self.results.items():
            summary[model_name] = {
                'accuracy': result['accuracy'],
                'macro_f1': result['macro_f1'],
                'training_time': result.get('training_time', 0),
                'prediction_time': result.get('prediction_time', 0)
            }
        
        return summary
    
    def print_performance_comparison(self):
        """Print performance comparison"""
        summary = self.get_performance_summary()
        
        if not summary:
            return
        
        print("\n=== Model Performance Comparison ===")
        print(f"{'Model':<10} {'Accuracy':<10} {'F1 Score':<10} {'Train Time':<12} {'Pred Time':<12}")
        print("-" * 60)
        
        for model_name, metrics in summary.items():
            print(f"{model_name:<10} {metrics['accuracy']:<10.4f} {metrics['macro_f1']:<10.4f} "
                  f"{metrics['training_time']:<12.2f} {metrics['prediction_time']:<12.6f}")
        
        # Find best models
        best_accuracy_model = max(summary.items(), key=lambda x: x[1]['accuracy'])
        best_f1_model = max(summary.items(), key=lambda x: x[1]['macro_f1'])
        fastest_training_model = min(summary.items(), key=lambda x: x[1]['training_time'])
        
        print(f"\nHighest accuracy: {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy']:.4f})")
        print(f"Highest F1 score: {best_f1_model[0]} ({best_f1_model[1]['macro_f1']:.4f})")
        print(f"Fastest training: {fastest_training_model[0]} ({fastest_training_model[1]['training_time']:.2f}s)")


def run_complete_pipeline(data_loader: CIFAR10DataLoader, save_dir: str = "./results", 
                         enable_tuning: bool = True, tuning_iterations: int = 20):
    """Run complete training pipeline"""
    print("Starting complete training pipeline...")
    
    # Create data loader
    print("Creating data loader...")
    # data_loader is already passed as parameter
    
    # Create training pipeline
    pipeline = TrainingPipeline(data_loader, save_dir)
    
    # Initialize models
    pipeline.initialize_models()
    
    # Train with default parameters
    pipeline.train_with_default_params()
    
    # Hyperparameter tuning (optional)
    if enable_tuning:
        pipeline.hyperparameter_tuning(n_iter=tuning_iterations)
    
    # Evaluate all models
    pipeline.evaluate_all_models()
    
    # Print performance comparison
    pipeline.print_performance_comparison()
    
    # Save results
    pipeline.save_results()
    
    return pipeline


if __name__ == "__main__":
    # Test code
    from data_preprocessing import CIFAR10DataLoader
    
    # Create data loader
    data_loader = CIFAR10DataLoader(batch_size=32, val_split=0.1)
    
    # Run pipeline
    pipeline = run_complete_pipeline(
        data_loader=data_loader,
        save_dir="./test_results",
        enable_tuning=False  # Disable tuning for quick test
    )
    
    print("Training pipeline test completed!") 