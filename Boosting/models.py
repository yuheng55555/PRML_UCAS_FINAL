"""
Boosting Model Implementation Module
Contains XGBoost, LightGBM, CatBoost model wrappers
"""

import numpy as np
import time
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# Filter common warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='lightgbm')


class BaseBoostingModel(ABC):
    """Base class for Boosting models"""
    
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.training_time = 0
        self.prediction_time = 0
        self.is_fitted = False
    
    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters"""
        pass
    
    @abstractmethod
    def get_param_grid(self) -> Dict[str, Any]:
        """Get hyperparameter search space"""
        pass
    
    @abstractmethod
    def create_model(self, params: Dict[str, Any]):
        """Create model instance"""
        pass
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            params: Optional[Dict[str, Any]] = None):
        """Train model"""
        if params is None:
            params = self.get_default_params()
        
        print(f"Starting training {self.model_name} model...")
        start_time = time.time()
        
        self.model = self.create_model(params)
        self.model.fit(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_fitted = True
        
        print(f"{self.model_name} training completed, time: {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet, please call fit method first")
        
        start_time = time.time()
        try:
            # Ensure input is numpy array
            if hasattr(X, 'values'):
                X = X.values
            predictions = self.model.predict(X)
        except Exception as e:
            print(f"Warning during prediction: {e}")
            # Retry
            predictions = self.model.predict(X)
        
        self.prediction_time = time.time() - start_time
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet, please call fit method first")
        
        if hasattr(self.model, 'predict_proba'):
            try:
                # Ensure input is numpy array
                if hasattr(X, 'values'):
                    X = X.values
                return self.model.predict_proba(X)
            except Exception as e:
                print(f"Warning during probability prediction: {e}")
                return self.model.predict_proba(X)
        else:
            raise NotImplementedError(f"{self.model_name} does not support probability prediction")
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            n_iter: int = 20, cv: int = 3) -> Dict[str, Any]:
        """Hyperparameter tuning"""
        print(f"Starting {self.model_name} hyperparameter tuning...")
        
        param_grid = self.get_param_grid()
        base_model = self.create_model(self.get_default_params())
        
        # Use random search for hyperparameter tuning
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1
        )
        
        random_search.fit(X_train, y_train)
        
        self.best_params = random_search.best_params_
        
        # Evaluate best model on validation set
        val_score = random_search.score(X_val, y_val)
        
        print(f"{self.model_name} best parameters: {self.best_params}")
        print(f"{self.model_name} validation accuracy: {val_score:.4f}")
        
        return self.best_params


class XGBoostModel(BaseBoostingModel):
    """XGBoost model wrapper"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost", random_state)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get XGBoost default parameters - optimized version"""
        return {
            'objective': 'multi:softmax',
            'num_class': 10,
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
            'tree_method': 'hist'
        }
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Get XGBoost hyperparameter search space - extended version"""
        return {
            'max_depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'n_estimators': [300, 500, 800, 1000],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0],
            'gamma': [0, 0.1, 0.2]
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create XGBoost model"""
        return xgb.XGBClassifier(**params)


class LightGBMModel(BaseBoostingModel):
    """LightGBM model wrapper"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("LightGBM", random_state)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get LightGBM default parameters - optimized version"""
        return {
            'objective': 'multiclass',
            'num_class': 10,
            'max_depth': 10,
            'learning_rate': 0.05,
            'n_estimators': 800,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'num_leaves': 127,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'feature_fraction': 0.8
        }
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Get LightGBM hyperparameter search space - extended version"""
        return {
            'max_depth': [8, 10, 12, 15],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'n_estimators': [500, 800, 1000, 1500],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'num_leaves': [63, 127, 255, 511],
            'min_child_samples': [10, 20, 30],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.5, 1.0, 2.0],
            'feature_fraction': [0.7, 0.8, 0.9]
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create LightGBM model"""
        # Add force_col_wise parameter to avoid feature name warnings
        params_copy = params.copy()
        params_copy['force_col_wise'] = True
        return lgb.LGBMClassifier(**params_copy)


class CatBoostModel(BaseBoostingModel):
    """CatBoost model wrapper"""
    
    def __init__(self, random_state: int = 42):
        super().__init__("CatBoost", random_state)
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get CatBoost default parameters - optimized version"""
        return {
            'objective': 'MultiClass',
            'classes_count': 10,
            'depth': 8,
            'learning_rate': 0.05,
            'iterations': 800,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'bagging_temperature': 1,
            'random_strength': 1,
            'random_seed': self.random_state,
            'thread_count': -1,
            'verbose': False,
            'allow_writing_files': False
        }
    
    def get_param_grid(self) -> Dict[str, Any]:
        """Get CatBoost hyperparameter search space - extended version"""
        return {
            'depth': [6, 8, 10, 12],
            'learning_rate': [0.01, 0.03, 0.05, 0.08, 0.1],
            'iterations': [500, 800, 1000, 1500],
            'l2_leaf_reg': [1, 3, 5, 9],
            'border_count': [64, 128, 255],
            'bagging_temperature': [0, 1, 2],
            'random_strength': [1, 2, 3]
        }
    
    def create_model(self, params: Dict[str, Any]):
        """Create CatBoost model"""
        return cb.CatBoostClassifier(**params)


class ModelEvaluator:
    """Model evaluator"""
    
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def evaluate_model(self, model: BaseBoostingModel, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate model performance"""
        print(f"Evaluating {model.model_name} model performance...")
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate per-class precision, recall, F1 score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None, zero_division=0
        )
        
        # Calculate macro averages
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Organize results
        results = {
            'model_name': model.model_name,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'per_class_precision': precision,
            'per_class_recall': recall,
            'per_class_f1': f1,
            'per_class_support': support,
            'confusion_matrix': cm,
            'training_time': model.training_time,
            'prediction_time': model.prediction_time,
            'class_names': self.class_names
        }
        
        return results
    
    def print_evaluation_results(self, results: Dict[str, Any]):
        """Print evaluation results"""
        print(f"\n=== {results['model_name']} Model Evaluation Results ===")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Macro Average Precision: {results['macro_precision']:.4f}")
        print(f"Macro Average Recall: {results['macro_recall']:.4f}")
        print(f"Macro Average F1 Score: {results['macro_f1']:.4f}")
        print(f"Training Time: {results['training_time']:.2f}s")
        print(f"Prediction Time: {results['prediction_time']:.4f}s")
        
        print("\nPer-class Detailed Metrics:")
        print(f"{'Class':<8} {'Precision':<9} {'Recall':<8} {'F1 Score':<8} {'Support':<8}")
        print("-" * 50)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<8} {results['per_class_precision'][i]:<9.4f} "
                  f"{results['per_class_recall'][i]:<8.4f} "
                  f"{results['per_class_f1'][i]:<8.4f} "
                  f"{results['per_class_support'][i]:<8}")


def test_models():
    """Test model implementations"""
    print("Testing Boosting models...")
    
    # Generate mock data
    np.random.seed(42)
    X_train = np.random.rand(1000, 100)
    y_train = np.random.randint(0, 10, 1000)
    X_test = np.random.rand(200, 100)
    y_test = np.random.randint(0, 10, 200)
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create models
    models = [
        XGBoostModel(),
        LightGBMModel(),
        CatBoostModel()
    ]
    
    evaluator = ModelEvaluator(class_names)
    
    # Test each model
    for model in models:
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        results = evaluator.evaluate_model(model, X_test, y_test)
        evaluator.print_evaluation_results(results)
    
    print("Model testing completed!")


if __name__ == "__main__":
    test_models() 