# CIFAR-10 Boosting Classification Project

This project uses three mainstream Boosting algorithms (XGBoost, LightGBM, CatBoost) to perform image classification experiments on the CIFAR-10 dataset, providing comprehensive performance evaluation and visualization analysis.

## 📋 Project Overview

CIFAR-10 is a classic computer vision dataset containing 32x32 color images across 10 categories. This project achieves efficient image classification through:

- **Multiple Feature Extraction**: Combines traditional computer vision features (HOG, LBP, edge features, etc.) with statistical features
- **Three Boosting Algorithms**: Complete implementation and comparison of XGBoost, LightGBM, and CatBoost
- **Hyperparameter Optimization**: Supports automatic hyperparameter tuning for optimal performance
- **Comprehensive Evaluation**: Includes accuracy, precision, recall, F1 score, and other metrics
- **Rich Visualization**: Confusion matrices, performance comparison charts, misclassified sample analysis, etc.

## 🏗️ Project Structure

```
PRML1.0/
├── data/                      # CIFAR-10 dataset storage directory
├── final_results/             # Final experiment results
│   └── plots/                 # Visualization charts
├── results/                   # Result files during training
│   ├── models/               # Saved model files
│   ├── training_history.json # Training history records
│   └── evaluation_results.json # Evaluation results
├── data_preprocessing.py      # Data preprocessing and feature engineering
├── models.py                 # Boosting model implementations
├── train_evaluate.py         # Training and evaluation pipeline
├── visualization.py          # Result visualization
├── main.py                   # Main program entry point
├── requirements.txt          # Project dependencies
└── README.md                # Project documentation
```

## 🚀 Quick Start

### Environment Requirements
- Python 3.7+
- CUDA support (optional, for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Project

#### 1. Quick Test Mode
```bash
python main.py --quick_test
```

#### 2. Full Training Mode
```bash
python main.py --enable_tuning
```

#### 3. Custom Parameter Run
```bash
python main.py --batch_size 128 --enable_tuning --tuning_iterations 100
```

### Command Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--batch_size` | int | 64 | Batch size |
| `--val_split` | float | 0.1 | Validation split ratio |
| `--enable_tuning` | flag | False | Enable hyperparameter tuning |
| `--tuning_iterations` | int | 50 | Hyperparameter tuning iterations |
| `--results_dir` | str | ./final_results | Results save directory |
| `--feature_engineering` | flag | True | Use feature engineering |
| `--quick_test` | flag | False | Quick test mode |

## 🔧 Core Features

### 1. Data Preprocessing (`data_preprocessing.py`)
- **Data Loading**: Automatically download and load CIFAR-10 dataset
- **Data Augmentation**: Random cropping, horizontal flipping, etc.
- **Feature Engineering**:
  - HOG (Histogram of Oriented Gradients) features
  - LBP (Local Binary Pattern) features
  - Edge detection features
  - Color histogram features
  - Texture features
  - Statistical features

### 2. Model Implementation (`models.py`)
- **XGBoost**: Gradient boosting decision trees
- **LightGBM**: Microsoft's efficient gradient boosting framework
- **CatBoost**: Yandex's gradient boosting algorithm
- **Unified Interface**: All models inherit from BaseBoostingModel base class
- **Hyperparameter Tuning**: Supports random search optimization

### 3. Training Evaluation (`train_evaluate.py`)
- **Complete Pipeline**: End-to-end process from data loading to model evaluation
- **Performance Metrics**: Accuracy, precision, recall, F1 score
- **Model Comparison**: Comprehensive comparison of multiple models
- **Result Saving**: Automatic saving of models and evaluation results

### 4. Visualization (`visualization.py`)
- **Performance Charts**: Accuracy, F1 score comparison charts
- **Confusion Matrices**: Detailed classification result analysis
- **Training Time Analysis**: Model efficiency comparison
- **Misclassified Samples**: Visual analysis of classification errors
- **Per-class Performance**: Detailed metrics for each category

## 📊 Expected Results

### Model Performance (Reference)
| Model | Accuracy | F1 Score | Training Time | Prediction Time |
|-------|----------|----------|---------------|-----------------|
| XGBoost | ~0.45 | ~0.44 | ~120s | ~0.05s |
| LightGBM | ~0.47 | ~0.46 | ~80s | ~0.03s |
| CatBoost | ~0.46 | ~0.45 | ~150s | ~0.04s |

*Note: Actual results may vary depending on hardware configuration and parameter settings*

### Generated Visualizations
- `performance_comparison.png`: Model performance comparison
- `confusion_matrices.png`: Confusion matrix heatmaps
- `time_comparison.png`: Training and prediction time comparison
- `per_class_performance.png`: Per-class detailed metrics
- `comprehensive_summary.png`: Comprehensive analysis summary
- `misclassified_samples_*.png`: Misclassified sample analysis

## 🛠️ Advanced Usage

### Custom Feature Engineering
```python
from data_preprocessing import CIFAR10DataLoader

# Enable advanced feature engineering
data_loader = CIFAR10DataLoader(
    batch_size=64,
    use_feature_engineering=True
)
```

### Manual Model Training
```python
from models import XGBoostModel
from train_evaluate import TrainingPipeline

# Create custom training pipeline
pipeline = TrainingPipeline(data_loader, "./custom_results")
pipeline.initialize_models()
pipeline.train_with_default_params()
```

### Custom Hyperparameter Tuning
```python
# Custom parameter grid
custom_params = {
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 300, 500]
}

model = XGBoostModel()
# Implement custom tuning logic
```

## 📈 Performance Optimization Tips

1. **Feature Engineering**: Enable feature engineering for better performance
2. **Hyperparameter Tuning**: Use `--enable_tuning` for optimal results
3. **Batch Size**: Adjust batch size based on available memory
4. **GPU Acceleration**: Ensure CUDA is available for faster training
5. **Data Augmentation**: Enabled by default for better generalization

## 🔍 Troubleshooting

### Common Issues

1. **Memory Error**
   - Reduce batch size: `--batch_size 32`
   - Disable feature engineering: remove `--feature_engineering`

2. **Slow Training**
   - Use quick test mode: `--quick_test`
   - Reduce tuning iterations: `--tuning_iterations 20`

3. **Import Errors**
   - Run setup test: `python test_setup.py`
   - Reinstall dependencies: `pip install -r requirements.txt`

### Performance Issues
- Ensure sufficient RAM (8GB+ recommended)
- Use SSD for faster data loading
- Close unnecessary applications during training

## 📚 Technical Details

### Algorithm Comparison
- **XGBoost**: Excellent performance, good interpretability
- **LightGBM**: Fastest training, memory efficient
- **CatBoost**: Best handling of categorical features, robust

### Feature Engineering Pipeline
1. Image normalization and preprocessing
2. HOG feature extraction (324 dimensions)
3. LBP feature extraction (26 dimensions)
4. Edge and color feature extraction
5. PCA dimensionality reduction
6. Feature standardization

### Evaluation Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: Per-class precision scores
- **Recall**: Per-class recall scores
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- CIFAR-10 dataset creators
- XGBoost, LightGBM, CatBoost development teams
- PyTorch and scikit-learn communities
- All contributors and users of this project

## 📞 Contact

For questions, suggestions, or issues, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation for common solutions

---

**Happy Coding! 🚀** 