"""
CIFAR-10 Boosting Classification Project - Main Entry Point
Uses XGBoost, LightGBM, CatBoost three Boosting algorithms for CIFAR-10 dataset classification
Supports feature engineering, hyperparameter tuning and comprehensive performance evaluation
"""

import argparse
import time
import os
from typing import Dict, Any

from data_preprocessing import CIFAR10DataLoader
from train_evaluate import TrainingPipeline, run_complete_pipeline
from visualization import ModelVisualizer


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Boosting Classification Project')
    
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--val_split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    parser.add_argument('--enable_tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--tuning_iterations', type=int, default=50,
                       help='Hyperparameter tuning iterations (default: 50)')
    parser.add_argument('--results_dir', type=str, default='./final_results',
                       help='Results save directory (default: ./final_results)')
    parser.add_argument('--feature_engineering', action='store_true', default=True,
                       help='Use feature engineering (default: True)')
    parser.add_argument('--quick_test', action='store_true',
                       help='Run quick test mode')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("CIFAR-10 Boosting Classification Project")
    print("=" * 80)
    print(f"Batch size: {args.batch_size}")
    print(f"Validation split: {args.val_split}")
    print(f"Enable hyperparameter tuning: {args.enable_tuning}")
    print(f"Use feature engineering: {args.feature_engineering}")
    if args.enable_tuning:
        print(f"Tuning iterations: {args.tuning_iterations}")
    print(f"Results save directory: {args.results_dir}")
    print("=" * 80)
    
    # Record total runtime
    total_start_time = time.time()
    
    try:
        if args.quick_test:
            run_quick_test(args)
        else:
            run_complete_training_pipeline(args)
    
    except KeyboardInterrupt:
        print("\n\nUser interrupted program execution")
    except Exception as e:
        print(f"\n\nError occurred during program execution: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        total_time = time.time() - total_start_time
        print(f"\nTotal runtime: {total_time:.2f} seconds")
        print("Program execution completed")


def run_complete_training_pipeline(args):
    """Run complete training pipeline"""
    print("\nStarting complete training pipeline...")
    
    # Create data loader
    print("1. Creating data loader...")
    data_loader = CIFAR10DataLoader(
        batch_size=args.batch_size,
        val_split=args.val_split,
        use_feature_engineering=args.feature_engineering
    )
    
    # Create training pipeline
    print("2. Creating training pipeline...")
    pipeline = TrainingPipeline(data_loader, args.results_dir)
    
    # Initialize models
    print("3. Initializing models...")
    pipeline.initialize_models()
    
    # Train with default parameters
    print("4. Training models with default parameters...")
    pipeline.train_with_default_params()
    
    # Hyperparameter tuning (optional)
    if args.enable_tuning:
        print("5. Performing hyperparameter tuning...")
        pipeline.hyperparameter_tuning(n_iter=args.tuning_iterations)
    else:
        print("5. Skipping hyperparameter tuning")
    
    # Evaluate all models
    print("6. Evaluating all models...")
    pipeline.evaluate_all_models()
    
    # Print performance comparison
    print("7. Performance comparison...")
    pipeline.print_performance_comparison()
    
    # Save results
    print("8. Saving results...")
    pipeline.save_results()
    
    # Visualize results
    print("9. Generating visualization charts...")
    visualize_results(pipeline, args.results_dir)
    
    print("\nComplete training pipeline finished!")
    return pipeline


def run_quick_test(args):
    """Run quick test mode"""
    print("\nRunning quick test mode...")
    
    # Use smaller data amount for quick testing
    quick_data_loader = CIFAR10DataLoader(
        batch_size=128,  # Larger batch size for efficiency
        val_split=0.2,   # More validation data
        use_feature_engineering=False
    )
    
    pipeline = TrainingPipeline(quick_data_loader, "./final_results")
    
    # Initialize models
    pipeline.initialize_models()
    
    # Only train with default parameters (no tuning to save time)
    pipeline.train_with_default_params()
    
    # Evaluate models
    pipeline.evaluate_all_models()
    
    # Show results
    pipeline.print_performance_comparison()
    
    # Generate visualization
    visualize_results(pipeline, "./final_results")
    
    print("Quick test completed!")
    return pipeline


def visualize_results(pipeline: TrainingPipeline, results_dir: str):
    """Generate visualization results"""
    print("\nStarting visualization chart generation...")
    
    if not pipeline.results:
        print("No results available for visualization")
        return
    
    # Create visualizer
    plots_dir = os.path.join(results_dir, "plots")
    visualizer = ModelVisualizer(plots_dir)
    
    class_names = pipeline.data_loader.classes
    
    try:
        # 1. Performance comparison chart
        print("Generating performance comparison chart...")
        visualizer.plot_performance_comparison(pipeline.results)
        
        # 2. Training time comparison chart
        print("Generating training time comparison chart...")
        visualizer.plot_training_time_comparison(pipeline.results)
        
        # 3. Confusion matrices
        print("Generating confusion matrices...")
        visualizer.plot_confusion_matrices(pipeline.results, class_names)
        
        # 4. Per-class performance comparison
        print("Generating per-class performance comparison...")
        visualizer.plot_per_class_performance(pipeline.results, class_names)
        
        # 5. Comprehensive summary chart
        print("Generating comprehensive summary chart...")
        visualizer.plot_comprehensive_summary(pipeline.results, class_names)
        
        # 6. Misclassified samples (for each model)
        print("Generating misclassified samples charts...")
        for model_name in pipeline.models.keys():
            try:
                misclassified_images, true_labels, pred_labels = \
                    pipeline.get_misclassified_samples(model_name, num_samples=16)
                
                if misclassified_images:
                    visualizer.plot_misclassified_samples(
                        misclassified_images, true_labels, pred_labels,
                        class_names, model_name
                    )
            except Exception as e:
                print(f"Error generating misclassified samples for {model_name}: {str(e)}")
        
        print("All visualization charts generated successfully!")
        
    except Exception as e:
        print(f"Error during visualization generation: {str(e)}")
        import traceback
        traceback.print_exc()


def print_performance_analysis(results: Dict[str, Any]):
    """Print detailed performance analysis"""
    if not results:
        print("No results available for analysis")
        return
    
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE ANALYSIS")
    print("="*80)
    
    # Overall performance comparison
    print("\n1. OVERALL PERFORMANCE COMPARISON")
    print("-" * 40)
    
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    
    for metric in metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        model_scores = [(name, result[metric]) for name, result in results.items()]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(model_scores, 1):
            print(f"  {i}. {model_name}: {score:.4f}")
    
    # Training efficiency analysis
    print("\n2. TRAINING EFFICIENCY ANALYSIS")
    print("-" * 40)
    
    time_metrics = ['training_time', 'prediction_time']
    
    for metric in time_metrics:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        model_times = [(name, result[metric]) for name, result in results.items()]
        model_times.sort(key=lambda x: x[1])
        
        for i, (model_name, time_val) in enumerate(model_times, 1):
            unit = "seconds" if "training" in metric else "seconds"
            print(f"  {i}. {model_name}: {time_val:.4f} {unit}")
    
    # Best model recommendations
    print("\n3. MODEL RECOMMENDATIONS")
    print("-" * 40)
    
    best_accuracy = max(results.items(), key=lambda x: x[1]['accuracy'])
    best_f1 = max(results.items(), key=lambda x: x[1]['macro_f1'])
    fastest_train = min(results.items(), key=lambda x: x[1]['training_time'])
    fastest_pred = min(results.items(), key=lambda x: x[1]['prediction_time'])
    
    print(f"Best Accuracy: {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Best F1 Score: {best_f1[0]} ({best_f1[1]['macro_f1']:.4f})")
    print(f"Fastest Training: {fastest_train[0]} ({fastest_train[1]['training_time']:.2f}s)")
    print(f"Fastest Prediction: {fastest_pred[0]} ({fastest_pred[1]['prediction_time']:.4f}s)")
    
    # Balance score (accuracy vs speed)
    print(f"\n4. BALANCE SCORE (Accuracy/Training_Time)")
    print("-" * 40)
    
    balance_scores = []
    for name, result in results.items():
        balance_score = result['accuracy'] / (result['training_time'] + 1)  # +1 to avoid division by very small numbers
        balance_scores.append((name, balance_score))
    
    balance_scores.sort(key=lambda x: x[1], reverse=True)
    
    for i, (model_name, score) in enumerate(balance_scores, 1):
        print(f"  {i}. {model_name}: {score:.6f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main() 