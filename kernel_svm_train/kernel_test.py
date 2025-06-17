import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
import itertools
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 新增多线程相关导入
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
from queue import Queue
import multiprocessing as mp
from functools import partial

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_cifar_data_optimized():
    """优化的CIFAR-10数据加载函数"""
    print("正在加载CIFAR-10数据...")
    
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # 加载训练数据
    X_train = []
    y_train = []
    for i in range(1, 6):
        batch = unpickle(f'cifar-10-batches-py/data_batch_{i}')
        X_train.append(batch[b'data'])
        y_train.extend(batch[b'labels'])
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    
    # 加载测试数据
    test_batch = unpickle('cifar-10-batches-py/test_batch')
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    
    # 重塑数据
    X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    # 加载类别标签
    meta = unpickle('cifar-10-batches-py/batches.meta')
    class_names = [name.decode('utf-8') for name in meta[b'label_names']]
    
    return X_train, y_train, X_test, y_test, class_names

def preprocess_data_worker(X_train, y_train, X_test, y_test, preprocess_params, 
                          train_sample_size, test_sample_size, random_seed):
    """独立的数据预处理函数，用于多线程"""
    np.random.seed(random_seed)
    
    # 随机采样
    train_indices = np.random.choice(len(X_train), 
                                   min(train_sample_size, len(X_train)), 
                                   replace=False)
    test_indices = np.random.choice(len(X_test), 
                                  min(test_sample_size, len(X_test)), 
                                  replace=False)
    
    X_train_sample = X_train[train_indices]
    y_train_sample = y_train[train_indices]
    X_test_sample = X_test[test_indices]
    y_test_sample = y_test[test_indices]
    
    # 展平图像
    X_train_flat = X_train_sample.reshape(len(X_train_sample), -1)
    X_test_flat = X_test_sample.reshape(len(X_test_sample), -1)
    
    # 标准化或最小-最大标准化
    if preprocess_params['normalization'] == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)
    
    # PCA降维
    if preprocess_params['use_pca']:
        pca_components = preprocess_params['pca_components']
        pca = PCA(n_components=min(pca_components, X_train_scaled.shape[1]))
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
    
    return X_train_scaled, y_train_sample, X_test_scaled, y_test_sample

def evaluate_single_combination(args):
    """评估单个参数组合的函数，用于多进程"""
    (combination_id, preprocess_params, svm_params, X_train, y_train, X_test, y_test, 
     train_sample_size, test_sample_size, cv_folds) = args
    
    try:
        start_time = time.time()
        
        # 使用组合ID作为随机种子，确保可重现性
        random_seed = combination_id + 42
        
        # 预处理数据
        X_train_processed, y_train_processed, X_test_processed, y_test_processed = \
            preprocess_data_worker(X_train, y_train, X_test, y_test, preprocess_params,
                                 train_sample_size, test_sample_size, random_seed)
        
        # 创建SVM模型
        model = SVC(**svm_params, random_state=random_seed)
        
        # 交叉验证
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
        cv_scores = cross_val_score(model, X_train_processed, y_train_processed, 
                                  cv=cv, scoring='accuracy', n_jobs=1)
        
        # 训练模型并在测试集上评估
        model.fit(X_train_processed, y_train_processed)
        test_score = model.score(X_test_processed, y_test_processed)
        
        # 记录结果
        result = {
            'combination_id': combination_id,
            'preprocessing': preprocess_params.copy(),
            'svm_params': svm_params.copy(),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_score': test_score,
            'training_time': time.time() - start_time,
            'success': True,
            'error': None
        }
        
        return result
        
    except Exception as e:
        return {
            'combination_id': combination_id,
            'preprocessing': preprocess_params.copy(),
            'svm_params': svm_params.copy(),
            'success': False,
            'error': str(e),
            'training_time': time.time() - start_time if 'start_time' in locals() else 0
        }

class OptimizedSVMHyperparameterOptimizer:
    def __init__(self, train_sample_size=2000, test_sample_size=500, cv_folds=3, 
                 n_workers=None, use_multiprocessing=True):
        """
        初始化优化的SVM超参数优化器
        
        Args:
            train_sample_size: 训练样本数量
            test_sample_size: 测试样本数量  
            cv_folds: 交叉验证折数
            n_workers: 工作线程/进程数，None表示自动检测
            use_multiprocessing: 是否使用多进程而不是多线程
        """
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.cv_folds = cv_folds
        self.n_workers = n_workers or min(mp.cpu_count(), 8)  # 限制最大进程数
        self.use_multiprocessing = use_multiprocessing
        self.results = []
        self.best_params = None
        self.best_score = 0
        
        print(f"使用 {'多进程' if use_multiprocessing else '多线程'} 并行计算")
        print(f"工作进程/线程数: {self.n_workers}")
        
        # 定义参数空间
        self.param_space = {
            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'C': [0.01, 0.1, 1, 5, 10, 50, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10],
            'degree': [2, 3, 4, 5],  # 仅用于poly核
            'coef0': [0, 0.1, 0.5, 1, 2]  # 用于poly和sigmoid核
        }
        
        # 数据预处理选项
        self.preprocessing_options = {
            'use_pca': [False, True],
            'pca_components': [50, 100, 200, 500],
            'normalization': ['standard', 'minmax']
        }
    
    def get_valid_param_combinations(self):
        """获取有效的参数组合"""
        combinations = []
        
        for kernel in self.param_space['kernel']:
            for C in self.param_space['C']:
                if kernel == 'linear':
                    combinations.append({'kernel': kernel, 'C': C})
                elif kernel == 'rbf':
                    for gamma in self.param_space['gamma']:
                        combinations.append({'kernel': kernel, 'C': C, 'gamma': gamma})
                elif kernel == 'poly':
                    for gamma in self.param_space['gamma']:
                        for degree in self.param_space['degree']:
                            for coef0 in self.param_space['coef0']:
                                combinations.append({
                                    'kernel': kernel, 'C': C, 'gamma': gamma,
                                    'degree': degree, 'coef0': coef0
                                })
                elif kernel == 'sigmoid':
                    for gamma in self.param_space['gamma']:
                        for coef0 in self.param_space['coef0']:
                            combinations.append({
                                'kernel': kernel, 'C': C, 'gamma': gamma, 'coef0': coef0
                            })
        
        return combinations
    
    def get_preprocessing_combinations(self):
        """获取预处理组合"""
        combinations = []
        for use_pca in self.preprocessing_options['use_pca']:
            for normalization in self.preprocessing_options['normalization']:
                if use_pca:
                    for pca_components in self.preprocessing_options['pca_components']:
                        combinations.append({
                            'use_pca': use_pca,
                            'pca_components': pca_components,
                            'normalization': normalization
                        })
                else:
                    combinations.append({
                        'use_pca': use_pca,
                        'pca_components': None,
                        'normalization': normalization
                    })
        return combinations
    
    def optimize_hyperparameters(self, X_train, y_train, X_test, y_test):
        """多线程/多进程超参数优化"""
        print("开始多线程/多进程超参数优化...")
        print(f"训练样本数: {self.train_sample_size}")
        print(f"测试样本数: {self.test_sample_size}")
        print("=" * 80)
        
        # 生成所有参数组合
        param_combinations = self.get_valid_param_combinations()
        preprocessing_combinations = self.get_preprocessing_combinations()
        
        # 创建所有任务
        all_tasks = []
        combination_id = 0
        
        for preprocess_params in preprocessing_combinations:
            for svm_params in param_combinations:
                task = (
                    combination_id, preprocess_params, svm_params,
                    X_train, y_train, X_test, y_test,
                    self.train_sample_size, self.test_sample_size, self.cv_folds
                )
                all_tasks.append(task)
                combination_id += 1
        
        total_combinations = len(all_tasks)
        print(f"总共需要测试 {total_combinations} 种参数组合")
        print("=" * 80)
        
        # 执行并行计算
        completed_tasks = 0
        successful_tasks = 0
        start_time = time.time()
        
        # 创建进度追踪
        progress_lock = threading.Lock()
        
        def update_progress(future):
            nonlocal completed_tasks, successful_tasks
            with progress_lock:
                completed_tasks += 1
                result = future.result()
                if result['success']:
                    successful_tasks += 1
                    self.results.append(result)
                    
                    # 更新最佳结果
                    if result['test_score'] > self.best_score:
                        self.best_score = result['test_score']
                        self.best_params = {
                            'preprocessing': result['preprocessing'],
                            'svm_params': result['svm_params']
                        }
                
                # 打印进度
                if completed_tasks % max(1, total_combinations // 20) == 0 or completed_tasks <= 10:
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed_tasks
                    remaining_time = avg_time_per_task * (total_combinations - completed_tasks)
                    
                    print(f"[{completed_tasks}/{total_combinations}] "
                          f"成功: {successful_tasks}, "
                          f"已用时: {elapsed_time:.1f}s, "
                          f"预计剩余: {remaining_time:.1f}s")
                    
                    if result['success']:
                        print(f"  最新结果: 测试分数 {result['test_score']:.4f}, "
                              f"当前最佳: {self.best_score:.4f}")
        
        # 选择执行器
        if self.use_multiprocessing:
            executor_class = ProcessPoolExecutor
            print("使用多进程执行...")
        else:
            executor_class = ThreadPoolExecutor
            print("使用多线程执行...")
        
        # 执行任务
        with executor_class(max_workers=self.n_workers) as executor:
            # 提交所有任务
            futures = []
            for task in all_tasks:
                future = executor.submit(evaluate_single_combination, task)
                future.add_done_callback(update_progress)
                futures.append(future)
            
            # 等待所有任务完成
            try:
                for future in as_completed(futures):
                    pass  # 回调函数已经处理了结果
            except KeyboardInterrupt:
                print("\n检测到中断信号，正在停止...")
                for future in futures:
                    future.cancel()
                raise
        
        total_time = time.time() - start_time
        print("\n" + "=" * 80)
        print("并行优化完成！")
        print(f"总时间: {total_time:.2f}秒")
        print(f"成功完成: {successful_tasks}/{total_combinations}")
        print(f"平均每个组合: {total_time/completed_tasks:.2f}秒")
        print(f"最佳测试分数: {self.best_score:.4f}")
        print("=" * 80)
        
        return self.results
    
    def optimize_hyperparameters_batch(self, X_train, y_train, X_test, y_test, batch_size=50):
        """批量处理优化，适用于内存限制的情况"""
        print("开始批量多线程超参数优化...")
        print(f"批量大小: {batch_size}")
        print("=" * 80)
        
        # 生成所有参数组合
        param_combinations = self.get_valid_param_combinations()
        preprocessing_combinations = self.get_preprocessing_combinations()
        
        # 创建所有任务
        all_tasks = []
        combination_id = 0
        
        for preprocess_params in preprocessing_combinations:
            for svm_params in param_combinations:
                task = (
                    combination_id, preprocess_params, svm_params,
                    X_train, y_train, X_test, y_test,
                    self.train_sample_size, self.test_sample_size, self.cv_folds
                )
                all_tasks.append(task)
                combination_id += 1
        
        total_combinations = len(all_tasks)
        print(f"总共需要测试 {total_combinations} 种参数组合")
        
        # 分批处理
        for batch_start in range(0, total_combinations, batch_size):
            batch_end = min(batch_start + batch_size, total_combinations)
            batch_tasks = all_tasks[batch_start:batch_end]
            
            print(f"\n处理批次 {batch_start//batch_size + 1}: "
                  f"任务 {batch_start+1}-{batch_end}")
            
            # 处理当前批次
            batch_start_time = time.time()
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = [executor.submit(evaluate_single_combination, task) 
                          for task in batch_tasks]
                
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result['success']:
                        self.results.append(result)
                        
                        # 更新最佳结果
                        if result['test_score'] > self.best_score:
                            self.best_score = result['test_score']
                            self.best_params = {
                                'preprocessing': result['preprocessing'],
                                'svm_params': result['svm_params']
                            }
                    
                    # 打印批次内进度
                    if (i + 1) % max(1, len(batch_tasks) // 5) == 0:
                        print(f"  批次进度: {i+1}/{len(batch_tasks)}, "
                              f"当前最佳: {self.best_score:.4f}")
            
            batch_time = time.time() - batch_start_time
            print(f"批次完成时间: {batch_time:.2f}秒")
        
        print("\n" + "=" * 80)
        print("批量优化完成！")
        print(f"最佳测试分数: {self.best_score:.4f}")
        print("=" * 80)
        
        return self.results

    # ...existing code... (保留原有的分析和可视化方法)
    
    def analyze_results(self):
        """分析优化结果"""
        if not self.results:
            print("没有结果可分析")
            return
        
        # 转换为DataFrame
        results_data = []
        for result in self.results:
            row = {}
            # 添加预处理参数
            for key, value in result['preprocessing'].items():
                row[f'prep_{key}'] = value
            # 添加SVM参数
            for key, value in result['svm_params'].items():
                row[f'svm_{key}'] = value
            # 添加性能指标
            row['cv_mean'] = result['cv_mean']
            row['cv_std'] = result['cv_std']
            row['test_score'] = result['test_score']
            row['training_time'] = result['training_time']
            
            results_data.append(row)
        
        df = pd.DataFrame(results_data)
        
        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f'svm_optimization_results_parallel_{timestamp}.csv', index=False)
        print(f"详细结果已保存到: svm_optimization_results_parallel_{timestamp}.csv")
        
        # 分析最佳结果
        self._visualize_results(df)
        self._print_top_results(df)
        
        return df
    
    def _visualize_results(self, df):
        """可视化结果 - 保持原有实现"""
        plt.figure(figsize=(20, 15))
        
        # 1. 不同核函数的性能对比
        plt.subplot(3, 4, 1)
        kernel_performance = df.groupby('svm_kernel')['test_score'].agg(['mean', 'std', 'max'])
        kernel_performance['mean'].plot(kind='bar', yerr=kernel_performance['std'], 
                                       capsize=5, color='skyblue', alpha=0.7)
        plt.title('不同核函数平均性能')
        plt.ylabel('测试准确率')
        plt.xticks(rotation=45)
        
        # 2. C参数影响
        plt.subplot(3, 4, 2)
        c_performance = df.groupby('svm_C')['test_score'].agg(['mean', 'std'])
        plt.errorbar(c_performance.index, c_performance['mean'], 
                    yerr=c_performance['std'], marker='o', capsize=5)
        plt.xscale('log')
        plt.xlabel('C值')
        plt.ylabel('测试准确率')
        plt.title('C参数对性能的影响')
        plt.grid(True, alpha=0.3)
        
        # 3. 训练时间分析
        plt.subplot(3, 4, 3)
        plt.scatter(df['training_time'], df['test_score'], alpha=0.6, c=df['test_score'], 
                   cmap='viridis')
        plt.xlabel('训练时间 (秒)')
        plt.ylabel('测试准确率')
        plt.title('训练时间 vs 性能')
        plt.colorbar(label='测试准确率')
        
        # 4. 性能分布直方图
        plt.subplot(3, 4, 4)
        plt.hist(df['test_score'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        plt.axvline(df['test_score'].mean(), color='red', linestyle='--', 
                   label=f'平均值: {df["test_score"].mean():.3f}')
        plt.axvline(df['test_score'].max(), color='green', linestyle='--', 
                   label=f'最大值: {df["test_score"].max():.3f}')
        plt.xlabel('测试准确率')
        plt.ylabel('频率')
        plt.title('性能分布')
        plt.legend()
        
        # 添加并行化效率分析
        plt.subplot(3, 4, 5)
        time_stats = df['training_time'].describe()
        labels = ['最小值', '25%', '50%', '75%', '最大值']
        values = [time_stats['min'], time_stats['25%'], time_stats['50%'], 
                 time_stats['75%'], time_stats['max']]
        plt.bar(labels, values, color='orange', alpha=0.7)
        plt.title('训练时间分布统计')
        plt.ylabel('时间(秒)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def _print_top_results(self, df, top_n=10):
        """打印最佳结果"""
        print("\n" + "=" * 100)
        print(f"TOP {top_n} 最佳参数组合:")
        print("=" * 100)
        
        top_results = df.nlargest(top_n, 'test_score')
        
        for i, (idx, result) in enumerate(top_results.iterrows()):
            print(f"\n第 {i+1} 名 - 测试准确率: {result['test_score']:.4f}")
            print("-" * 50)
            print("预处理参数:")
            print(f"  - 使用PCA: {result['prep_use_pca']}")
            if result['prep_use_pca']:
                print(f"  - PCA成分数: {result['prep_pca_components']}")
            print(f"  - 标准化方法: {result['prep_normalization']}")
            
            print("SVM参数:")
            print(f"  - 核函数: {result['svm_kernel']}")
            print(f"  - C值: {result['svm_C']}")
            if 'svm_gamma' in result and pd.notna(result['svm_gamma']):
                print(f"  - Gamma: {result['svm_gamma']}")
            if 'svm_degree' in result and pd.notna(result['svm_degree']):
                print(f"  - 多项式度数: {result['svm_degree']}")
            if 'svm_coef0' in result and pd.notna(result['svm_coef0']):
                print(f"  - Coef0: {result['svm_coef0']}")
            
            print(f"性能指标:")
            print(f"  - 交叉验证: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
            print(f"  - 测试准确率: {result['test_score']:.4f}")
            print(f"  - 训练时间: {result['training_time']:.2f}秒")
        
        print("\n" + "=" * 100)
    
    def get_best_model_config(self):
        """获取最佳模型配置"""
        if not self.best_params:
            print("尚未进行优化，请先运行optimize_hyperparameters")
            return None
        
        config = {
            'preprocessing': self.best_params['preprocessing'],
            'svm_params': self.best_params['svm_params'],
            'best_score': self.best_score
        }
        
        return config

def main():
    """主函数"""
    print("=" * 80)
    print("并行SVM超参数优化系统")
    print("=" * 80)
    
    try:
        # 1. 加载数据
        X_train, y_train, X_test, y_test, class_names = load_cifar_data_optimized()
        
        # 2. 创建优化器
        optimizer = OptimizedSVMHyperparameterOptimizer(
            train_sample_size=1500,      # 训练样本数
            test_sample_size=400,        # 测试样本数
            cv_folds=3,                  # 交叉验证折数
            n_workers=6,                 # 工作进程数
            use_multiprocessing=True     # 使用多进程
        )
        
        # 3. 运行优化（选择一种方式）
        print("选择优化方式:")
        print("1. 全并行优化（推荐，速度最快）")
        print("2. 批量并行优化（内存友好）")
        
        choice = input("请输入选择 (1 或 2): ").strip()
        
        if choice == "2":
            results = optimizer.optimize_hyperparameters_batch(
                X_train, y_train, X_test, y_test, batch_size=100
            )
        else:
            results = optimizer.optimize_hyperparameters(
                X_train, y_train, X_test, y_test
            )
        
        # 4. 分析结果
        if len(results) > 0:
            df_results = optimizer.analyze_results()
            
            # 5. 获取最佳配置
            best_config = optimizer.get_best_model_config()
            
            if best_config is not None:
                print("\n" + "=" * 80)
                print("最终最佳配置:")
                print("=" * 80)
                print(f"最佳测试准确率: {best_config['best_score']:.4f}")
                print(f"预处理配置: {best_config['preprocessing']}")
                print(f"SVM参数配置: {best_config['svm_params']}")
                
                # 计算效率提升
                total_time = sum(result['training_time'] for result in results)
                if optimizer.use_multiprocessing:
                    estimated_sequential_time = total_time
                    actual_parallel_time = total_time / optimizer.n_workers
                    speedup = estimated_sequential_time / actual_parallel_time
                    print(f"\n并行化效率分析:")
                    print(f"理论加速比: {speedup:.2f}x")
                    print(f"使用进程数: {optimizer.n_workers}")
        
        print("\n" + "=" * 80)
        print("并行优化完成！")
        print("=" * 80)
        
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()