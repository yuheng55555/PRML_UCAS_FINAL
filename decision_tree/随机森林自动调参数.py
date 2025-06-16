# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from collections import Counter
import seaborn as sns
import os
import pickle
import urllib.request
import tarfile

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 手动下载和加载 CIFAR-10 数据集
def load_cifar10():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    dir_name = "cifar-10-batches-py"

    # 下载数据集
    if not os.path.exists(filename):
        print("开始下载 CIFAR-10 数据集...")
        urllib.request.urlretrieve(url, filename)
        print("数据集下载完成。")

    # 解压数据集
    if not os.path.exists(dir_name):
        print("开始解压 CIFAR-10 数据集...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("数据集解压完成。")

    # 加载训练数据
    X_train = []
    y_train = []
    for i in range(1, 6):
        file_path = os.path.join(dir_name, f"data_batch_{i}")
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            X_train.append(batch['data'])
            y_train.extend(batch['labels'])

    # 加载测试数据
    file_path = os.path.join(dir_name, "test_batch")
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        X_test = batch['data']
        y_test = batch['labels']

    # 转换为 numpy 数组
    X_train = np.concatenate(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test

# 加载数据集
X_train, X_test, y_train, y_test = load_cifar10()

# 将图像数据归一化并转换为一维向量
X_train_flattened = X_train.reshape(X_train.shape[0], -1) / 255.0
X_test_flattened = X_test.reshape(X_test.shape[0], -1) / 255.0

# 使用PCA进行降维
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train_flattened)
X_test_pca = pca.transform(X_test_flattened)

# 可视化数据分布
CIFAR10_LABELS = ['飞机', '汽车', '鸟', '猫', '鹿', '狗', '青蛙', '马', '船', '卡车']

def visualize_data_distribution(y_train, y_test):
    """可视化数据分布"""
    print("正在生成数据分布可视化...")

    plt.figure(figsize=(18, 6))

    # 训练数据分布
    plt.subplot(1, 3, 1)
    train_counts = Counter(y_train)
    plt.bar([CIFAR10_LABELS[i] for i in range(10)],
            [train_counts[i] for i in range(10)],
            color='skyblue', alpha=0.7)
    plt.title('训练数据类别分布', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 测试数据分布
    plt.subplot(1, 3, 2)
    test_counts = Counter(y_test)
    plt.bar([CIFAR10_LABELS[i] for i in range(10)],
            [test_counts[i] for i in range(10)],
            color='lightcoral', alpha=0.7)
    plt.title('测试数据类别分布', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 总体分布比较
    plt.subplot(1, 3, 3)
    x = np.arange(10)
    width = 0.35
    plt.bar(x - width / 2, [train_counts[i] for i in range(10)],
            width, label='训练集', alpha=0.7, color='darkblue')
    plt.bar(x + width / 2, [test_counts[i] for i in range(10)],
            width, label='测试集', alpha=0.7, color='darkred')
    plt.title('训练集与测试集分布对比', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(x, [CIFAR10_LABELS[i] for i in range(10)], rotation=45, fontsize=10)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

visualize_data_distribution(y_train, y_test)

# 可视化样本图像
def visualize_sample_images(X, y, CIFAR10_LABELS, num_samples=20):
    """可视化样本图像"""
    print("正在生成样本图像可视化...")

    # 重塑图像数据
    images = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    plt.figure(figsize=(18, 10))
    for i in range(num_samples):
        plt.subplot(4, 5, i + 1)
        plt.imshow(images[i])
        plt.title(f'{CIFAR10_LABELS[y[i]]}', fontsize=12, fontweight='bold')
        plt.axis('off')
    plt.suptitle('CIFAR-10 样本图像', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

visualize_sample_images(X_train, y_train, CIFAR10_LABELS)

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],  # 树的数量
    'max_depth': [10, 15, 20],  # 树的最大深度
    'min_samples_split': [2, 5, 10],  # 节点分裂所需的最小样本数
    'min_samples_leaf': [1, 2, 4],  # 叶子节点所需的最小样本数
    'max_features': ['sqrt', 'log2', None]  # 每个分裂考虑的最大特征数
}

# 创建随机森林分类器
rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1)

# 使用网格搜索寻找最佳参数
grid_search = GridSearchCV(estimator=rf_clf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# 拟合训练数据
grid_search.fit(X_train_pca, y_train)

# 输出最佳参数和对应准确率
print(f"最佳参数组合: {grid_search.best_params_}")
print(f"最佳准确率: {grid_search.best_score_}")

# 使用最佳参数重新训练模型
best_rf_clf = grid_search.best_estimator_

# 在测试集上进行预测
y_pred = best_rf_clf.predict(X_test_pca)

# 打印分类报告
print("\n优化后的分类报告:")
print(classification_report(y_test, y_pred, target_names=CIFAR10_LABELS))

# 可视化混淆矩阵
def visualize_confusion_matrix(y_test, y_pred):
    """可视化混淆矩阵"""
    print("正在生成混淆矩阵可视化...")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12},
                xticklabels=CIFAR10_LABELS, yticklabels=CIFAR10_LABELS)
    plt.title('混淆矩阵', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

visualize_confusion_matrix(y_test, y_pred)

# 可视化各类别准确率
def visualize_class_accuracy(y_test, y_pred):
    """可视化各类别准确率"""
    print("正在生成各类别准确率可视化...")

    class_accuracy = []
    for i in range(10):
        mask = y_test == i
        if np.sum(mask) > 0:
            acc = np.sum(y_pred[mask] == i) / np.sum(mask)
            class_accuracy.append(acc)
        else:
            class_accuracy.append(0)

    plt.figure(figsize=(12, 6))
    plt.bar(CIFAR10_LABELS, class_accuracy, color='lightgreen', alpha=0.7)
    plt.title('各类别准确率', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

visualize_class_accuracy(y_test, y_pred)

# 可视化预测分布与真实分布对比
def visualize_prediction_distribution(y_test, y_pred):
    """可视化预测分布与真实分布对比"""
    print("正在生成预测分布与真实分布对比可视化...")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(y_test, bins=np.arange(11) - 0.5, edgecolor='black', alpha=0.7, color='darkblue')
    plt.title('真实分布', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(np.arange(10), CIFAR10_LABELS, rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.hist(y_pred, bins=np.arange(11) - 0.5, edgecolor='black', alpha=0.7, color='darkred')
    plt.title('预测分布', fontsize=14, fontweight='bold')
    plt.xlabel('类别', fontsize=12)
    plt.ylabel('样本数量', fontsize=12)
    plt.xticks(np.arange(10), CIFAR10_LABELS, rotation=45, fontsize=10)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

visualize_prediction_distribution(y_test, y_pred)

# 计算学习曲线
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure(figsize=(12, 6))
    plt.title(title, fontsize=14, fontweight='bold')
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel('训练样本数', fontsize=12)
    plt.ylabel('准确率', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy')
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='训练集准确率', linewidth=2, markersize=8)
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='交叉验证集准确率', linewidth=2, markersize=8)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.show()

plot_learning_curve(best_rf_clf, '优化后的随机森林分类器学习曲线',
                    X_train_pca, y_train, cv=5, n_jobs=-1)