# 导入所需的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from collections import Counter
import seaborn as sns
import os
import pickle
import urllib.request
import tarfile
import time
import pandas as pd
from tqdm import tqdm

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

# 使用PCA进行降维（保留更多特征）
pca = PCA(n_components=200, random_state=42)
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
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
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
    plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()


visualize_sample_images(X_train, y_train, CIFAR10_LABELS)

# 使用更小的子集进行参数搜索（为了节省时间）
# 在实际应用中可以使用全量数据
subset_size = 5000  # 用于参数搜索的样本数量
indices = np.random.choice(len(X_train_pca), subset_size, replace=False)
X_search = X_train_pca[indices]
y_search = y_train[indices]

# 定义参数网格
param_grid = {
    'max_depth': [10, 15, 20, 25, 30, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8],
    'criterion': ['gini', 'entropy'],
    'max_features': ['sqrt', 'log2', None]
}

# 创建决策树分类器
base_clf = DecisionTreeClassifier(random_state=42)

# 使用网格搜索寻找最优参数
print("开始参数搜索...")
start_time = time.time()

# 使用更少的交叉验证折数以加速搜索
grid_search = GridSearchCV(estimator=base_clf, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_search, y_search)

end_time = time.time()
print(f"参数搜索完成，耗时: {end_time - start_time:.2f}秒")

# 输出最佳参数
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("\n最佳参数组合:", best_params)
print(f"交叉验证最佳准确率: {best_score:.4f}")

# 使用最佳参数创建模型
best_clf = DecisionTreeClassifier(**best_params, random_state=42)

# 在完整训练集上训练最佳模型
print("\n在完整训练集上训练最佳模型...")
best_clf.fit(X_train_pca, y_train)

# 进行预测
y_pred = best_clf.predict(X_test_pca)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"\n测试集准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=CIFAR10_LABELS))


# 可视化混淆矩阵
def visualize_confusion_matrix(y_test, y_pred):
    """可视化混淆矩阵"""
    print("正在生成混淆矩阵可视化...")

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 12},
                xticklabels=CIFAR10_LABELS, yticklabels=CIFAR10_LABELS)
    plt.title(f'混淆矩阵 (准确率: {accuracy:.4f})', fontsize=14, fontweight='bold')
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
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

    # 添加准确率数值标签
    for i, acc in enumerate(class_accuracy):
        plt.text(i, acc + 0.02, f'{acc:.3f}', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()

    return class_accuracy  # 返回计算得到的 class_accuracy


# 调用 visualize_class_accuracy 函数并获取返回值
class_accuracy = visualize_class_accuracy(y_test, y_pred)


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
    plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


visualize_prediction_distribution(y_test, y_pred)


# 可视化学习曲线
def plot_learning_curve(estimator, title, X, y, cv=5, train_sizes=None, n_jobs=-1):
    """
    生成学习曲线可视化

    参数:
        estimator : 模型实例
        title : 图表标题
        X : 特征数据
        y : 目标数据
        cv : 交叉验证折数
        train_sizes : 训练集大小数组
        n_jobs : 并行任务数
    """
    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)

    plt.figure(figsize=(12, 8))

    # 计算学习曲线
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy',
        random_state=42, shuffle=True)

    # 计算平均值和标准差
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="训练集准确率")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="交叉验证准确率")

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("训练样本数", fontsize=14)
    plt.ylabel("准确率", fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加准确率数值标签
    for i, size in enumerate(train_sizes):
        plt.text(size, train_scores_mean[i] + 0.01, f"{train_scores_mean[i]:.4f}",
                 fontsize=10, ha='center', color='red')
        plt.text(size, test_scores_mean[i] - 0.02, f"{test_scores_mean[i]:.4f}",
                 fontsize=10, ha='center', color='green')

    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()

    return train_sizes, train_scores_mean, test_scores_mean


# 绘制学习曲线
print("正在生成学习曲线...")
train_sizes, train_scores, val_scores = plot_learning_curve(
    best_clf,
    f"决策树学习曲线 (最佳参数: {str(best_params)[:60]}...)",
    X_train_pca,
    y_train,
    cv=3,  # 减少折数以加快计算速度
    train_sizes=np.linspace(0.1, 1.0, 5)  # 使用5个训练集大小点
)

# 保存模型性能结果
results = {
    'best_params': best_params,
    'best_cv_score': best_score,
    'test_accuracy': accuracy,
    'class_accuracy': dict(zip(CIFAR10_LABELS, class_accuracy)),
    'learning_curve': {
        'train_sizes': train_sizes,
        'train_scores': train_scores,
        'val_scores': val_scores
    }
}

with open('model_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n模型训练和评估完成！所有结果已保存。")