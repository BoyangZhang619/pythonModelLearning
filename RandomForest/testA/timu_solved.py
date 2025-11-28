# 完整代码：基于随机森林的房屋快速售出预测
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, roc_curve, auc, precision_recall_curve
)
from sklearn.inspection import permutation_importance

# 全局绘图设置
plt.rcParams['font.sans-serif'] = ['SimSun']
plt.rcParams['axes.unicode_minus'] = False
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv")
IMG_DIR = os.path.join(BASE_DIR, "img")

# ==============================
# 一、数据预处理
# ==============================
# 1. 数据加载与筛选
data = pd.read_csv(os.path.join(CSV_DIR,"generated_housing_data.csv"))[
    ["days_on_market", "price", "area", "bedrooms",
     "bathrooms", "year_built", "renovated",
     "distance_to_metro", "community_score"]
]

# 2. 缺失值处理
data = data.dropna()
print(f"处理后样本量：{data.shape[0]}，特征数：{data.shape[1]-1}")

# 3. 目标变量与自变量定义
sold_fast = (data["days_on_market"] <= 30).astype(int)
data = data.drop("days_on_market", axis=1)
features_name = data.columns.tolist()
X = data.to_numpy()
y = sold_fast.to_numpy()

# 4. 训练集与测试集划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=123, stratify=y
)
print(f"训练集：{X_train.shape[0]}样本，测试集：{X_test.shape[0]}样本")

# ==============================
# 二、模型建立与超参数选择（基于OOB）
# ==============================
# 1. 实验函数：遍历树数量计算OOB误差
def experiment(trees_list):
    results = []
    print("树数量与OOB误差对应关系：")
    for t in trees_list:
        rf = RandomForestClassifier(
            n_estimators=t, max_depth=12, random_state=123,
            oob_score=True, bootstrap=True, class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        oob_err = 1 - rf.oob_score_
        results.append((t, oob_err, rf))
        if (t - trees_list[0]) % 10 == 0:
            print(f"trees={t:3d}  OOB error={oob_err:.4f}", end=" | ")
    return results

# 2. 运行实验
trees_list = list(range(20, 101))
results = experiment(trees_list)

# 3. OOB误差曲线可视化
plt.figure(figsize=(10, 6))
trees = [r[0] for r in results]
oob_errs = [r[1] for r in results]
plt.plot(trees, oob_errs, marker='o', markersize=4, color='#1f77b4')
best_idx = np.argmin(oob_errs)
plt.scatter(trees[best_idx], oob_errs[best_idx], color='red', s=100, zorder=5)
plt.annotate(f'最优树数量：{trees[best_idx]}\nOOB误差：{oob_errs[best_idx]:.4f}',
             xy=(trees[best_idx], oob_errs[best_idx]),
             xytext=(trees[best_idx]+5, oob_errs[best_idx]+0.01),
             arrowprops=dict(arrowstyle='->', color='red'))
plt.xlabel("树数量（Number of Trees）", fontsize=12)
plt.ylabel("OOB误差（OOB Error）", fontsize=12)
plt.title("随机森林树数量与OOB误差关系", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR,"oob_error_vs_trees.png"), dpi=300, bbox_inches='tight')
plt.show()

# 4. 提取最佳/最差模型
best_idx = np.argmin([r[1] for r in results])
best_trees, best_oob, best_rf = results[best_idx]
worst_idx = np.argmax([r[1] for r in results])
worst_trees, worst_oob, worst_rf = results[worst_idx]
print(f"\n最佳模型：树数量={best_trees}，OOB误差={best_oob:.4f}")
print(f"最差模型：树数量={worst_trees}，OOB误差={worst_oob:.4f}")

# ==============================
# 三、最佳OOB模型评估
# ==============================
# 1. 预测结果计算
y_best_train_prob = best_rf.predict_proba(X_train)[:, 1]
y_best_test_prob = best_rf.predict_proba(X_test)[:, 1]
y_best_pred = (y_best_test_prob >= 0.5).astype(int)

# 2. 定量指标输出
print("\n===== 最佳OOB模型测试集评估 =====")
print(f"训练集 log loss：{log_loss(y_train, y_best_train_prob):.4f}")
print(f"测试集 log loss：{log_loss(y_test, y_best_test_prob):.4f}")
print(f"准确率：{accuracy_score(y_test, y_best_pred):.4f}")
print(f"精确率：{precision_score(y_test, y_best_pred, zero_division=0):.4f}")
print(f"召回率：{recall_score(y_test, y_best_pred):.4f}")
print(f"F1 分数：{f1_score(y_test, y_best_pred):.4f}")

# 3. 特征重要性
best_perm_import = permutation_importance(
    best_rf, X_train, y_train, n_repeats=10, random_state=123, n_jobs=-1
)
best_importances = best_perm_import.importances_mean
sorted_idx = np.argsort(best_importances)

plt.figure(figsize=(10, 6))
plt.barh(np.array(features_name)[sorted_idx], best_importances[sorted_idx], color='#2ca02c')
plt.xlabel("置换特征重要性", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("最佳OOB模型：特征重要性排序", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.savefig(os.path.join(IMG_DIR,"best_model_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.show()

# 4. ROC曲线
best_fpr, best_tpr, _ = roc_curve(y_test, y_best_test_prob)
best_roc_auc = auc(best_fpr, best_tpr)

plt.figure(figsize=(8, 6))
plt.plot(best_fpr, best_tpr, color='#1f77b4', lw=2, label=f'AUC = {best_roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机分类')
plt.xlabel("假正率", fontsize=12)
plt.ylabel("真正率（召回率）", fontsize=12)
plt.title("最佳OOB模型：ROC曲线", fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR,"best_model_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# 5. PR曲线
best_precision, best_recall, _ = precision_recall_curve(y_test, y_best_test_prob)

plt.figure(figsize=(8, 6))
plt.plot(best_recall, best_precision, color='#ff7f0e', lw=2)
plt.xlabel("召回率", fontsize=12)
plt.ylabel("精确率", fontsize=12)
plt.title("最佳OOB模型：PR曲线", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR,"best_model_pr_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# ==============================
# 四、最差OOB模型评估（对比）
# ==============================
# 1. 预测结果计算
y_worst_train_prob = worst_rf.predict_proba(X_train)[:, 1]
y_worst_test_prob = worst_rf.predict_proba(X_test)[:, 1]
y_worst_pred = (y_worst_test_prob >= 0.5).astype(int)

# 2. 定量指标输出
print("\n===== 最差OOB模型测试集评估 =====")
print(f"训练集 log loss：{log_loss(y_train, y_worst_train_prob):.4f}")
print(f"测试集 log loss：{log_loss(y_test, y_worst_test_prob):.4f}")
print(f"准确率：{accuracy_score(y_test, y_worst_pred):.4f}")
print(f"精确率：{precision_score(y_test, y_worst_pred, zero_division=0):.4f}")
print(f"召回率：{recall_score(y_test, y_worst_pred):.4f}")
print(f"F1 分数：{f1_score(y_test, y_worst_pred):.4f}")

# 3. 特征重要性
worst_perm_import = permutation_importance(
    worst_rf, X_train, y_train, n_repeats=10, random_state=123, n_jobs=-1
)
worst_importances = worst_perm_import.importances_mean
sorted_idx = np.argsort(worst_importances)

plt.figure(figsize=(10, 6))
plt.barh(np.array(features_name)[sorted_idx], worst_importances[sorted_idx], color='#d62728')
plt.xlabel("置换特征重要性", fontsize=12)
plt.ylabel("特征名称", fontsize=12)
plt.title("最差OOB模型：特征重要性排序", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.savefig(os.path.join(IMG_DIR,"worst_model_feature_importance.png"), dpi=300, bbox_inches='tight')
plt.show()

# 4. ROC曲线
worst_fpr, worst_tpr, _ = roc_curve(y_test, y_worst_test_prob)
worst_roc_auc = auc(worst_fpr, worst_tpr)

plt.figure(figsize=(8, 6))
plt.plot(worst_fpr, worst_tpr, color='#d62728', lw=2, label=f'AUC = {worst_roc_auc:.4f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='随机分类')
plt.xlabel("假正率", fontsize=12)
plt.ylabel("真正率（召回率）", fontsize=12)
plt.title("最差OOB模型：ROC曲线", fontsize=14, fontweight='bold')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR,"worst_model_roc_curve.png"), dpi=300, bbox_inches='tight')
plt.show()

# 5. PR曲线
worst_precision, worst_recall, _ = precision_recall_curve(y_test, y_worst_test_prob)

plt.figure(figsize=(8, 6))
plt.plot(worst_recall, worst_precision, color='#9467bd', lw=2)
plt.xlabel("召回率", fontsize=12)
plt.ylabel("精确率", fontsize=12)
plt.title("最差OOB模型：PR曲线", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(IMG_DIR,"worst_model_pr_curve.png"), dpi=300, bbox_inches='tight')
plt.show()