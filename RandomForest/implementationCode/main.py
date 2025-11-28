# Retry with lighter settings to complete within runtime
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from copy import deepcopy
import random
import math
import pandas as pd

np.random.seed(0)
random.seed(0)

def make_synthetic(n_samples=400, n_informative=3, n_noise=7, flip_y=0.05):
    """
    生成具有标准正态分布的特征数据,形状为(n_samples, n_informative + n_noise)。
    :param n_samples: 数据量
    :param n_informative: 有效值
    :param n_noise: 噪音值
    :param flip_y: 标签翻转率
    :return: X, y, weights
    """
    # 生成具有标准正态分布的特征数据,形状为(n_samples, n_informative)
    X_inf = np.random.randn(n_samples, n_informative)
    # 创建线性递减的权重向量,从1.0递减到0.5
    weights = np.linspace(1.0, 0.5, n_informative)
    print(weights)
    # 计算特征和权重的点积,得到线性预测值
    logits = X_inf.dot(weights)
    # 使用sigmoid函数将线性预测值转换为概率
    probs = 1 / (1 + np.exp(-logits))
    # 根据概率生成二分类标签(0或1)
    y = (np.random.rand(n_samples) < probs).astype(int)
    # 生成噪声特征,形状为(n_samples, n_noise)
    X_noise = np.random.randn(n_samples, n_noise)
    # 将有信息特征和噪声特征水平堆叠
    X = np.hstack([X_inf, X_noise])
    # 随机翻转部分标签以增加噪声
    flip_mask = np.random.rand(n_samples) < flip_y
    y[flip_mask] = 1 - y[flip_mask]
    # 返回特征矩阵X、标签y和真实权重
    return X, y, weights

X, y, true_weights = make_synthetic(n_samples=400, n_informative=3, n_noise=7, flip_y=0.07)
n_samples, n_features = X.shape
print(X.shape)
feature_names = [f"f{i}" for i in range(n_features)]

def gini(y):
    """
    计算一个分类样本集合 y 的基尼不纯度（Gini impurity）。
    如果所有样本属于同一类,G = 0（纯）
    如果类别均匀分布,G 最大（最不纯）
    G in [0,0.5]
    :param y: 一个类别标签的列表
    :return:
    """
    if len(y) == 0:
        return 0.0
    counts = np.bincount(y, minlength=2)
    # np.bincount([6,1,9]) => np.array([coup[1] for coup in sorted(dict(Counter([6,1,9])).items(),key=lambda x:x[0])])
    p = counts / counts.sum()
    return 1.0 - sum(p**2)

def weighted_gini(left_y, right_y):
    """
    计算一个分裂后的加权基尼不纯度,用于决策树判断最佳分裂点。
    基尼不纯度越小 → 数据集合越“纯”,决策树就越喜欢这样的分裂。
    决策树算法会选择分裂使 加权基尼不纯度最小 的特征和阈值。
    :param left_y: 分裂后左子节点的样本标签
    :param right_y: 分裂后右子节点的样本标签
    :return:
    """
    n = len(left_y) + len(right_y)
    return (len(left_y) * gini(left_y) + len(right_y) * gini(right_y)) / n

class Node:
    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

# def build_tree(X, y, feature_indices, max_depth=999, min_samples_split=2, depth=0):
#     # 类似_build_tree_with_node_subsampling中的builder函数,不知道ai为什么写了这个无引用的函数，所以便先注释了
#     num_samples = len(y)
#     num_pos = y.sum()
#     if num_samples < min_samples_split or depth >= max_depth or num_pos == 0 or num_pos == num_samples:
#         majority = int((y.sum() >= (num_samples - y.sum())))
#         return Node(value=majority)
#     best_feat, best_thr, best_score = None, None, 1.0
#     best_l_idx = best_r_idx = None
#     for feat in feature_indices:
#         vals = X[:, feat]
#         sorted_idx = np.argsort(vals)
#         sorted_vals = vals[sorted_idx]
#         sorted_y = y[sorted_idx]
#         for i in range(1, num_samples):
#             if sorted_vals[i] == sorted_vals[i-1]:
#                 continue
#             thr = 0.5 * (sorted_vals[i] + sorted_vals[i-1])
#             l_y = sorted_y[:i]
#             r_y = sorted_y[i:]
#             score = weighted_gini(l_y, r_y)
#             if score < best_score:
#                 best_score = score
#                 best_feat = feat
#                 best_thr = thr
#                 best_l_idx = sorted_idx[:i].copy()
#                 best_r_idx = sorted_idx[i:].copy()
#     if best_feat is None:
#         majority = int((y.sum() >= (num_samples - y.sum())))
#         return Node(value=majority)
#     left = build_tree(X[best_l_idx], y[best_l_idx], feature_indices, max_depth, min_samples_split, depth+1)
#     right = build_tree(X[best_r_idx], y[best_r_idx], feature_indices, max_depth, min_samples_split, depth+1)
#     return Node(feature=best_feat, threshold=best_thr, left=left, right=right)

def predict_tree(node, x):
    """
    predict tree
    :param node: 单个决策树的根节点
    :param x: 单个样本,在外部已被拆分
    :return: predicted value
    """
    while node.value is None:
        # 仅有叶子节点有value,故通过该值以判断是否为内部节点以继续
        if x[node.feature] <= node.threshold:
            # 比对当前单个样本在当前节点的最优分割特征列(下标)的值与当前节点的分割阈值以判断进入左子树还是右子树
            node = node.left
        else:
            node = node.right
    # 到达叶子节点返回预测值
    return node.value

class RandomForestScratch:
    def __init__(self, n_trees=50, max_depth=12, min_samples_split=2, m_try=None, bootstrap=True):
        """
        n_trees 和 m_try 共同控制模型的随机性
        max_depth 和 min_samples_split 共同控制单棵树的复杂度
        bootstrap 影响数据采样方式，与其他参数协同工作
        对于m_try:普通决策树：在所有特征中找最佳分裂,随机森林的树：在随机子集中找最佳分裂
        :param n_trees: 决定随机森林中决策树的数量
        :param max_depth: 限制每棵决策树的最大深度,过大过小会导致过拟合与欠拟合问题
        :param min_samples_split: 最小样本分裂数,过大过小会导致欠拟合与过拟合
        :param m_try: 每次分裂时考虑的特征数量,默认int(math.sqrt(总特征数))
        :param bootstrap: 是否采用有放回采样,True时约采用63.2%数据,False为全部
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.m_try = m_try
        self.bootstrap = bootstrap
        self.trees = []
        self.oob_indices = []

    def fit(self, X, y):
        n, d = X.shape
        # n => 样本数, d => 特征数
        if self.m_try is None:
            self.m_try = int(math.sqrt(d))
            #处理默认
        self.trees = []
        # 存储所有决策树
        self.oob_indices = []
        # 存储每棵树的袋外样本索引
        for t in range(self.n_trees):
            if self.bootstrap:
                indices = np.random.choice(n, size=n, replace=True)
                # print(indices.shape) #(400,)
                oob = np.setdiff1d(np.arange(n), np.unique(indices))
                # 序列0-n有放回取n个,并将0-n序列与所取序列去重进行left join以放入oob
            else:
                indices = np.arange(n)
                oob = np.array([], dtype=int)
            self.oob_indices.append(oob)
            print(f"{indices=}")
            X_sample = X[indices] #np特色,不使用for直接取
            y_sample = y[indices] #对应的标签与结果
            tree = self._build_tree_with_node_subsampling(X_sample, y_sample)
            self.trees.append(tree)

    def _build_tree_with_node_subsampling(self, X_sub, y_sub):
        n, d = X_sub.shape
        # n: 样本数, d: 特征数
        def builder(X_node, y_node, depth):
            num_samples = len(y_node) # 当前节点的样本数
            num_pos = y_node.sum() # 正样本数量,因为y_node的值只有0或1
            if num_samples < self.min_samples_split or depth >= self.max_depth or num_pos == 0 or num_pos == num_samples:
                # 终止条件:样本数小于最小分裂样本数|达到最大深度|所有样本属于同一类
                majority = int((y_node.sum() >= (num_samples - y_node.sum())))
                # 返回多数的那一类,int((1的数量)>=(总数-1的数量))
                return Node(value=majority)
            feat_candidates = np.random.choice(d, size=self.m_try, replace=False)
            # 不放回获取self.m_try个features
            best_feat, best_thr, best_score = None, None, float('inf')
            # 最佳基尼不纯度取决于n分类问题的n值，即self.min_sample_split,可以直接写1
            best_l_idx = best_r_idx = None
            for feat in feat_candidates:
                vals = X_node[:, feat]
                sorted_idx = np.argsort(vals) #返回可以让数组排好序的索引序列
                sorted_vals = vals[sorted_idx] #获取已排序序列对应的样本
                sorted_y = y_node[sorted_idx] #获取已排序序列对应的目标或标签
                for i in range(1, num_samples):
                    #逐两个元素间进行对应分裂以计算最小平均基尼系数
                    if sorted_vals[i] == sorted_vals[i-1]:
                        # 若相邻两值相同则不必分割,至少在当前特征下其将一直保存至叶子节点
                        continue
                    # 计算分割阈值,类似分割下标
                    thr = 0.5 * (sorted_vals[i] + sorted_vals[i-1])
                    l_y = sorted_y[:i]
                    r_y = sorted_y[i:]
                    # 取对应左右子集,随后进行基尼纯度计算
                    score = weighted_gini(l_y, r_y)
                    if score < best_score:
                        # 通过基尼纯度得到有更好的分割点并更新最优值
                        best_score = score
                        best_feat = feat
                        best_thr = thr
                        best_l_idx = sorted_idx[:i].copy()
                        best_r_idx = sorted_idx[i:].copy()
            if best_feat is None:
                # 当每个特征都无有效可用分割阈值,返回...
                majority = int((y_node.sum() >= (num_samples - y_node.sum())))
                return Node(value=majority)
            # 递归构建左右子树
            left = builder(X_node[best_l_idx], y_node[best_l_idx], depth+1)
            right = builder(X_node[best_r_idx], y_node[best_r_idx], depth+1)
            # 返回当前节点(左右子节点都已求完了的节点)
            return Node(feature=best_feat, threshold=best_thr, left=left, right=right)
        # 返回一个包含最优特征与最优切割阈值的各个节点(除了叶子节点仅有value)的二叉树
        return builder(X_sub, y_sub, 0)

    def predict(self, X):
        preds = np.array([[predict_tree(tree, x) for tree in self.trees] for x in X])
        # 对每个单个样本都进行所有的已有的决策树的预测,返回的preds的shape值应该是(len(X),len(self.trees))
        maj = np.array([Counter(row).most_common(1)[0][0] for row in preds])
        # 对于每个单个样本的多个预测值取众数,maj为一维数组
        return maj
    def _predict_oob(self,X,tree_indices=None):
        """
        注释见oob_score部分,这个函数是为了减少代码量新添的
        :param X:
        :param tree_indices: 要算的数,默认全部
        :return:
        """
        n = X.shape[0]
        votes = [defaultdict(int) for _ in range(n)]
        counts = np.zeros(n,dtype=int)
        trees = self.trees if tree_indices is None else [self.trees[i] for i in tree_indices]
        oob_indices = self.oob_indices if tree_indices is None else [self.oob_indices[i] for i in tree_indices]
        for tree,oob in zip(trees,oob_indices):
            for i in oob:
                pred = predict_tree(tree,X[i])
                votes[i][pred] += 1
                counts[i] += 1
        preds = np.full(n,-1,dtype=int)
        for i in range(n):
            if counts[i] > 0:
                preds[i] = max(votes[i].items(),key=lambda x:x[1])[0]
        return preds,counts > 0
    def oob_score(self, X, y):
        n = X.shape[0]
        # votes = [defaultdict(int) for _ in range(n)]
        # # 给每个参数提供一个存储各个目标值可能被预测次数的dict
        # counts = np.zeros(n, dtype=int)
        # # 存储每个参数被预测的次数
        # for tree, oob in zip(self.trees, self.oob_indices):
        #     # 每一个树都对其对应的每一项oob样本进行预测
        #     for i in oob:
        #         pred = predict_tree(tree, X[i])
        #         votes[i][pred] += 1
        #         counts[i] += 1
        #         # 获取预测值并存储和记录
        # oob_preds = np.full(n, -1, dtype=int)
        # valid = counts > 0 #返回一个bool数组以进行后续操作
        # for i in range(n):
        #     if counts[i] > 0:
        #         oob_preds[i] = max(votes[i].items(), key=lambda kv: kv[1])[0]
        #        # 若该值有被预测过,则计算其被预测结果的众数并放入oob_preds数组,该数组即为最终预测结果
        oob_preds,valid=self._predict_oob(X)
        oob_err = np.mean(oob_preds[valid] != y[valid])
        # 计算预测值与实际值不对应的占比,即错误率
        return oob_err, oob_preds, valid

    def permutation_importance(self, X, y, valid_mask, baseline_oob_preds=None, n_repeats=10, baseline_err=None):
        """

        :param X:
        :param y:
        :param baseline_oob_preds: 预测的基准预测结果一维数组
        :param valid_mask: bool数组,形容某样本是否被预测过
        :param n_repeats: 重复打乱的次数,默认 10
        :return:
        """
        n, d = X.shape
        if baseline_err is None and baseline_oob_preds is not None:
            baseline_err = np.mean(baseline_oob_preds[valid_mask] != y[valid_mask])
        importances = np.zeros(d)
        for feat in range(d):
            # 循环特征个数次
            errs = [] # 存放各个错误率
            for r in range(n_repeats):
                X_perm = X.copy()
                perm = X_perm[:, feat].copy()
                np.random.shuffle(perm)
                X_perm[:, feat] = perm
                # 对于第d个特征列,有使其重复n_repeats次打乱行为
                # votes = [defaultdict(int) for _ in range(n)]
                # counts = np.zeros(n, dtype=int)
                # for tree, oob in zip(self.trees, self.oob_indices):
                #     for i in oob:
                #         pred = predict_tree(tree, X_perm[i])
                #         votes[i][pred] += 1
                #         counts[i] += 1
                # perm_oob_preds = np.full(n, -1, dtype=int)
                # for i in range(n):
                #     if counts[i] > 0:
                #         perm_oob_preds[i] = max(votes[i].items(), key=lambda kv: kv[1])[0]
                perm_oob_preds,_ = self._predict_oob(X_perm)
                perm_err = np.mean(perm_oob_preds[valid_mask] != y[valid_mask])
                # 计算
                errs.append(perm_err)
            # 算出错误率与原未打乱样本错误率之差,正相关
            importances[feat] = np.mean(errs) - baseline_err
        # 返回一个长度为n_features的数组，表示每个特征的重要性分数
        # 负值（罕见）可能表明模型在打乱该特征后性能反而提高了，这可能表明存在数据泄露或过拟合
        return importances

def experiment(n_trees_list, X, y, max_depth=12, min_samples_split=2, m_try=None):
    results = []
    for T in n_trees_list:
        # 计算不同允许trees对应的结果
        rf = RandomForestScratch(n_trees=T, max_depth=max_depth, min_samples_split=min_samples_split, m_try=m_try, bootstrap=True)
        rf.fit(X, y)
        oob_err, oob_preds, valid = rf.oob_score(X, y)
        results.append((T, oob_err, rf, oob_preds, valid))
        print(f"Trees={T:3d}  OOB error={oob_err:.4f}  valid_oob_samples={valid.sum()}")
    return results

n_trees_list = [i for i in range(3,11,3)]
results = experiment(n_trees_list, X, y, max_depth=12, min_samples_split=2, m_try=int(math.sqrt(n_features)))

# 下面都是画图的
Ts = [r[0] for r in results]
oob_errors = [r[1] for r in results]
plt.figure()
plt.plot(Ts, oob_errors, marker='o')
plt.xlabel("Number of trees")
plt.ylabel("OOB error")
plt.title("OOB error vs number of trees")
plt.grid(True)
plt.show()

# 获取OOB错误率最低的一个实例
baseline_oob_err, best_rf, _, valid_mask = results[np.argmin([r[1] for r in results])][1:]
importances = best_rf.permutation_importance(X, y, valid_mask, n_repeats=10,baseline_err=baseline_oob_err)

imp_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False).reset_index(drop=True)

# 画图
plt.figure(figsize=(8,4))
plt.bar(imp_df['feature'], imp_df['importance'])
plt.xlabel("Feature")
plt.ylabel("Importance (increase in OOB error)")
plt.title("Permutation-based Feature Importances (higher = more important)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\nSummary:")
print(f"True informative features were f0..f{len(true_weights)-1} (descending true influence).")
print("Top importances from permutation should align with these.")
print(f"OOB error for final forest (T={results[-1][0]}): {baseline_oob_err:.4f}")
