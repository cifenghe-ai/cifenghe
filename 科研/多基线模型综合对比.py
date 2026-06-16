"""
多基线模型综合对比
====================
在现有 W1/W2/W3 集成模型基础上，新增 2 个机器学习基线模型：
  1.   MLP 神经网络 (MLP Regressor)
  2.   支持向量回归 (SVR)

共计 5 个模型的 R² / RMSE 全对比。
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

epsilon = 1e-12

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.unicode_minus'] = False

desktop = os.path.join(os.path.expanduser("~"), "Desktop")


# =====================================================
# 数据加载
# =====================================================
def load_txt_with_brackets(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().rstrip(',')
            if not line:
                continue
            line = line.replace('[', '').replace(']', '')
            parts = [float(x) for x in line.split(',')]
            data.append(parts)
    return np.array(data)


# =====================================================
# W1 —— 幂函数模型 (L, W → Weight)
# =====================================================
def train_W1(L, W, Y):
    def model(X, k, b, d):
        L_, W_ = X
        return k * (np.maximum(L_, epsilon) ** b) * (np.maximum(W_, epsilon) ** d)

    X = np.array([L, W])
    popt, _ = curve_fit(
        model, X, Y,
        bounds=([0, -10, -10], [np.inf, 10, 10]),
        maxfev=20000
    )
    Yp = model((L, W), *popt)
    return Yp, popt


# =====================================================
# W2 —— FROE（ERR 正交前向回归）
# =====================================================
def train_W2_FROE(L, W, Y, max_degree=12, err_threshold=0.99):
    sL, sW = StandardScaler(), StandardScaler()
    Ls = sL.fit_transform(L.reshape(-1, 1)).ravel()
    Ws = sW.fit_transform(W.reshape(-1, 1)).ravel()

    y = Y.copy()
    terms = [np.ones(len(y)), Ls, Ws]
    names = ['1', 'L', 'W']

    for d in range(2, max_degree + 1):
        for i in range(d + 1):
            j = d - i
            if i == 0:
                terms.append(Ws ** j)
                names.append(f'W^{j}')
            elif j == 0:
                terms.append(Ls ** i)
                names.append(f'L^{i}')
            else:
                terms.append((Ls ** i) * (Ws ** j))
                names.append(f'L^{i}W^{j}')

    P = np.column_stack(terms)

    selected, remaining = [], list(range(P.shape[1]))
    W_ortho, A, g = [], [], []

    Ey = np.dot(y, y)
    cum_err = 0.0

    while remaining:
        best = None
        best_err = -np.inf

        for idx in remaining:
            p = P[:, idx].copy()
            w = p.copy()
            alpha = []

            for wi in W_ortho:
                a = np.dot(wi, p) / np.dot(wi, wi)
                alpha.append(a)
                w -= a * wi

            norm_w = np.dot(w, w)
            if norm_w < epsilon:
                continue

            gi = np.dot(w, y) / norm_w
            err = (gi ** 2 * norm_w) / Ey

            if err > best_err:
                best_err = err
                best = (idx, w, gi, alpha)

        if best is None:
            break

        idx, w, gi, alpha = best
        selected.append(idx)
        remaining.remove(idx)
        W_ortho.append(w)
        A.append(alpha)
        g.append(gi)

        cum_err += best_err
        if cum_err >= err_threshold:
            break

    m = len(selected)
    theta = np.zeros(m)

    for i in range(m - 1, -1, -1):
        theta[i] = g[i]
        for k in range(i + 1, m):
            theta[i] -= A[k][i] * theta[k]

    P_sel = P[:, selected]
    W2_pred = P_sel @ theta

    class W2Model:
        def __init__(self, theta, selected):
            self.theta = theta
            self.selected = selected
        def predict(self, P):
            return P[:, self.selected] @ self.theta

    return W2_pred, W2Model(theta, selected), sL, sW, P


# =====================================================
# W3 线性融合（固定权重）
# =====================================================
def train_W3(W1, W2, Y):
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    Y = np.asarray(Y, dtype=float)

    e1 = Y - W1
    e2 = Y - W2

    var1 = np.var(e1) + epsilon
    var2 = np.var(e2) + epsilon

    alpha1 = var2 / (var1 + var2)
    alpha2 = 1 - alpha1

    W3 = alpha1 * W1 + alpha2 * W2
    return W3, alpha1, alpha2


# =====================================================
# MLP 神经网络（带 GridSearchCV）
# =====================================================
def train_mlp(L_train, W_train, Y_train, L_test, W_test):
    """MLP 神经网络回归，GridSearchCV 调参"""
    X_tr = np.column_stack([L_train, W_train])
    X_te = np.column_stack([L_test, W_test])

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    Y_tr_s = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()

    param_grid = {
        'hidden_layer_sizes': [(32,), (64,), (32, 16), (64, 32)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01],
    }

    grid = GridSearchCV(
        MLPRegressor(
            activation='relu',
            solver='adam',
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr_s, Y_tr_s)

    best = grid.best_estimator_
    X_te_s = scaler_X.transform(X_te)
    Yp_s = best.predict(X_te_s)
    Yp = scaler_Y.inverse_transform(Yp_s.reshape(-1, 1)).ravel()

    print(f"MLP 最优参数: {grid.best_params_}")
    print(f"MLP CV R² = {grid.best_score_:.4f}")
    print(f"MLP 隐藏层结构: {best.hidden_layer_sizes}")
    return Yp, best, scaler_X, scaler_Y


# =====================================================
# SVR 支持向量回归（带 GridSearchCV）
# =====================================================
def train_svr(L_train, W_train, Y_train, L_test, W_test):
    """SVR 回归，GridSearchCV 调参"""
    X_tr = np.column_stack([L_train, W_train])
    X_te = np.column_stack([L_test, W_test])

    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_tr)
    Y_tr_s = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).ravel()

    param_grid = {
        'kernel': ['rbf', 'poly', 'linear'],
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'epsilon': [0.01, 0.1],
    }

    grid = GridSearchCV(
        SVR(),
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=0,
    )
    grid.fit(X_tr_s, Y_tr_s)

    best = grid.best_estimator_
    X_te_s = scaler_X.transform(X_te)
    Yp_s = best.predict(X_te_s)
    Yp = scaler_Y.inverse_transform(Yp_s.reshape(-1, 1)).ravel()

    print(f"SVR 最优参数: {grid.best_params_}")
    print(f"SVR CV R² = {grid.best_score_:.4f}")
    return Yp, best, scaler_X, scaler_Y


# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":

    # ==================== 1. 数据加载 ====================
    train_file = os.path.join(desktop, "data1_data2.xlsx")
    df = pd.read_excel(train_file).dropna()

    L_train = df["L"].values.astype(float)
    W_train = df["W"].values.astype(float)
    Y_train = df["W1"].values.astype(float)

    print(f"训练集样本数: {len(L_train)}")

    test_path = os.path.join(desktop, "长吻鮠矩阵.txt")
    test_data = load_txt_with_brackets(test_path)

    L_test = test_data[:, 0].astype(float)
    W_test = test_data[:, 1].astype(float)
    Y_test = test_data[:, 2].astype(float)

    print(f"测试集样本数: {len(L_test)}")

    # ==================== 2. 训练所有模型 ====================
    results = {}   # {模型简称: 测试集预测值}

    # --- W1 幂函数模型 ---
    print("\n" + "=" * 60)
    print("  [1/5] W1 — 幂函数模型")
    print("=" * 60)
    W1_tr, w1_param = train_W1(L_train, W_train, Y_train)
    def W1_predict(Lv, Wv, p):
        return p[0] * (np.maximum(Lv, epsilon) ** p[1]) * (np.maximum(Wv, epsilon) ** p[2])
    W1_test = W1_predict(L_test, W_test, w1_param)
    results['W1 (幂函数)'] = W1_test
    print(f"  W1 测试集 R² = {r2_score(Y_test, W1_test):.6f}")

    # --- W2 FROE 正交回归 ---
    print("\n" + "=" * 60)
    print("  [2/5] W2 — FROE 正交回归")
    print("=" * 60)
    W2_tr, w2_model, sL, sW, P_train = train_W2_FROE(L_train, W_train, Y_train)

    Ls_test = sL.transform(L_test.reshape(-1, 1)).ravel()
    Ws_test = sW.transform(W_test.reshape(-1, 1)).ravel()
    terms_test = [np.ones(len(Ls_test)), Ls_test, Ws_test]
    for d in range(2, 13):
        for i in range(d + 1):
            j = d - i
            if i == 0:
                terms_test.append(Ws_test ** j)
            elif j == 0:
                terms_test.append(Ls_test ** i)
            else:
                terms_test.append((Ls_test ** i) * (Ws_test ** j))
    P_test = np.column_stack(terms_test)
    W2_test = w2_model.predict(P_test)
    results['W2 (FROE)'] = W2_test
    print(f"  W2 测试集 R² = {r2_score(Y_test, W2_test):.6f}")

    # --- W3 线性融合 ---
    print("\n" + "=" * 60)
    print("  [3/5] W3 — 线性融合")
    print("=" * 60)
    W3_tr, alpha1, alpha2 = train_W3(W1_tr, W2_tr, Y_train)
    W3_test = alpha1 * W1_test + alpha2 * W2_test
    results['W3 (线性融合)'] = W3_test
    print(f"  W3 测试集 R² = {r2_score(Y_test, W3_test):.6f}")

    # --- MLP 神经网络 ---
    print("\n" + "=" * 60)
    print("  [4/5] MLP 神经网络")
    print("=" * 60)
    mlp_pred, mlp_model, mlp_sX, mlp_sY = train_mlp(L_train, W_train, Y_train, L_test, W_test)
    results['MLP (神经网络)'] = mlp_pred
    print(f"  MLP 测试集 R² = {r2_score(Y_test, mlp_pred):.6f}")

    # --- SVR 支持向量回归 ---
    print("\n" + "=" * 60)
    print("  [5/5] SVR 支持向量回归")
    print("=" * 60)
    svr_pred, svr_model, svr_sX, svr_sY = train_svr(L_train, W_train, Y_train, L_test, W_test)
    results['SVR (支持向量)'] = svr_pred
    print(f"  SVR 测试集 R² = {r2_score(Y_test, svr_pred):.6f}")

    # ==================== 3. 汇总评估 ====================
    print("\n" + "=" * 70)
    print("  测试集评估汇总 (Test Set Evaluation Summary)")
    print("=" * 70)
    print(f"{'模型':24s} {'R²':>10s} {'RMSE':>10s}")
    print("-" * 50)

    for name, pred in results.items():
        r2 = r2_score(Y_test, pred)
        rmse = np.sqrt(mean_squared_error(Y_test, pred))
        print(f"{name:24s} {r2:10.6f} {rmse:10.4f}")

    print("-" * 50)

    # 找出最佳
    best_name = max(results, key=lambda k: r2_score(Y_test, results[k]))
    best_r2 = r2_score(Y_test, results[best_name])
    best_rmse = np.sqrt(mean_squared_error(Y_test, results[best_name]))
    print(f"\n  最佳模型: {best_name}  (R² = {best_r2:.6f}, RMSE = {best_rmse:.4f})")

    # ==================== 4. 绘图 ====================
    model_names = list(results.keys())
    model_predictions = list(results.values())
    n_models = len(model_names)

    r2_vals = [r2_score(Y_test, p) for p in model_predictions]
    rmse_vals = [np.sqrt(mean_squared_error(Y_test, p)) for p in model_predictions]

    # 颜色方案：前3个集成模型(蓝)，后2个ML模型(橙/红)
    bar_colors = ['#1f77b4', '#4A90D9', '#74B3E8', '#ff7f0e', '#d62728']

    # 简短 x 轴标签
    short_names = [
        'W1\n幂函数',
        'W2\nFROE',
        'W3\n线性融合',
        'MLP\n神经网络',
        'SVR\n支持向量',
    ]

    # --- 图1：R² 柱状图 ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(range(n_models), r2_vals, width=0.55, color=bar_colors, edgecolor='black', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, r2_vals)):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    ax1.set_xticks(range(n_models))
    ax1.set_xticklabels(short_names, fontsize=8)
    ax1.set_ylabel('决定系数 R²\nCoefficient of Determination')
    ax1.set_title('模型 R² 对比\nR² Comparison')
    ax1.set_ylim(0, max(r2_vals) * 1.18)
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path1 = os.path.join(desktop, 'ML_baseline_R2_comparison.jpg')
    plt.savefig(save_path1, format='jpg', dpi=600, bbox_inches='tight')
    print(f"\nR² 对比图已保存：{save_path1}")
    plt.show()

    # --- 图2：RMSE 柱状图 ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    bars = ax2.bar(range(n_models), rmse_vals, width=0.55, color=bar_colors, edgecolor='black', linewidth=0.5)
    for i, (bar, val) in enumerate(zip(bars, rmse_vals)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    ax2.set_xticks(range(n_models))
    ax2.set_xticklabels(short_names, fontsize=8)
    ax2.set_ylabel('均方根误差 RMSE\nRoot Mean Squared Error')
    ax2.set_title('模型 RMSE 对比\nRMSE Comparison')
    ax2.set_ylim(0, max(rmse_vals) * 1.18)
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    plt.tight_layout()
    save_path2 = os.path.join(desktop, 'ML_baseline_RMSE_comparison.jpg')
    plt.savefig(save_path2, format='jpg', dpi=600, bbox_inches='tight')
    print(f"RMSE 对比图已保存：{save_path2}")
    plt.show()

    # --- 图3：全部模型预测 vs 真实散点面板 (3×2) ---
    n_cols = 3
    n_rows = 2
    fig3, axes = plt.subplots(n_rows, n_cols, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (name, pred) in enumerate(results.items()):
        ax = axes[idx]
        y_min = min(Y_test.min(), pred.min())
        y_max = max(Y_test.max(), pred.max())
        ax.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=0.8, alpha=0.4)
        ax.scatter(Y_test, pred, c=bar_colors[idx], s=20, alpha=0.75,
                   edgecolors='black', linewidth=0.3)
        r2 = r2_score(Y_test, pred)
        rmse = np.sqrt(mean_squared_error(Y_test, pred))
        ax.set_xlabel('真实体重 (g)\nActual Weight')
        ax.set_ylabel('预测体重 (g)\nPredicted Weight')
        clean_name = name.split(' (')[0] if ' (' in name else name
        ax.set_title(f'{clean_name}\nR² = {r2:.4f}   RMSE = {rmse:.2f}')
        ax.grid(True, alpha=0.3)

    # 隐藏第6个空白子图
    axes[5].set_visible(False)

    fig3.suptitle('各模型预测值 vs 真实值对比\nPredicted vs Actual Weight — All Models',
                  fontsize=11, y=1.01)
    plt.tight_layout()
    save_path3 = os.path.join(desktop, 'ML_baseline_pred_vs_true_panels.jpg')
    plt.savefig(save_path3, format='jpg', dpi=600, bbox_inches='tight')
    print(f"多模型散点面板已保存：{save_path3}")
    plt.show()

    # --- 图4：MLP + SVR 特征重要性 ---
    from sklearn.inspection import permutation_importance

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(9, 4))

    # MLP 排列重要性
    X_te_for_pi = np.column_stack([L_test, W_test])
    pi_mlp = permutation_importance(mlp_model, X_te_for_pi, Y_test, n_repeats=10, random_state=42)
    mlp_imp = np.nan_to_num(pi_mlp.importances_mean, nan=0.0, posinf=0.0, neginf=0.0)
    mlp_ylim = max(np.max(mlp_imp), 0.01) * 1.3
    ax4a.bar(['体长 L\nLength', '体宽 W\nWidth'], mlp_imp, width=0.35,
             color=['#5B9BD5', '#ED7D31'], edgecolor='black')
    for i, v in enumerate(mlp_imp):
        ax4a.text(i, v + mlp_ylim * 0.02, f'{v:.4f}', ha='center', fontsize=9)
    ax4a.set_title('MLP 排列重要性\nMLP Permutation Importance')
    ax4a.set_ylabel('重要性 Importance')
    ax4a.set_ylim(0, mlp_ylim)
    ax4a.grid(axis='y', linestyle='--', alpha=0.3)

    # SVR 排列重要性
    pi_svr = permutation_importance(svr_model, X_te_for_pi, Y_test, n_repeats=10, random_state=42)
    svr_imp = np.nan_to_num(pi_svr.importances_mean, nan=0.0, posinf=0.0, neginf=0.0)
    svr_ylim = max(np.max(svr_imp), 0.01) * 1.3
    ax4b.bar(['体长 L\nLength', '体宽 W\nWidth'], svr_imp, width=0.35,
             color=['#5B9BD5', '#ED7D31'], edgecolor='black')
    for i, v in enumerate(svr_imp):
        ax4b.text(i, v + svr_ylim * 0.02, f'{v:.4f}', ha='center', fontsize=9)
    ax4b.set_title('SVR 排列重要性\nSVR Permutation Importance')
    ax4b.set_ylabel('重要性 Importance')
    ax4b.set_ylim(0, svr_ylim)
    ax4b.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout(pad=1.0)
    save_path4 = os.path.join(desktop, 'MLP_SVR_feature_importance.jpg')
    plt.savefig(save_path4, format='jpg', dpi=600)
    print(f"特征重要性图已保存：{save_path4}")
    plt.show()

    # --- 图5：MLP 预测值 vs 真实值 ---
    fig5, ax5 = plt.subplots(figsize=(8, 7))
    y_min = min(Y_test.min(), mlp_pred.min())
    y_max = max(Y_test.max(), mlp_pred.max())
    ax5.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=0.8, alpha=0.4)
    ax5.scatter(Y_test, mlp_pred, c='#ff7f0e', s=25, alpha=0.75,
                edgecolors='black', linewidth=0.3)
    r2_mlp = r2_score(Y_test, mlp_pred)
    rmse_mlp = np.sqrt(mean_squared_error(Y_test, mlp_pred))
    ax5.set_xlabel('真实体重 (g)\nActual Weight')
    ax5.set_ylabel('预测体重 (g)\nPredicted Weight')
    ax5.set_title(f'MLP 神经网络预测值 vs 真实值\nMLP Predicted vs Actual Weight\nR² = {r2_mlp:.4f}   RMSE = {rmse_mlp:.2f}')
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path5 = os.path.join(desktop, 'MLP_pred_vs_true.jpg')
    plt.savefig(save_path5, format='jpg', dpi=600, bbox_inches='tight')
    print(f"MLP 预测对比图已保存：{save_path5}")
    plt.show()

    # --- 图6：SVR 预测值 vs 真实值 ---
    fig6, ax6 = plt.subplots(figsize=(8, 7))
    y_min = min(Y_test.min(), svr_pred.min())
    y_max = max(Y_test.max(), svr_pred.max())
    ax6.plot([y_min, y_max], [y_min, y_max], 'k--', linewidth=0.8, alpha=0.4)
    ax6.scatter(Y_test, svr_pred, c='#d62728', s=25, alpha=0.75,
                edgecolors='black', linewidth=0.3)
    r2_svr = r2_score(Y_test, svr_pred)
    rmse_svr = np.sqrt(mean_squared_error(Y_test, svr_pred))
    ax6.set_xlabel('真实体重 (g)\nActual Weight')
    ax6.set_ylabel('预测体重 (g)\nPredicted Weight')
    ax6.set_title(f'SVR 支持向量回归预测值 vs 真实值\nSVR Predicted vs Actual Weight\nR² = {r2_svr:.4f}   RMSE = {rmse_svr:.2f}')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path6 = os.path.join(desktop, 'SVR_pred_vs_true.jpg')
    plt.savefig(save_path6, format='jpg', dpi=600, bbox_inches='tight')
    print(f"SVR 预测对比图已保存：{save_path6}")
    plt.show()

    print("\n" + "=" * 60)
    print("  全部图表已生成，保存至桌面。")
    print("=" * 60)
