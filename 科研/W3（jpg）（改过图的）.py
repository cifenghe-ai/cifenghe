import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

epsilon = 1e-12

def load_txt_with_brackets(path):
    """加载带括号的文本数据"""
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
# W1 —— 幂函数模型
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

    print("\n===== W1 =====")
    print(f"W1 = {popt[0]:.6f} * L^{popt[1]:.6f} * W^{popt[2]:.6f}")
    print(f"W1 R² = {r2_score(Y, Yp):.6f}")

    return Yp, popt


# =====================================================
# W2 —— FROE（ERR）
# =====================================================
def train_W2_FROE(L, W, Y, max_degree=12, err_threshold=0.99):
    # 输入标准化
    sL, sW = StandardScaler(), StandardScaler()
    Ls = sL.fit_transform(L.reshape(-1, 1)).ravel()
    Ws = sW.fit_transform(W.reshape(-1, 1)).ravel()

    y = Y.copy()
    #移除常数项
    #terms = [Ls, Ws]
    #names = ['L', 'W']
    #未移除常数项
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

    # FROE
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

        #=======================每项err输出====================================
        print("\n===== 懒狗 free 一行代码都不愿意写  =====")
        print(f"Step {len(selected)}: '{names[idx]}'  ERR = {best_err:.6f}")
        #====================================================================

        cum_err += best_err
        if cum_err >= err_threshold:
            break

    # 回代求 θ
    m = len(selected)
    theta = np.zeros(m)

    for i in range(m - 1, -1, -1):
        theta[i] = g[i]
        for k in range(i + 1, m):
            theta[i] -= A[k][i] * theta[k]

    # 输出模型
    print("\n===== W2 (FROE) =====")
    print("W2 = ", end="")
    for i, idx in enumerate(selected):
        c = theta[i]
        sign = "+" if c >= 0 else "-"
        print(f"{sign} {abs(c):.6f} * {names[idx]} ", end="")
    print()

    P_sel = P[:, selected]
    W2_pred = P_sel @ theta

    print(f"W2 R² = {r2_score(Y, W2_pred):.6f}")
    print(f"累计 ERR = {cum_err:.4f}")

    class W2Model:
        def __init__(self, theta, selected):
            self.theta = theta
            self.selected = selected

        def predict(self, P):
            return P[:, self.selected] @ self.theta

    return W2_pred, W2Model(theta, selected), sL, sW, P


# =====================================================
# W3 线性融合
# =====================================================
def train_W3(W1, W2, Y):
    W1 = np.asarray(W1, dtype=float)
    W2 = np.asarray(W2, dtype=float)
    Y  = np.asarray(Y,  dtype=float)

    # 残差
    e1 = Y - W1
    e2 = Y - W2

    # 残差的总体方差（分母 N）
    var1 = np.var(e1) + epsilon   # 防止除零
    var2 = np.var(e2) + epsilon

    # 固定权重（最优线性无偏估计）
    alpha1 = var2 / (var1 + var2)
    alpha2 = 1 - alpha1

    # 融合输出
    W3 = alpha1 * W1 + alpha2 * W2

    print("\n===== W3 (固定权重融合) =====")
    print(f"W1 残差方差 = {var1:.6f}")
    print(f"W2 残差方差 = {var2:.6f}")
    print(f"alpha1 = {alpha1:.6f}, alpha2 = {alpha2:.6f}")
    print(f"W3 = {alpha1:.6f} * W1 + {alpha2:.6f} * W2")

    return W3, alpha1, alpha2   # 返回标量权重
# =====================================================
# 主程序
# =====================================================
if __name__ == "__main__":

    df = pd.read_excel(r"C:\Users\33701\Desktop\data1_data2.xlsx").dropna()

    L = df["L"].values
    W = df["W"].values
    Y = df["W1"].values

    W1_tr, w1_param = train_W1(L, W, Y)
    W2_tr, w2_model, sL, sW, P = train_W2_FROE(L, W, Y)

    W3_tr, alpha1_fixed, alpha2_fixed = train_W3(W1_tr, W2_tr, Y)

    print("\n===== 训练集性能 =====")
    print(f"W3 R² = {r2_score(Y, W3_tr):.6f}")
    print(f"W3 RMSE = {np.sqrt(mean_squared_error(Y, W3_tr)):.6f}")

    # 加载测试集并评估
    test_path = r"C:\Users\33701\Desktop\长吻鮠矩阵.txt"
    test_data = load_txt_with_brackets(test_path)

    if test_data.shape[1] >= 3:
        L_test = test_data[:, 0]  # 假设第一列是长
        W_test = test_data[:, 1]  # 假设第二列是宽
        Y_test = test_data[:, 2]  # 假设第三列是重

        print(f"\n===== 测试集评估 =====")
        print(f"测试集样本数: {len(L_test)}")


        # W1在测试集的预测函数
        def W1_predict(L_val, W_val, params):
            k, b_exp, d_exp = params
            return k * (np.maximum(L_val, epsilon) ** b_exp) * (np.maximum(W_val, epsilon) ** d_exp)


        # W1测试集预测
        W1_test_pred = W1_predict(L_test, W_test, w1_param)
        print(f"W1 测试集 R² = {r2_score(Y_test, W1_test_pred):.6f}")
        print(f"W1 测试集 RMSE = {np.sqrt(mean_squared_error(Y_test, W1_test_pred)):.6f}")

        # W2在测试集的预测
        # 首先标准化测试集的L和W
        Ls_test = sL.transform(L_test.reshape(-1, 1)).ravel()
        Ws_test = sW.transform(W_test.reshape(-1, 1)).ravel()

    #移除常数项
       # terms_test = [Ls_test, Ws_test]
    # 未移除常数项
        terms_test = [np.ones(len(Ls_test)), Ls_test, Ws_test]
        max_degree = 12
        for d in range(2, max_degree + 1):
            for i in range(d + 1):
                j = d - i
                if i == 0:
                    terms_test.append(Ws_test ** j)
                elif j == 0:
                    terms_test.append(Ls_test ** i)
                else:
                    terms_test.append((Ls_test ** i) * (Ws_test ** j))

        P_test = np.column_stack(terms_test)
        W2_test_pred = w2_model.predict(P_test)
        print(f"W2 测试集 R² = {r2_score(Y_test, W2_test_pred):.6f}")
        print(f"W2 测试集 RMSE = {np.sqrt(mean_squared_error(Y_test, W2_test_pred)):.6f}")

        # W3在测试集的预测
        # ===== W3（测试集 Gaussian 自适应融合）=====
        #e1_test = Y_test - W1_test_pred
        #e2_test = Y_test - W2_test_pred

        #sigma1_test = np.std(e1_test) + epsilon
        #sigma2_test = np.std(e2_test) + epsilon

        #p1_test = np.exp(-(e1_test ** 2) / (2 * sigma1_test ** 2)) / sigma1_test
        #p2_test = np.exp(-(e2_test ** 2) / (2 * sigma2_test ** 2)) / sigma2_test

        #alpha1_test = p1_test / (p1_test + p2_test + epsilon)
        #alpha2_test = p2_test / (p1_test + p2_test + epsilon)

        W3_test_pred = alpha1_fixed * W1_test_pred + alpha2_fixed * W2_test_pred

        print(f"W3 测试集 R² = {r2_score(Y_test, W3_test_pred):.6f}")
        print(f"W3 测试集 RMSE = {np.sqrt(mean_squared_error(Y_test, W3_test_pred)):.6f}")
        print(f"融合权重（来自训练集）: alpha1={alpha1_fixed:.6f}, alpha2={alpha2_fixed:.6f}")

    else:
        print(f"\n===== 测试集数据格式错误 =====")
    # ==================== 绘制 W3 融合模型预测 vs 真实对比图 ====================
    import matplotlib.pyplot as plt
    import os

    # 设置中文字体
    plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
    plt.rcParams['font.size'] = 8  # 修改为8 pt
    plt.rcParams['axes.unicode_minus'] = False

    desktop = r'C:\Users\33701\Desktop'

    #========================================================================================
    "请不要用那可笑的改这里"
    plt.figure(figsize=(6, 6))
    plt.scatter(Y_test, Y_test, marker='x', s=30, c='blue', alpha=0.6, label='真实体重 （Actual Weight）')
    plt.scatter(Y_test, W3_test_pred, marker='o', s=30, c='green', alpha=0.6, label='预测体重 （Predicted Weight）')

    plt.xlabel('体重 (g)\nWeight (g)')
    plt.ylabel('真实体重\预测体重 (g)\nActual Weight\Predicted Weight (g)')
    plt.title('实际体重与预测体重对比\nActual vs Predicted Weight')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    #========================================================================================
    
    # ==================== 改这里 ====================
    save_path = os.path.join(desktop, 'W3_pred_vs_true.jpg')

    plt.savefig(save_path,
                format='jpg',
                dpi=600,
                bbox_inches='tight')

    print(f"\nW3 对比图已保存至：{save_path}")

    plt.show()