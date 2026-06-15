import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==================== 字体设置（符合要求） ====================
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.size'] = 8   # 改为8 pt
plt.rcParams['axes.unicode_minus'] = False

# 自动桌面路径（更安全）
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# ==================== 1. 读取训练集 ====================
train_file = os.path.join(desktop, "data1_data2.xlsx")
df_train = pd.read_excel(train_file)

data_train = df_train.iloc[:, :3].copy()
for i in range(3):
    data_train.iloc[:, i] = pd.to_numeric(data_train.iloc[:, i], errors='coerce')
data_train.dropna(inplace=True)

if len(data_train) == 0:
    raise ValueError("请检查文件内容。")

L_train = data_train.iloc[:, 0].values.astype(float)
W_train = data_train.iloc[:, 2].values.astype(float)

# ==================== 2. 读取测试集 ====================
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

test_file = os.path.join(desktop, "长吻鮠矩阵.txt")
test_data = load_txt_with_brackets(test_file)

if test_data.shape[1] < 3:
    raise ValueError("测试集至少需要三列")

L_test = test_data[:, 0].astype(float)
W_test = test_data[:, 2].astype(float)

# ==================== 3. 多项式拟合 ====================
mse_dict, rmse_dict, r2_dict = {}, {}, {}
pred_dict, coeffs_dict = {}, {}

L_plot = np.linspace(min(L_test), max(L_test), 300)

for degree in range(1, 7):
    X_train = np.vander(L_train, degree + 1, increasing=True)
    coeffs = np.linalg.lstsq(X_train, W_train, rcond=None)[0]
    coeffs_dict[degree] = coeffs

    X_test = np.vander(L_test, degree + 1, increasing=True)
    W_pred_test = X_test @ coeffs

    mse_dict[degree] = mean_squared_error(W_test, W_pred_test)
    rmse_dict[degree] = np.sqrt(mse_dict[degree])
    r2_dict[degree] = r2_score(W_test, W_pred_test)

    X_plot = np.vander(L_plot, degree + 1, increasing=True)
    pred_dict[degree] = X_plot @ coeffs

# ==================== 4. 选最佳模型 ====================
best_degree = min(mse_dict, key=mse_dict.get)
best_coeffs = coeffs_dict[best_degree]

# ==================== 5. 计算最佳预测 ====================
X_test_best = np.vander(L_test, best_degree + 1, increasing=True)
W_pred_test_best = X_test_best @ best_coeffs

# ==================== 6. 绘图 ====================
plt.figure(figsize=(8, 5))

plt.plot(L_plot, pred_dict[best_degree],
         color='blue',
         linewidth=2,
         label=f'{best_degree}阶拟合曲线 ({best_degree} order fitting curve)')

plt.scatter(L_test, W_pred_test_best,
            color='green',
            s=20,
            zorder=5,
            label='测试集预测值 (Test set predicted values)',
            marker='x')

plt.xlabel('体长 L/cm \nLength')
plt.ylabel('体重 W/g \nWeight')

plt.title(f'测试集预测值拟合曲线（{best_degree}阶）\nTest set prediction value fitting curve ({best_degree} order)')

plt.legend()
plt.grid(True)
plt.tight_layout()

# ==================== 7. 保存 JPG（600 dpi） ====================
save_path = os.path.join(desktop, f'test_prediction_curve_order{best_degree}.jpg')

plt.savefig(save_path,
            format='jpg',
            dpi=600,
            bbox_inches='tight')

print(f"\n图像已保存至：{save_path}")

plt.show()