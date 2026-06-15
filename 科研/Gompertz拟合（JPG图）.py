import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
import os

#  字体设置
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']  # 英文+中文
plt.rcParams['font.size'] = 8   # 改为8 pt
plt.rcParams['axes.unicode_minus'] = False

desktop = r'C:\Users\33701\Desktop'

# 1. 读取训练集
train_file = r"C:\Users\33701\Desktop\data1_data2.xlsx"
df_train = pd.read_excel(train_file)

data_train = df_train.iloc[:, :3].copy()
for i in range(3):
    data_train.iloc[:, i] = pd.to_numeric(data_train.iloc[:, i], errors='coerce')
data_train.dropna(inplace=True)

if len(data_train) == 0:
    raise ValueError("训练集清洗后无有效数据")

L_train = data_train.iloc[:, 0].values
W_train = data_train.iloc[:, 2].values

# 2. 读取测试集
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

test_file = r"C:\Users\33701\Desktop\长吻鮠矩阵.txt"
test_data = load_txt_with_brackets(test_file)


L_test = test_data[:, 0]
W_test = test_data[:, 2]

#  3. Gompertz模型
def gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))

# 4. 拟合
a0 = np.max(W_train)
b0 = 1.0
c0 = 0.1 / (np.median(L_train) if np.median(L_train) != 0 else 1)
p0 = [a0, b0, c0]

popt, _ = curve_fit(gompertz, L_train, W_train, p0=p0, bounds=(0, np.inf), maxfev=10000)
a, b, c = popt

#  5. 测试集评估
W_test_pred = gompertz(L_test, *popt)

mse = mean_squared_error(W_test, W_test_pred)
rmse = np.sqrt(mse)

# 6. 绘图
L_min = min(min(L_train), min(L_test))
L_max = max(max(L_train), max(L_test))
L_range = np.linspace(L_min, L_max, 500)
W_fit = gompertz(L_range, *popt)

plt.figure(figsize=(10, 6))

plt.plot(L_range, W_fit,
         label="Gompertz拟合曲线(Gompertz fitting curve)",
         color='blue')

plt.scatter(L_test, W_test_pred,
            color='green',
            label='测试集预测值(Predicted values of the test set)',
            marker='x')

plt.ylabel("体长 L/cm \nLength")
plt.xlabel("体重 W/g \nWeight")
plt.title("Gompertz 模型：训练集与测试集效果\nGompertz model: Training Set and Test Set effects")

plt.legend()
plt.grid(True)
plt.tight_layout()

# 7. 保存为 JPG（600 dpi）
output_path = os.path.join(desktop, 'Gompertz_comparison_models.jpg')

plt.savefig(output_path,
            format='jpg',
            dpi=600,               # 600 dpi
            bbox_inches='tight')

plt.show()