import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==================== 全局设置 ====================
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.size'] = 8   # 改为8 pt
plt.rcParams['axes.unicode_minus'] = False

# 自动桌面路径（更稳）
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# ==================== 读取训练集 ====================
train_file = os.path.join(desktop, "data1_data2.xlsx")
df_train = pd.read_excel(train_file)

data_train = df_train.iloc[:, :3].copy()
data_train = data_train.apply(pd.to_numeric, errors='coerce')
data_train.dropna(inplace=True)
data_train.columns = ['Length', 'Width', 'Weight']

# ==================== 读取测试集 ====================
def load_txt(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().replace('[', '').replace(']', '').rstrip(',')
            if line:
                data.append([float(x) for x in line.split(',')])
    return np.array(data)

test_file = os.path.join(desktop, "长吻鮠矩阵.txt")
test_array = load_txt(test_file)

df_test = pd.DataFrame(test_array[:, :3], columns=['Length', 'Width', 'Weight'])

# ==================== 双语标签 ====================
variables = ['Length', 'Width', 'Weight']

titles = {
    'Length': '体长分布 \n Distribution of Length',
    'Width':  '体宽分布 \n Distribution of Width',
    'Weight': '体重分布 \n Distribution of Weight'
}

xlabels = {
    'Length': '体长 L / cm \n Length',
    'Width':  '体宽 W / cm \n Width',
    'Weight': '体重 W / g \n Weight'
}

ylabel = '概率密度 \n Probability Density'

# ==================== 绘图 ====================
for var in variables:

    plt.figure(figsize=(6, 4))

    train_vals = data_train[var]
    test_vals = df_test[var]

    # KDE 曲线
    sns.kdeplot(train_vals,
                fill=True,
                alpha=0.4,
                linewidth=1.5,
                label='训练集 (Training set)')

    sns.kdeplot(test_vals,
                fill=True,
                alpha=0.4,
                linewidth=1.5,
                label='测试集 (Test set)')

    # 标签
    plt.xlabel(xlabels[var])
    plt.ylabel(ylabel)
    plt.title(titles[var])

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # ==================== 保存 JPG（600 dpi） ====================
    save_path = os.path.join(desktop, f'kde_{var}_bilingual.jpg')

    plt.savefig(save_path,
                format='jpg',
                dpi=600,
                bbox_inches='tight')

    print(f"{var} 图已保存：{save_path}")

    plt.show()

print("\n✅ 已生成三张双语KDE图（600 dpi，JPG，论文可用）")