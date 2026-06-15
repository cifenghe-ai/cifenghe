import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== 字体设置 ====================
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['font.size'] = 8   # 改为8 pt
plt.rcParams['axes.unicode_minus'] = False

# 自动获取桌面路径（更安全）
desktop = os.path.join(os.path.expanduser("~"), "Desktop")

# ==================== 模型名称 ====================
models = ['多项式拟合\nPolynomial fitting', 'Gompertz拟合\nGompertz fitting', 'Y1', 'Y2', 'Y3']

# ==================== 数据 ====================
rmse_values = [80.866009, 121.7770, 76.576577, 45.914234, 33.408051]
r2_values = [0.878710, 0.7249, 0.891236, 0.960899, 0.979299]

# ==================== 图1：RMSE ====================
plt.figure(figsize=(8, 5))

plt.bar(models, rmse_values,
        width=0.6,
        color='coral',
        edgecolor='black')

plt.xlabel('模型\nModel')
plt.ylabel('均方根误差 RMSE\nRoot Mean Squared Error')
plt.title('不同模型的 RMSE 比较\nComparison of RMSE across different models')

plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path1 = os.path.join(desktop, 'RMSE_comparison_models.jpg')

plt.savefig(save_path1,
            format='jpg',
            dpi=600,
            bbox_inches='tight')

print(f"RMSE图已保存：{save_path1}")

plt.show()

# ==================== 图2：R² ====================
plt.figure(figsize=(8, 5))

plt.bar(models, r2_values,
        width=0.6,
        color='seagreen',
        edgecolor='black')

plt.xlabel('模型\nModel')
plt.ylabel('决定系数 R²\nCoefficient of determination')
plt.title('不同模型的 R² 比较\nComparison of R² values across different models')

plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

save_path2 = os.path.join(desktop, 'R2_comparison_models.jpg')

plt.savefig(save_path2,
            format='jpg',
            dpi=600,
            bbox_inches='tight')

print(f"R²图已保存：{save_path2}")

plt.show()