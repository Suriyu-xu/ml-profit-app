import numpy as np
from sklearn.linear_model import LinearRegression
# 数据拆分成「训练集」和「测试集」
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
data_v1 = np.array([
    [1200, 3000, 500, 8500],
    [1500, 3600, 600, 10200],
    [1100, 2800, 400, 7800],
    [1800, 4200, 800, 12500],
    [2000, 4800, 1000, 13800],
    [1600, 3900, 700, 11000],
    [1400, 3500, 550, 9600],
    [1900, 4500, 900, 13200],
    [1700, 4100, 750, 11800],
    [1300, 3200, 450, 8900],
    [2100, 5000, 1100, 14500],
    [2300, 5500, 1200, 15800]
])
# 影响利润的3个因素（客流量、成本、营销费）
X_v1 = data_v1[:, :3]
# 净利润
y_v1 = data_v1[:, 3]
X_train_v1, X_test_v1, y_train_v1, y_test_v1 = train_test_split(
    X_v1, y_v1, test_size=0.2, random_state=42
)
#初始化线性回归模型
model_baseline = LinearRegression()
# 训练模型，自动计算每个因素对利润的影响权重
model_baseline.fit(X_train_v1, y_train_v1)
# 测试模型
y_pred_v1 = model_baseline.predict(X_test_v1)
# 均方误差越小越好
mse_v1 = mean_squared_error(y_test_v1, y_pred_v1)
# 模型拟合程度，越接近1越好
r2_v1 = r2_score(y_test_v1, y_pred_v1)
print(f"  客流量：每增加1人，净利润增加 {model_baseline.coef_[0]:.2f} 元")
print(f"  原料成本：每增加1元，净利润变化 {model_baseline.coef_[1]:.2f} 元")
print(f"  营销费用：每增加1元，净利润变化 {model_baseline.coef_[2]:.2f} 元")
print(f"  均方误差(MSE)：{mse_v1:.2f} 元²")
print(f"  R²拟合度：{r2_v1:.4f}")
# 下月经营计划用以预测，预计2000客人、4700原料、900营销费
next_month = np.array([[2000, 4700, 900]])
predicted_profit = model_baseline.predict(next_month)[0]
print(f"  下月预计客流量：2000人")
print(f"  下月预计原料成本：4700元")
print(f"  下月预计营销费用：900元")
print(f"\n模型预测净利润：{predicted_profit:.2f} 元")
# 数据集v2：修正了异常数据，进行数据清洗，14500→14600，并新增数据
data_v2 = np.array([
    [1200, 3000, 500, 8500],
    [1500, 3600, 600, 10200],
    [1100, 2800, 400, 7800],
    [1800, 4200, 800, 12500],
    [2000, 4800, 1000, 13800],
    [1600, 3900, 700, 11000],
    [1400, 3500, 550, 9600],
    [1900, 4500, 900, 13200],
    [1700, 4100, 750, 11800],
    [1300, 3200, 450, 8900],
    [2100, 5000, 1100, 14600],
    [2300, 5500, 1200, 15800],
    [2500, 6000, 1500, 17200],
    [2400, 5800, 1300, 16500],
    [2200, 5300, 1150, 15100]
])
X_v2 = data_v2[:, :3]
y_v2 = data_v2[:, 3]
X_train_v2, X_test_v2, y_train_v2, y_test_v2 = train_test_split(
    X_v2, y_v2, test_size=0.2, random_state=42
)
model_improved = LinearRegression()
model_improved.fit(X_train_v2, y_train_v2)
y_pred_v2 = model_improved.predict(X_test_v2)
mse_v2 = mean_squared_error(y_test_v2, y_pred_v2)
r2_v2 = r2_score(y_test_v2, y_pred_v2)
print(f"  均方误差：{mse_v2:.2f} 元²")
print(f"  R²拟合度：{r2_v2:.4f}")
print(f"  MSE降低了 {mse_v1 - mse_v2:.2f}，误差更小")
print(f"  R²提升了 {r2_v2 - r2_v1:.4f}，拟合度更高")
print("\n")
# 测试输入全为0时，利润应接近0
test_1_input = np.array([[0, 0, 0]])
test_1_result = model_improved.predict(test_1_input)[0]
print(f"  预测利润：{test_1_result:.2f} 元")
# 测试客流量增加，利润应上升
test_2_low = model_improved.predict(np.array([[1000, 3000, 500]]))[0]
test_2_high = model_improved.predict(np.array([[2000, 3000, 500]]))[0]
print(f"  测试客流量1000人 → 利润{test_2_low:.2f}元")
print(f"  测试客流量2000人 → 利润{test_2_high:.2f}元")
# 测试原料成本升高，利润应该下降
test_3_low = model_improved.predict(np.array([[1500, 3000, 500]]))[0]
test_3_high = model_improved.predict(np.array([[1500, 5000, 500]]))[0]
print(f"  测试成本3000元 → 利润{test_3_low:.2f}元")
print(f"  测试成本5000元 → 利润{test_3_high:.2f}元")



