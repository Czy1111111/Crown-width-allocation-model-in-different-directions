import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

# 1. 读取数据
file_path = r'F:\1\1.xlsx'
df = pd.read_excel(file_path)

# 2. 筛选 TYPE == 0（杉木）
df = df[df['TYPE'] == 0].copy()

# 3. 提取目标变量
y = df['单方向冠幅']

# 4. One-Hot 编码分类变量
cat_cols = ['方向', 'TYPE_nearest1', 'TYPE_nearest2']
df_cat = pd.get_dummies(df[cat_cols].astype(str), prefix=cat_cols)

# 5. 数值特征
exclude_cols = ['单方向冠幅'] + cat_cols
num_cols = [col for col in df.columns if col not in exclude_cols and df[col].dtype in [np.float64, np.int64]]

# 6. 合并所有特征
X = pd.concat([df[num_cols], df_cat], axis=1)

# 7. 去除缺失值
X = X[y.notna()]
y = y[y.notna()]

# 8. 三阶段划分：Train (70%) + Val (15%) + Test (15%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=42)

# 9. 定义模型
models = {
    'LightGBM': LGBMRegressor(n_estimators=100, early_stopping_rounds=10, random_state=42, verbose=-1),
    'CatBoost': CatBoostRegressor(n_estimators=100, early_stopping_rounds=10, verbose=0, random_state=42),
    'HistGB': HistGradientBoostingRegressor(max_iter=100, early_stopping=True, validation_fraction=0.15, random_state=42)
}

# 10. 保存特征用于 MoE
train_features = []
val_features = []
test_features = []

print("=== 模型性能（在 Test 上） ===")
for name, model in models.items():
    print(f"\n训练模型: {name}")
    if name == 'LightGBM':
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    elif name == 'CatBoost':
        model.fit(X_train, y_train, eval_set=(X_val, y_val))
    else:
        model.fit(X_train, y_train)

    train_pred = model.predict(X_train).reshape(-1, 1)
    val_pred = model.predict(X_val).reshape(-1, 1)
    test_pred = model.predict(X_test).reshape(-1, 1)

    train_features.append(train_pred)
    val_features.append(val_pred)
    test_features.append(test_pred)

    r2 = r2_score(y_test, test_pred)
    print(f"{name} R² on Test: {r2:.4f}")

# 11. 构建 MoE 输入特征
X_train_moe = np.concatenate(train_features, axis=1)
X_val_moe = np.concatenate(val_features, axis=1)
X_test_moe = np.concatenate(test_features, axis=1)

# 12. 保存特征及标签
np.save('X_train_moe.npy', X_train_moe)
np.save('X_val_moe.npy', X_val_moe)
np.save('X_test_moe.npy', X_test_moe)

np.save('y_train_moe.npy', y_train.values)
np.save('y_val_moe.npy', y_val.values)
np.save('y_test_moe.npy', y_test.values)
