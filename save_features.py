import pandas as pd
import xgboost as xgb
import joblib

# 📥 Load and prepare your training data
df = pd.read_csv("D:\\Ishika!\\AI\\house-prices-advanced-regression-techniques\\train house price.csv")

# 🎯 Separate features and target
x_train = df.drop(columns=["SalePrice"])
y_train = df["SalePrice"]

# 🧼 Encode categorical columns
x_train = pd.get_dummies(x_train)

# 🧼 Handle missing values
x_train = x_train.fillna(x_train.mean())

# 🧠 Train your model
model = xgb.XGBRegressor()
model.fit(x_train, y_train)

# 💾 Save the model and feature list
joblib.dump(model, "xgb_model.joblib")
joblib.dump(x_train.columns.tolist(), "feature_list.pkl")
print("✅ Model and feature list saved successfully.")