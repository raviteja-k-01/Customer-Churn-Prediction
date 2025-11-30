import pickle
import pandas as pd

# ✅ Load trained model
model = pickle.load(open(r"C:\Users\ravit\Desktop\Customer\model\churn_model.pkl", "rb"))

# 🧠 Full customer data (all expected features)
new_customer = {
     "Tenure": 2,                        # Very new customer (just joined)
    "PreferredLoginDevice": 1,          # Uses desktop rarely (not mobile-active)
    "CityTier": 3,                      # Lives in a small city with fewer facilities
    "WarehouseToHome": 40,              # Far from warehouse (long delivery times)
    "PreferredPaymentMode": 2,          # Card (but infrequent online payments)
    "Gender": 1,                        # Female
    "HourSpendOnApp": 1,                # Hardly uses the app
    "NumberOfDeviceRegistered": 1,      # Only one device
    "PreferedOrderCat": 3,              # Grocery — low-margin, low frequency
    "SatisfactionScore": 1,             # Very low satisfaction
    "MaritalStatus": 0,                 # Single
    "NumberOfAddress": 1,               # One address only
    "Complain": 3,                      # Multiple complaints
    "OrderAmountHikeFromlastYear": -10, # Spending dropped from last year
    "CouponUsed": 0,                    # Doesn’t use offers anymore
    "OrderCount": 2,                    # Rarely orders
    "DaySinceLastOrder": 180,           # 6 months since last order
    "CashbackAmount": 0 # No cashback
}

# ✅ All features your model expects
expected_features = [
    'Tenure', 'PreferredLoginDevice', 'CityTier', 'WarehouseToHome', 'PreferredPaymentMode',
    'Gender', 'HourSpendOnApp', 'NumberOfDeviceRegistered', 'PreferedOrderCat',
    'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain',
    'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount',
    'DaySinceLastOrder', 'CashbackAmount'
]

# 🧮 Create DataFrame in the same order as training
df = pd.DataFrame([{feature: new_customer.get(feature, 0) for feature in expected_features}])

# 🔍 Predict churn
prediction = model.predict(df)[0]
prob = model.predict_proba(df)[0][1]

# 🎯 Show result
if prediction == 1:
    print(f"⚠️ Customer likely to CHURN. (Confidence: {prob:.2f})")
else:
    print(f"✅ Customer likely to STAY. (Confidence: {1 - prob:.2f})")
