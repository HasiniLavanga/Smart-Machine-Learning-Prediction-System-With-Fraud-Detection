import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset (TAKE SAMPLE for speed 🔥)
df = pd.read_csv("creditcard.csv")

# 👉 Reduce data size (VERY IMPORTANT)
df = df.sample(n=50000, random_state=42)

# Preprocessing
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df = df.drop(['Time'], axis=1)

# Features & target
X = df.drop('Class', axis=1)
y = df['Class']

# Handle imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42
)

# 👉 Faster model
model = RandomForestClassifier(n_estimators=50, random_state=42)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "fraud_model.pkl")

print("✅ Model saved successfully!")