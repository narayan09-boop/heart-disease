import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

df = pd.read_csv("dataset.csv")

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=df)
plt.title("Target Class Distribution")
plt.tight_layout()
plt.savefig("plots/target_distribution.png")
plt.close()

numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Histogram of {col}")
    plt.tight_layout()
    plt.savefig(f"plots/histogram_{col}.png")
    plt.close()

X = df.drop('target', axis=1)
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

y_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("plots/roc_curve.png")
plt.close()

# Custom patient input
print("\n--- Predict Heart Disease for New Patient ---")
input_data = []

input_data.append(float(input("Age: ")))
input_data.append(int(input("Sex (0 = Female, 1 = Male): ")))
input_data.append(int(input("Chest Pain Type (1-4): ")))
input_data.append(float(input("Resting Blood Pressure (mm Hg): ")))
input_data.append(float(input("Serum Cholesterol (mg/dL): ")))
input_data.append(int(input("Fasting Blood Sugar > 120? (1 = Yes, 0 = No): ")))
input_data.append(int(input("Resting ECG (0 = normal, 1 = ST-T abnormality, 2 = LVH): ")))
input_data.append(float(input("Maximum Heart Rate Achieved: ")))
input_data.append(int(input("Exercise Induced Angina (1 = Yes, 0 = No): ")))
input_data.append(float(input("Oldpeak (ST depression): ")))
input_data.append(int(input("Slope (1 = upsloping, 2 = flat, 3 = downsloping): ")))

input_array = np.array([input_data])
input_scaled = scaler.transform(input_array)
prediction = model.predict(input_scaled)[0]

if prediction == 1:
    print("\n⚠️ The model predicts: **Patient likely has Heart Disease.**")
else:
    print("\n✅ The model predicts: **Patient does NOT have Heart Disease.**")
