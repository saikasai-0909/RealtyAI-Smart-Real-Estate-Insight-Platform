import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Simulated dataset for demo ---
# Suppose each region has two features: mean_intensity and area
X = np.random.rand(200, 2)  # 200 samples, 2 features
y = np.random.choice(["Residential", "Commercial"], size=200)

# --- Split and train ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)

# --- Evaluate ---
y_pred = clf.predict(X_test)
print("✅ Classifier Accuracy:", accuracy_score(y_test, y_pred))

# --- Save the model ---
joblib.dump(clf, "residential_commercial_classifier.joblib")
print("💾 Model saved as 'residential_commercial_classifier.joblib'")
