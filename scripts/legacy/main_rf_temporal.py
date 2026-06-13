import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

DATA_PATH = r"C:\Users\user\OneDrive\Desktop\financial_regime_clasification\data\processed\xau_with_all_macro.csv"

TARGET = "forward regime"

FEATURES = [
    "open", "high", "low", "close", "volume",
    "past_ret_1h", "past_ret_2h", "past_ret_4h",
    "past_ret_8h", "past_ret_16h",
    "vix", "dxy",
    "tnx", "real_rate",
    "breakeven_inflation", "gold_yield_spread",
    "tnx_change_5d", "real_rate_change_5d",
    "negative_real_rate",
    "tnx_change_24h",
    "real_rate_change_24h",
    "tnx_change_168h",
]

df = pd.read_csv(DATA_PATH)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

mapping = {
    "downward": -1,
    "range": 0,
    "upward": 1,
}

df[TARGET] = df[TARGET].map(mapping)

df = df.ffill().bfill()

X = df[FEATURES].copy()
y = df[TARGET].copy()

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test  = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test  = y.iloc[split_idx:]

class_weights = {
    0: 1.51174,
    1: 5.3341,
    -1: 6.6205,
}

pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),

    # SMOTE only affects training indside fit().
    # it is NOT applied to the test set.
    ("smote", SMOTE(random_state=42)),

    ("clf", RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split = 4,
        min_samples_leaf = 2,
        class_weight= class_weights,
        n_jobs = -1,
        random_state=42,
    ))
])

pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

print("\n === classification report: random forest temporal split ====\n")
print(classification_report(y_test, y_pred))

print("\n ==== confusion matrix ====\n")
print(confusion_matrix(y_test, y_pred))

clf = pipe.named_steps['clf']
importances = clf.feature_importances_

plt.figure(figsize=(10, 7))
plt.barh(FEATURES, importances)
plt.title("random forest feature importance - temporal split")
plt.tight_layout()
plt.show()