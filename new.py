import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

## from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\data\processed\\xau_with_all_macro.csv")
print(df)

features = [
    "date",
    "open","high","low","close","volume",
     "past_ret_1h","past_ret_2h",
     "past_ret_4h",
    "past_ret_8h","past_ret_16h",
    "probability",
     "0.6 percent prediction","1 percent prediction","1.5 percent prediction", 
    'vix', 'dxy',
    'tnx', 'real_rate',	
    'breakeven_inflation', 'gold_yield_spread',	
    'tnx_change_5d','real_rate_change_5d','negative_real_rate',	
    'tnx_change_24h', 
    'real_rate_change_24h', 'tnx_change_168h'
]

df["date"] = pd.to_datetime(df["date"]).map(pd.Timestamp.toordinal)
#df = df.fillna(method="ffill").fillna(method="bfill")
df = df.ffill().bfill()

mapping = {"range": 0, "upward": 1, "downward": -1}
df["forward regime"] = df["forward regime"].map(mapping)

X = df[features]
y = df["forward regime"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


class_weights = {
    0: 1.511747430249633,
    1: 5.33419689119171,
    -1: 6.620578778135048
}

pipe = ImbPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("smote", SMOTE(random_state=42)),
    ("clf", RandomForestClassifier(
        n_estimators=400, 
        max_depth=None,                  
        min_samples_split=4,       
        min_samples_leaf=2,  
        class_weight=class_weights,
        n_jobs=-1,
        random_state=42
    ))
])


pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)
print("\n--- Classification Report (RandomForest) ---\n")
print(classification_report(y_test, y_pred))

y_score = pipe.predict_proba(X_test)
classes = sorted(y.unique())

plt.figure(figsize=(8,6))
for i, c in enumerate(classes):
    fpr, tpr, _ = roc_curve((y_test == c).astype(int), y_score[:, i])
    plt.plot(fpr, tpr, label=f"Class {c}")
plt.plot([0,1], [0,1], 'k--')
plt.legend(); plt.title("ROC Curve – RandomForest")
plt.xlabel("FPR"); plt.ylabel("TPR")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,6))
for i, c in enumerate(classes):
    prec, rec, _ = precision_recall_curve((y_test == c).astype(int), y_score[:, i])
    plt.plot(rec, prec, label=f"Class {c}")
plt.legend(); plt.title("PR Curve – RandomForest")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.grid(True)
plt.show()

clf = pipe.named_steps["clf"]
importances = clf.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.title("Random Forest Feature Importance")
plt.show()