import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier

# load and prepare data
df = pd.read_csv(r"data\\raw\\raw.csv")
df = df.dropna(subset=['forward regime'])

# simple features
features_cols = ['open', 'high', 'low', 'close', 'volume', 
                 'past_ret_1h', 'past_ret_2h', 'past_ret_4h',
                 'past_ret_8h', 'past_ret_16h']

X = df[features_cols].fillna(0).values
y = df['forward regime'].map({'downward':0, 'range':1, 'upward':2}).values

# remove any remaining NaN/Inf
X = np.nan_to_num(X)

print(f"Total samples: {len(X)}")
print(f"Unique classes: {np.unique(y)}")

# split
n = len(X)
train_size = 2000
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# simple model
model = CatBoostClassifier(iterations=50, verbose=0)
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)
acc = np.mean(pred == y_test)
print(f"Success! Accuracy: {acc:.3f}")
print(f"Class distribution in test: {np.bincount(y_test)}")