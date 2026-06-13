
"""
correction: we should not let TCN sequences cross from Silver into DXY. Ill give you a clean pretraining script that builds sequences per market safely.
"""

import os
import joblib
import torch
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder

from src.models.tcn_model import TCNClassifier

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import WeightedRandomSampler


X_PATH = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\data\\pretrain\\X_pretrain.npy"
Y_PATH = r"C:\\Users\\user\\OneDrive\\Desktop\\\financial_regime_clasification\\data\\pretrain\\y_pretrain.npy"
OUT_DIR = r"C:\\Users\\user\\OneDrive\\Desktop\\financial_regime_clasification\\experiments\\pretraining"

SEQUENCE_LENGTH = 36
NUM_CHANNELS = (16,32)
KERNEL_SIZE = 2
DROPOUT =  0.25          #0.35039058736129347
EPOCHS = 38
BATCH_SIZE = 32
LEARNING_RATE = 0.001966625789577707

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []

    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i: i + sequence_length])
        y_seq.append(y[i + sequence_length - 1])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X = np.load(X_PATH)
    y = np.load(Y_PATH, allow_pickle=True)

    print("Loaded pretrainig data:")
    print("X:", X.shape)
    print("y:", y.shape)
    print(pd.Series(y).value_counts())

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y_encoded, SEQUENCE_LENGTH)

    print("X_seq:", X_seq.shape)
    print("y_seq:", y_seq.shape)

    split_idx = int(len(X_seq) * 0.8)

    X_train = X_seq[:split_idx]
    X_valid = X_seq[split_idx:]

    y_train = y_seq[:split_idx]
    y_valid = y_seq[split_idx:]

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    X_train_tensor = torch.FloatTensor(X_train).to(DEVICE)
    y_train_tensor = torch.LongTensor(y_train).to(DEVICE)

    X_valid_tensor = torch.FloatTensor(X_valid).to(DEVICE)

    model = TCNClassifier(
        input_size = X.shape[1],
        num_classes = len(label_encoder.classes_),
        num_channels= NUM_CHANNELS,
        kernel_size= KERNEL_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = LEARNING_RATE,
    )

    # ver 0
    criterion = torch.nn.CrossEntropyLoss()

    # class weights added to 0 version - ver 1
    # classes = np.unique(y_train)
    #
    # weights = compute_class_weight(
    # class_weight="balanced",
    # classes=classes,
    # y=y_train,
    # )
    
    # weights = torch.FloatTensor(weights).to(DEVICE)
    
    # criterion = torch.nn.CrossEntropyLoss(weight=weights)

    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)

    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
        )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        )
    
    # ver 0 and 1
    #loader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=BATCH_SIZE,
    #    shuffle=False,
    #)

    model.train()

    for epoch in range(EPOCHS):
        losses = []

        for batch_X, batch_y in loader:
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if (epoch+1) % 10 == 0:
            print(f"epoch {epoch+1}/{EPOCHS}, loss {np.mean(losses):.4f}")

    model.eval()

    with torch.no_grad():
        valid_logits = model(X_valid_tensor)
        valid_pred = torch.argmax(valid_logits, dim=1).cpu().numpy()

    macro_f1 = f1_score(y_valid, valid_pred, average='macro')

    print('\nPretraiinig validation macro f1:', macro_f1)

    print(
        classification_report(
        y_valid,
        valid_pred,
        target_names = [str(c) for c in label_encoder.classes_],
    )
    )

    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'input_size': X.shape[1],
            'num_classes': len(label_encoder.classes_),
            'sequence_length': SEQUENCE_LENGTH,
            'num_channels': NUM_CHANNELS,
            'kernel_size': KERNEL_SIZE,
            'dropout': DROPOUT,
            'learniing_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'label_encoder': label_encoder,
        },
        os.path.join(OUT_DIR, 'pretrained_tcn.pt')
    )

    joblib.dump(
        scaler,
        os.path.join(OUT_DIR, 'pretrained_scaler.pkl')
    )

    print('\nSaved pretrained model:')
    print(os.path.join(OUT_DIR, 'pretrained_tcn.pt'))
    print(os.path.join(OUT_DIR, 'pretrained_scaler.pkl'))

if __name__ == "__main__":
    main()