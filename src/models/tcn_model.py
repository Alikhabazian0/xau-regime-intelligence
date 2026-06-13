"""
Temporal Convolutional Network (TCN) for financial regim classification
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class Chomp1d(nn.Module):
    """Remove right-side padding for causal convolution"""
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()
    
class TemporalBlock(nn.Module):
    """singal TCN Block with dilated causal convolutions"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs,
                               kernel_size, stride=stride, 
                               padding=padding, dilation=dilation)
        self.chomp1   = Chomp1d(padding)
        self.relu1    = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2   = Chomp1d(padding)
        self.relu2    = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2 
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    """Stack of Temporal Blocks""" 
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        # num_levels = len(num_channels)

        for i in range(len(num_channels)):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            #layers += [TemporalBlock(in_channels, out_channels, kernel_size,
            #                         stride=1, dilation=dilation_size,
            #                         padding=(kernel_size-1) * dilation_size,
            
            #                         dropout=dropout)]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                              dilation=dilation_size, padding=(kernel_size-1)*dilation_size,
                              dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNClassifier(nn.Module):
    """TCN for classification"""
    def __init__(self, input_size, num_classes, num_channels=(64,128,64),
                 kernel_size=3, dropout=0.2):
        super().__init__()
        self.tcn = TemporalConvNet(num_inputs=input_size, num_channels=num_channels, 
                                   kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1) # (batch, features, seq_len)
        y = self.tcn(x)
        y = y[:, :, -1] # take last timestep
        return self.linear(y)
    
class PyTorchTCN(BaseEstimator, ClassifierMixin):
    """scikit-learn compatible TCN wrapper"""

    def __init__(self, sequence_length=48, num_channels=(64,128,64),
                 kernel_size=3, dropout=0.2, epochs=50, batch_size=32,
                 learning_rate=0.001, device=None, verbose=False):
        
        ## best practice based on optuna with 50 trials on tcn model
        ## {
        # "sequence_length": 36,
        # "num_channels": [16, 32],
        # "kernel_size": 2,
        # "dropout": 0.35039058736129347,
        # "learning_rate": 0.001966625789577707,
        # "batch_size": 32,
        # "epochs": 38
        # }

        self.sequence_length = sequence_length
        self.num_channels = num_channels
        self.kernel_size  = kernel_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.model = None
        self.input_size_ = None
        self.label_encoder_ = LabelEncoder()

    def _create_sequences(self, X):
        """Create sliding window sequences"""
        X_seq = []
        for i in range(len(X) - self.sequence_length + 1):
            X_seq.append(X[i:i+self.sequence_length])
        return np.array(X_seq, dtype=np.float32)
    
    def fit(self, X, y):

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        y_encoded = self.label_encoder_.fit_transform(y)

        self.input_size_ = X.shape[1]
        print("DEBUG input size:", self.input_size_)

        # create sequences
        X_seq = self._create_sequences(X)
        y_seq = y_encoded[self.sequence_length-1:]

        # convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.LongTensor(y_seq).to(self.device)

        # initialize model
        num_classes = len(np.unique(y_encoded))

        self.model = TCNClassifier(
            input_size= self.input_size_,
            num_classes=num_classes,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout
        ).to(self.device)

        # training
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        from sklearn.utils.class_weight import compute_class_weight
        
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_encoded),
            y=y_encoded
            )
        # added for third training to use softer weights
        weights = np.sqrt(weights)

        weights = torch.FloatTensor(weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=weights)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if self.verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{self.epochs}, loss: {avg_loss:.4f}")
        
        from sklearn.metrics import classification_report, f1_score

        self.model.eval()

        with torch.no_grad():
            train_outputs = self.model(X_tensor)
            train_preds   = torch.argmax(train_outputs, dim=1).cpu().numpy()
        
        train_f1_macro = f1_score(
            y_seq, train_preds, average="macro"
        )

        train_f1_weighted = f1_score(
            y_seq, train_preds, average="weighted"
        )

        print("\nTrain Results")
        print(f"train macro F1: {train_f1_macro:.4f}")
        print(f"train weighted F1: {train_f1_weighted:.4f}")

        print(
            classification_report(
                y_seq, 
                train_preds, 
                #target_names=self.label_encoder_.classes_
                target_names=[str(c) for c in self.label_encoder_.classes_]
            )
        )
        return self
    
    def predict(self, X):
        probabilities = self.predict_proba(X)
        encoded_preds = np.argmax(probabilities, axis=1)
        return self.label_encoder_.inverse_transform(encoded_preds)
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)

        X_seq = self._create_sequences(X)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        probabilities = probabilities.cpu().numpy()

        fisrt_proba = probabilities[0]
        padded_probas = [fisrt_proba] * (self.sequence_length-1)
        padded_probas += probabilities.tolist()

        return np.array(padded_probas)