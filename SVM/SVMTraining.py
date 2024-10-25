import numpy as np  # for basic operations over arrays
import torch
from joblib import dump
import pandas as pd
from sklearn.model_selection import train_test_split
from SVMModel import SVM
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Updated to_cuda function
def to_cuda(tensor):
    if torch.cuda.is_available():
        if torch.is_tensor(tensor):
            return tensor.to(torch.device('cuda'))
        else:
            return torch.from_numpy(tensor).to(torch.device('cuda'))
    else:
        if torch.is_tensor(tensor):
            return tensor
        else:
            return torch.from_numpy(tensor)

# Load the dataset
#data = pd.read_excel("../featureExtraction/alphabet_data.xlsx", header=0)
data = pd.read_excel("../featureExtraction/numbers_data.xlsx", header=0)
data.pop("CHARACTER")
y, X = data.pop("GROUPVALUE"), data.copy()
X = X.to_numpy()
y = y.to_numpy()
# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# print("Features Shape:", X.shape)
# print("Target Shape:", y.shape)


# Test SVM

svm = SVM(kernel='linear', k=10)

# svm.fit(X_train, y_train, eval_train=True)

# Convert your data to PyTorch tensors
X_train_tensor = to_cuda(torch.from_numpy(X_train).float())
y_train_tensor = to_cuda(torch.from_numpy(y_train).float())

# Now pass these tensors to your fit method
svm.fit(X_train_tensor, y_train_tensor, eval_train=True)


y_pred_val = svm.predict(X_val)

y_pred_train = svm.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)

accuracy_val = accuracy_score(y_val, y_pred_val)
f1_val = f1_score(y_val, y_pred_val, average='weighted', zero_division=1)
precision_val = precision_score(y_val, y_pred_val, average='weighted', zero_division=1)
recall_val = recall_score(y_val, y_pred_val, average='weighted', zero_division=1)

print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
print(f"Prediction Accuracy: {accuracy_val * 100:.2f}%")
print(f"F1 Score: {f1_val:.2f}")
print(f"Precision: {precision_val:.2f}")
print(f"Recall: {recall_val:.2f}")

# Save the model
modelPath = "SVM_model_alphabet_SIBI_tensor.pth"
# modelPath = "SVM_model_number_SIBI_tensor.pth"
dump(svm, modelPath)
