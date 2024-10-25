import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from KNNModel import KNNModel
from joblib import dump

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Function to plot accuracy graph
def plotAccuracyGraph(trainAccuracies, valAccuracies, epoch):
    plt.plot(range(1, epoch + 1), trainAccuracies, 'bo-', label='Training Accuracy')
    plt.plot(range(1, epoch + 1), valAccuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


# Load data
#data = pd.read_excel("../featureExtraction/alphabet_data.xlsx", header=0)
data = pd.read_excel("../featureExtraction/numbers_data.xlsx", header=0)
data.pop("CHARACTER")

groupValue, coordinates = data.pop("GROUPVALUE"), data.copy()
coordinates = np.reshape(coordinates.values, (coordinates.shape[0], 63))
groupValue = groupValue.to_numpy()

k_folds = 4
foldTrainAccuracies = []
foldValAccuracies = []
foldTrainPrecision = []
foldValPrecision = []
foldTrainRecall = []
foldValRecall = []
foldTrainF1 = []
foldValF1 = []

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

for fold, (trainIndex, valIndex) in enumerate(kf.split(coordinates)):
    print(f"Training on fold {fold + 1}/{k_folds}")

    # Splitting data for the current fold
    training = coordinates[trainIndex]
    groupValueTraining = groupValue[trainIndex]

    validation = coordinates[valIndex]
    groupValueValidation = groupValue[valIndex]

    # Model setup
    model = KNNModel(n_neighbors=5)

    # Training
    model.train(training, groupValueTraining)

    # Predictions
    trainPredictions = model.predict(training)
    valPredictions = model.predict(validation)

    # Calculating accuracies
    trainAccuracy = accuracy_score(groupValueTraining, trainPredictions)
    valAccuracy = accuracy_score(groupValueValidation, valPredictions)
    valPrecision = precision_score(groupValueValidation, valPredictions, average='macro')
    valRecall = recall_score(groupValueValidation, valPredictions, average='macro')
    valF1Score = f1_score(groupValueValidation, valPredictions, average='macro')
    # Calculating metrics for training data
    trainPrecision = precision_score(groupValueTraining, trainPredictions, average='macro')
    trainRecall = recall_score(groupValueTraining, trainPredictions, average='macro')
    trainF1Score = f1_score(groupValueTraining, trainPredictions, average='macro')

    # Store metrics for each fold
    foldTrainPrecision.append(trainPrecision)
    foldValPrecision.append(valPrecision)
    foldTrainRecall.append(trainRecall)
    foldValRecall.append(valRecall)
    foldTrainF1.append(trainF1Score)
    foldValF1.append(valF1Score)

    print(f'Fold: {fold + 1}, Validation Precision: {valPrecision}, Validation Recall: {valRecall}, Validation '
          f'F1-Score: {valF1Score}')

    foldTrainAccuracies.append(trainAccuracy)
    foldValAccuracies.append(valAccuracy)

# Calculate and print average metrics
avgTrainAccuracy = np.mean(foldTrainAccuracies)
avgValAccuracy = np.mean(foldValAccuracies)
avgTrainPrec = np.mean(foldTrainPrecision)
avgValPrec = np.mean(foldValPrecision)
avgTrainRec = np.mean(foldTrainRecall)
avgValRec = np.mean(foldValRecall)
avgTrainF1 = np.mean(foldTrainF1)
avgValF1 = np.mean(foldValF1)

# Print final average accuracy
print(f"Final Average Training Accuracy: {avgTrainAccuracy * 100:.2f}%")
print(f"Final Average Validation Accuracy: {avgValAccuracy * 100:.2f}%")
print(f"Final Average Training Precision: {avgTrainPrec * 100:.2f}%")
print(f"Final Average Validation Precision: {avgValPrec * 100:.2f}%")
print(f"Final Average Training Recall: {avgTrainRec * 100:.2f}%")
print(f"Final Average Validation Recall: {avgValRec * 100:.2f}%")
print(f"Final Average Training F1-Score: {avgTrainF1 * 100:.2f}%")
print(f"Final Average Validation F1-Score: {avgValF1 * 100:.2f}%")


# Save the model
#modelPath = "KNN_model_alphabet_SIBI.pth"
#modelPath = "KNN_model_number_SIBI.pth"
#dump(model, modelPath)
