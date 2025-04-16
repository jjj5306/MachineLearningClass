import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import seaborn as sns

# MNIST dataset fetch
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist["data"], mnist["target"]

# data type conversion
X = X.astype('float32')
y = y.astype('int')

# train and test data split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# SVM model creation and training
print("SVM model training.")
# kernel: rbf, C: 5, gamma: scale
svm = SVC(kernel='rbf', C=5, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# prediction
print("Predicting.")
y_pred = svm.predict(X_test)

# performance evaluation
print("\n===== Model performance evaluation =====")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# confusion matrix visualization
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

