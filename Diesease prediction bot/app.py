import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from keras.models import Sequential
from keras.layers import Dense

# Load dataset
train_df = pd.read_csv('Training.csv')
test_df = pd.read_csv('Testing.csv')

# Drop unnamed columns if exist
train_df = train_df.loc[:, ~train_df.columns.str.contains('^Unnamed')]
test_df = test_df.loc[:, ~test_df.columns.str.contains('^Unnamed')]

# Features & labels
X = train_df.drop('prognosis', axis=1)
y = train_df['prognosis']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Naive Bayes
nb = MultinomialNB()
nb.fit(X_train, y_train)
nb_preds = nb.predict(X_val)
nb_acc = accuracy_score(y_val, nb_preds)
nb_f1 = f1_score(y_val, nb_preds, average='weighted')
print("Naive Bayes Accuracy:", nb_acc)
print("Naive Bayes F1 Score:", nb_f1)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_preds = dt.predict(X_val)
dt_acc = accuracy_score(y_val, dt_preds)
dt_f1 = f1_score(y_val, dt_preds, average='weighted')
print("Decision Tree Accuracy:", dt_acc)
print("Decision Tree F1 Score:", dt_f1)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_val)
knn_acc = accuracy_score(y_val, knn_preds)
knn_f1 = f1_score(y_val, knn_preds, average='weighted')
print("KNN Accuracy:", knn_acc)
print("KNN F1 Score:", knn_f1)

# ANN Model
model = Sequential()
model.add(Dense(128, input_dim=X.shape[1], activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(len(np.unique(y_encoded)), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

# Evaluate ANN
ann_score = model.evaluate(X_val, y_val)
ann_acc = ann_score[1]
ann_preds = model.predict(X_val)
ann_preds = np.argmax(ann_preds, axis=1)
ann_f1 = f1_score(y_val, ann_preds, average='weighted')
print("ANN Accuracy:", ann_acc)
print("ANN F1 Score:", ann_f1)

# Graph 1: Bar Chart for Accuracy Comparison
models = ['Naive Bayes', 'Decision Tree', 'KNN', 'ANN']
accuracies = [nb_acc, dt_acc, knn_acc, ann_acc]

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

# Graph 2: F1 Score Comparison
f1_scores = [nb_f1, dt_f1, knn_f1, ann_f1]

plt.figure(figsize=(8, 5))
plt.bar(models, f1_scores, color=['blue', 'green', 'orange', 'purple'])
plt.title('Model F1 Score Comparison')
plt.ylabel('F1 Score')
plt.ylim(0, 1)
plt.show()

# Graph 3: Box Plot for Model Accuracy Comparison
accuracy_data = [nb_acc, dt_acc, knn_acc, ann_acc]

plt.figure(figsize=(8, 5))
plt.boxplot(accuracy_data, labels=models, patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Model Accuracy Box Plot')
plt.ylabel('Accuracy')
plt.show()

# Symptom Checker Bot
print("\nWelcome to Disease Symptom Checker Bot!")
print("Available Symptoms:\n", list(X.columns))

# Input symptoms
input_symptoms = input("\nEnter symptoms separated by commas:\nSymptoms: ").lower().replace(" ", "_").split(',')

# Build input vector
all_symptoms = list(X.columns)
input_vector = [1 if symptom in input_symptoms else 0 for symptom in all_symptoms]
input_vector = np.array(input_vector).reshape(1, -1)

# Predict with models
nb_pred = le.inverse_transform(nb.predict(input_vector))[0]
dt_pred = le.inverse_transform(dt.predict(input_vector))[0]
knn_pred = le.inverse_transform(knn.predict(input_vector))[0]
ann_pred = le.inverse_transform([np.argmax(model.predict(input_vector))])[0]

print(f"\nPredictions based on your symptoms:")
print(f"Naive Bayes: {nb_pred}")
print(f"Decision Tree: {dt_pred}")
print(f"KNN: {knn_pred}")
print(f"ANN: {ann_pred}")

# Graph 4: Pie Chart for Predictions
predictions = [nb_pred, dt_pred, knn_pred, ann_pred]
labels, counts = np.unique(predictions, return_counts=True)

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['cyan', 'magenta', 'yellow', 'lightgreen'])
plt.title('Disease Prediction Votes')
plt.show()
