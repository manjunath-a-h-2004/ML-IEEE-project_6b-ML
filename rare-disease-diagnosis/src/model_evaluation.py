from sklearn.metrics import accuracy_score, precision_score, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance and return accuracy, precision, and confusion matrix."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, precision, conf_matrix

def print_evaluation_results(model_name, accuracy, precision, conf_matrix):
    print(f"\nModel: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print("Confusion Matrix:")
    print(conf_matrix)
