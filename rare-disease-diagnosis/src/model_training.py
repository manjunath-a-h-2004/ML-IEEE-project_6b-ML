from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def train_ann(X_train, y_train):
    """Train Artificial Neural Network (ANN) model."""
    ann = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
    ann.fit(X_train, y_train)
    return ann

def train_naive_bayes(X_train, y_train):
    """Train Naive Bayes model."""
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    return nb

def train_cart(X_train, y_train):
    """Train Classification and Regression Tree (CART) model."""
    cart = DecisionTreeClassifier(max_depth=5, random_state=42)
    cart.fit(X_train, y_train)
    return cart

def train_wknn(X_train, y_train):
    """Train Weighted K-Nearest Neighbors (W-KNN) model."""
    wknn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    wknn.fit(X_train, y_train)
    return wknn
