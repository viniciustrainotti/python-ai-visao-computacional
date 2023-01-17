import csv
import random
from sklearn.feature_extraction import FeatureHasher
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_data(file_name):
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile))
        header = data[0]
        rows = data[1:]
        random.shuffle(rows)
        X = [row[:-1] for row in rows]
        y = [row[-1] for row in rows]
        return X, y, header

def extract_features(X):
    feature_hasher = FeatureHasher(n_features=10)
    X_transformed = feature_hasher.transform(X)
    return X_transformed

def train_and_evaluate(X_train, y_train, X_test, y_test):
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVC()
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    return results

if __name__ == '__main__':
    file_name = 'data.csv'
    X, y, header = load_data(file_name)
    X = extract_features(X)
    X_train, X
