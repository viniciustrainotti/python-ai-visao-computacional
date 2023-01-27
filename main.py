import csv
import sys
import os
import joblib
import cv2
import numpy as np
import random
from sklearn.feature_extraction import FeatureHasher
from skimage.feature import greycomatrix, greycoprops
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def load_data(folder_name):
    X = []
    y = []
    for class_label in os.listdir(folder_name):
        class_folder = os.path.join(folder_name, class_label)
        if os.path.isdir(class_folder):
            for image_path in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_path)
                image = cv2.imread(image_path)
                features = extract_features_opencv(image)
                X.append(features)
                y.append(class_label)
    return X, y

def extract_features_opencv(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to segment the teeth and cavities
    ret, thresholded_image = cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
    # Extract shape and texture features from the thresholded image
    glcm = greycomatrix(thresholded_image, [1], [0], levels=256)
    shape_features = cv2.HuMoments(cv2.moments(thresholded_image)).flatten()
    texture_features = [greycoprops(glcm, prop) for prop in ('contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy')]
    # Concatenate the shape and texture features
    texture_features = np.ravel(texture_features)
    features = np.concatenate((shape_features, texture_features))
    return features

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
    file_name = 'banco_dentes_IAA'
    X, y = load_data(file_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8, random_state=42)
    results = train_and_evaluate(X_train, y_train, X_test, y_test)
    best_clf = max(results, key=lambda x: results[x]['accuracy'])
    joblib.dump(best_clf, 'best_model.pkl')
    for classifier, metrics in results.items():
        print(f'{classifier} Results:')
        print(f'Accuracy: {metrics["accuracy"]}')
        print(f'Precision: {metrics["precision"]}')
        print(f'Recall: {metrics["recall"]}')
        print(f'F1 Score: {metrics["f1"]}')
