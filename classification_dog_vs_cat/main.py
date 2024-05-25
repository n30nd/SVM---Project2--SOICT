# extract dataset
from zipfile import ZipFile

dataset_train = "train.zip"
    
with ZipFile(dataset_train, 'r') as zip:
    zip.extractall()

import os
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import GridSearchCV
import cv2
import seaborn as sns
import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC


folder_path = f"Dataset/"
os.makedirs(folder_path, exist_ok=True)

# define path
confusion_image_path = os.path.join(folder_path, 'confusion matrix.png')
classification_file_path = os.path.join(folder_path, 'classification_report.txt')
model_file_path = os.path.join(folder_path, "svm_model.pkl")

# Path dataset
dataset_dir = "Dataset/"
# train_dir = os.path.join(dataset_dir, "train")
# test_dir = os.path.join(dataset_dir, "test1")
train_dir = "train"
test_dir = "test1"


# load data, preprocessing data, and labeling
# dog = 1, cat = 0
train_images = os.listdir(train_dir)
features = []
labels = []
image_size = (50, 50)

# Proses train images
for image in tqdm(train_images, desc="Processing Train Images"):
    if image[0:3] == 'cat' :
        label = 0
    else :
        label = 1
    image_read = cv2.imread(train_dir+"/"+image)
    image_resized = cv2.resize(image_read, image_size)
    image_normalized = image_resized / 255.0
    image_flatten = image_normalized.flatten()
    features.append(image_flatten)
    labels.append(label)

del train_images

features = np.asarray(features)
labels = np.asarray(labels)

# train test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True, random_state=42)

del features
del labels

# PCA, SVM, & Pipeline
n_components = 0.8
pca = PCA(n_components=n_components)
svm = SVC()
pca = PCA(n_components=n_components, random_state=42)
pipeline = Pipeline([
    ('pca', pca),
    ('svm', svm)
])


# Tham số tốt nhất từ GridSearchCV
best_params = {'pca__n_components': 0.9, 'svm__kernel': 'rbf'}

# Tạo mô hình với các tham số tối ưu
best_pipeline = Pipeline([
    ('pca', PCA(n_components=best_params['pca__n_components'])),
    ('svm', SVC(kernel=best_params['svm__kernel']))
])

# Huấn luyện mô hình với dữ liệu ban đầu
# Giả sử bạn đã có dữ liệu `X_train` và `y_train`
best_pipeline.fit(X_train, y_train)

# Lưu mô hình tốt nhất
model_path = 'best_svm_model.pkl'
joblib.dump(best_pipeline, model_path)
