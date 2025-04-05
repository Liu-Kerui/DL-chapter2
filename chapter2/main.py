import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix, f1_score
from skimage.feature import hog
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import os
import cv2
import seaborn as sns
from sklearn.metrics import confusion_matrix


# 1. 选择医疗场景的X-ray肺炎检测数据集
def load_xray_data(data_dir):
    categories = ['NORMAL', 'PNEUMONIA']
    data = []
    labels = []

    for category in categories:
        path = os.path.join(data_dir, category)
        label = categories.index(category)

        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            feature, _ = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
            data.append(feature)
            labels.append(label)

    return np.array(data), np.array(labels)


# 加载数据集
data_dir = "ChestXRay2017/chest_xray/train"
X, y = load_xray_data(data_dir)

# 处理数据不均衡问题
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# PCA降维
pca = PCA(n_components=50)
X = pca.fit_transform(X)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 2. 训练线性分类模型
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.decision_function(X_test) if hasattr(model, 'decision_function') else model.predict_proba(X_test)[:,1]
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{model.__class__.__name__} Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred))

    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 使用热图可视化混淆矩阵
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["NORMAL", "PNEUMONIA"],
                yticklabels=["NORMAL", "PNEUMONIA"])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model.__class__.__name__}')
    plt.show()

    return y_pred, y_score


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),
    "Perceptron": Perceptron(),
    "Softmax Regression": MLPClassifier(hidden_layer_sizes=(), activation='logistic', max_iter=500)
}

results = {}
for name, model in models.items():
    print(f"Training {name}...")
    y_pred, y_score = train_model(model, X_train, y_train, X_test, y_test)
    results[name] = (y_pred, y_score)

# 3. 评估与可视化
fig, ax = plt.subplots()
for name, (y_pred, y_score) in results.items():
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve')
ax.legend()
plt.show()
