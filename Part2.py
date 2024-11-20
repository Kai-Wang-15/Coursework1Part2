# 导入必要的库
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import re
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. 加载数据
file_path = 'bbc-text.csv'
# 检查文件是否存在
if os.path.exists(file_path):
    print("File found!")
    data = pd.read_csv(file_path, encoding='ISO-8859-1')
    print("File loaded successfully.")
else:
    print("File not found!")
    exit()  # 退出程序，防止后续代码执行

# 检查数据
print("Dataset overview:")
print(data.head())
print(data.info())

# 2. 数据预处理
# 清洗文本：移除非字母字符、数字、多余空格，并转换为小写
def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  # 移除非字母字符
    text = re.sub(r'\d+', '', text)  # 移除数字
    text = re.sub(r'\s+', ' ', text)  # 移除多余空格
    return text.lower().strip()

data['cleaned_text'] = data['text'].apply(clean_text)

# 标签编码：将类别转换为数字
categories = data['category'].unique()
category_mapping = {category: idx for idx, category in enumerate(categories)}
data['category_encoded'] = data['category'].map(category_mapping)

# 3. 特征提取
# 使用 TF-IDF 向量化文本
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')  # 自动处理停用词
X = vectorizer.fit_transform(data['cleaned_text'])
y = data['category_encoded']

# 4. 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 模型训练
# 使用逻辑回归模型
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# 6. 模型评估
y_pred = classifier.predict(X_test)

# 准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy}")

# 分类报告
report = classification_report(y_test, y_pred, target_names=categories)
print(report)

# 混淆矩阵
ConfusionMatrixDisplay.from_estimator(classifier, X_test, y_test, display_labels=categories, cmap='Blues')
plt.title("Confusion Matrix")
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
