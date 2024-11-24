# 导入所需的库
import numpy as np 
import pandas as pd
import os
import torch
import random
import gc
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, Dataset, random_split

# 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载训练和测试数据
train = pd.read_csv('E:/shuju/train.tsv', sep='\t')
test = pd.read_csv('E:/shuju/test.tsv', sep='\t')

# 使用TF-IDF对文本数据进行向量化
def tfidf_vectorization(train_df, test_df):
    vectorizer = TfidfVectorizer(max_features=5000)  # 调整max_features以适应需求
    X_train = vectorizer.fit_transform(train_df['Phrase'].astype(str)).toarray()  # 将训练集文本转换为TF-IDF向量
    X_test = vectorizer.transform(test_df['Phrase'].astype(str)).toarray()  # 将测试集文本转换为TF-IDF向量
    
    return X_train, X_test, vectorizer

X_train, X_test, vectorizer = tfidf_vectorization(train, test)

# 将TF-IDF数据转换为PyTorch张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

# 创建自定义数据集类
class PhraseDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features  # 特征数据
        self.labels = labels  # 标签数据
        
    def __len__(self):
        return len(self.features)  # 返回数据集的长度
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]  # 返回特征和标签
        else:
            return self.features[idx]  # 只返回特征

# 准备数据集
train_labels = train['Sentiment'].values  # 获取训练集标签
train_dataset = PhraseDataset(X_train_tensor, train_labels)  # 创建训练数据集

# 将数据集拆分为训练集和验证集
train_size = int(0.8 * len(train_dataset))  # 计算训练集大小
val_size = len(train_dataset) - train_size  # 计算验证集大小
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])  # 拆分数据集
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)  # 创建训练数据加载器
val_loader = DataLoader(val_subset, batch_size=32)  # 创建验证数据加载器

# 定义情感分类模型
class SentimentNN(nn.Module):
    def __init__(self, input_dim, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # 第一个全连接层
        self.fc2 = nn.Linear(64, 32)  # 第二个全连接层
        self.fc3 = nn.Linear(32, output_size)  # 输出层
        self.dropout = nn.Dropout(0.5)  # Dropout层，防止过拟合

    def forward(self, x):
        x = F.relu(self.fc1(x))  # 应用ReLU激活函数
        x = self.dropout(x)  # 应用Dropout
        x = F.relu(self.fc2(x))  # 应用ReLU激活函数
        x = self.dropout(x)  # 应用Dropout
        x = self.fc3(x)  # 输出层
        return x

# 设置训练参数
input_dim = X_train.shape[1]  # 输入维度
output_size = 5  # 输出类别数
net = SentimentNN(input_dim, output_size).to(device)  # 创建模型并移动到指定设备
net.train()  # 设置模型为训练模式

# 设置超参数
epochs = 100  # 训练轮数
lr = 0.001  # 学习率

optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # 优化器
criterion = nn.CrossEntropyLoss()  # 损失函数

# 早停设置
best_val_acc = 0  # 最佳验证准确率
patience = 10  # 早停耐心值
counter = 0  # 早停计数器

# 训练循环
for e in range(epochs):
    net.train()  # 设置模型为训练模式
    running_loss = 0.0  # 记录训练损失
    running_acc = 0.0  # 记录训练准确率

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移动到指定设备

        optimizer.zero_grad()  # 清零梯度
        
        output = net(inputs)  # 前向传播

        loss = criterion(output, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()  # 累加损失
        running_acc += (output.argmax(dim=1) == labels).float().mean()  # 累加准确率

    print(f"Epoch {e + 1}/{epochs}, Loss: {running_loss / len(train_loader):.6f}, Acc: {running_acc / len(train_loader):.6f}")

    # 验证阶段
    net.eval()  # 设置模型为评估模式
    val_loss = 0.0  # 记录验证损失
    val_acc = 0.0  # 记录验证准确率
    with torch.no_grad():  # 关闭梯度计算
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)  # 将数据移动到指定设备
            val_output = net(val_inputs)  # 前向传播
            val_loss += criterion(val_output, val_labels).item()  # 累加损失
            val_acc += (val_output.argmax(dim=1) == val_labels).float().mean().item()  # 累加准确率

    val_acc /= len(val_loader)  # 计算平均验证准确率
    print(f"Validation Loss: {val_loss / len(val_loader):.6f}, Validation Accuracy: {val_acc:.6f}")

    # 早停检查
    if val_acc > best_val_acc:
        best_val_acc = val_acc  # 更新最佳验证准确率
        counter = 0  # 重置早停计数器
    else:
        counter += 1  # 增加早停计数器
        if counter >= patience:
            print("Early stopping triggered.")  # 触发早停
            break

# 测试集预测
net.eval()  # 设置模型为评估模式
test_predictions = []  # 存储测试集预测结果
with torch.no_grad():  # 关闭梯度计算
    test_loader = DataLoader(PhraseDataset(X_test_tensor), batch_size=32)  # 创建测试数据加载器
    for test_inputs in test_loader:
        test_inputs = test_inputs.to(device)  # 将数据移动到指定设备
        test_output = net(test_inputs)  # 前向传播
        test_predictions.extend(test_output.argmax(dim=1).cpu().numpy())  # 存储预测结果

# 创建输出DataFrame，包含PhraseId和Sentiment
output_df = pd.DataFrame({
    'PhraseId': test['PhraseId'],  # 确保'PhraseId'对应于测试集中的正确列
    'Sentiment': test_predictions
})

# 保存预测结果到CSV文件
output_path = 'E:/shuju/answer/predictions.csv'

# 如果输出目录不存在，则创建
output_dir = os.path.dirname(output_path)
os.makedirs(output_dir, exist_ok=True)

# 保存预测结果
output_df.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")