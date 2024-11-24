
# import pandas as pd
# import numpy as np
# import time
# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt
# from joblib import dump

# # 加载数据
# def load_data(train_path, test_path):
#     train_data = pd.read_csv(train_path, sep='\t')
#     test_data = pd.read_csv(test_path, sep='\t')
    
#     # 数据清洗
#     train_data['Phrase'].fillna("", inplace=True)
#     test_data['Phrase'].fillna("", inplace=True)
#     train_data['Phrase'] = train_data['Phrase'].astype(str)
#     test_data['Phrase'] = test_data['Phrase'].astype(str)
    
#     return train_data, test_data

# # 特征提取
# def extract_features(train_texts, test_texts, max_features=5000):
#     tfidf = TfidfVectorizer(max_features=max_features)
#     X_train_tfidf = tfidf.fit_transform(train_texts)
#     X_test_tfidf = tfidf.transform(test_texts)
#     return X_train_tfidf, X_test_tfidf, tfidf

# # 模型训练与 K 折交叉验证
# def train_and_evaluate(X, y, k=5):
#     kf = KFold(n_splits=k, shuffle=True, random_state=42)
#     fold_accuracies = []
#     all_val_true = []
#     all_val_pred = []
    
#     print("开始 K 折交叉验证...")
#     for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
#         print(f"正在处理第 {fold + 1} 折...")
        
#         # 划分训练集和验证集
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
#         # 定义随机森林模型
#         rf_classifier = RandomForestClassifier(
#             n_estimators=200, max_depth=10, random_state=42, n_jobs=-1, class_weight='balanced'
#         )
        
#         # 训练模型
#         rf_classifier.fit(X_train, y_train)
        
#         # 验证集预测
#         y_val_pred = rf_classifier.predict(X_val)
        
#         # 保存每折结果
#         accuracy = accuracy_score(y_val, y_val_pred)
#         fold_accuracies.append(accuracy)
#         all_val_true.extend(y_val)
#         all_val_pred.extend(y_val_pred)
        
#         print(f"第 {fold + 1} 折准确率: {accuracy:.4f}")
        
#         # 混淆矩阵可视化
#         cm = confusion_matrix(y_val, y_val_pred)
#         disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#         disp.plot()
    
#     # 输出交叉验证总体结果
#     print("\nK 折交叉验证完成！")
#     print(f"平均验证集准确率: {np.mean(fold_accuracies):.4f}")
#     print("\n验证集分类报告:")
#     print(classification_report(all_val_true, all_val_pred))
    
#     return rf_classifier, fold_accuracies

# # 在完整训练集上训练模型并预测测试集
# def train_on_full_data_and_predict(model, X_train, y_train, X_test, test_data, output_path="test_predictions.tsv"):
#     model.fit(X_train, y_train)
#     test_predictions = model.predict(X_test)
#     test_data['Sentiment'] = test_predictions
#     test_data.to_csv(output_path, sep='\t', index=False)
#     print(f"测试集预测结果已保存至 {output_path}")

# # 特征重要性分析
# def plot_feature_importance(model, tfidf, top_n=20):
#     feature_importances = model.feature_importances_
#     feature_names = tfidf.get_feature_names_out()
#     indices = np.argsort(feature_importances)[-top_n:]
    
#     plt.figure(figsize=(10, 8))
#     plt.barh(range(len(indices)), feature_importances[indices], align='center')
#     plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
#     plt.xlabel('Feature Importance')
#     plt.title('Top Important Features')
#     plt.show()

# # 超参数优化（可选）
# def optimize_hyperparameters(X, y):
#     param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [10, 20, None],
#         'min_samples_split': [2, 5, 10],
#         'min_samples_leaf': [1, 2, 4]
#     }
#     grid_search = GridSearchCV(
#         estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
#         param_grid=param_grid,
#         cv=3,
#         scoring='accuracy',
#         verbose=2
#     )
#     grid_search.fit(X, y)
#     print("最佳参数组合:", grid_search.best_params_)
#     return grid_search.best_estimator_

# # 主流程
# if __name__ == "__main__":
#     # 加载数据
#     train_data, test_data = load_data("train.tsv", "test.tsv")
#     X_train_full = train_data['Phrase']
#     y_train_full = train_data['Sentiment']
    
#     # 特征提取
#     X_train_full_tfidf, X_test_tfidf, tfidf = extract_features(X_train_full, test_data['Phrase'])
    
#     # 交叉验证
#     start_time = time.time()
#     rf_classifier, fold_accuracies = train_and_evaluate(X_train_full_tfidf, y_train_full)
#     print(f"总用时: {time.time() - start_time:.2f} 秒")
    
#     # 可选：绘制特征重要性
#     plot_feature_importance(rf_classifier, tfidf)
    
#     # 测试集预测
#     train_on_full_data_and_predict(rf_classifier, X_train_full_tfidf, y_train_full, X_test_tfidf, test_data)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import time

# 加载数据
def load_data(train_path, test_path):
    train_data = pd.read_csv(train_path, sep='\t')
    test_data = pd.read_csv(test_path, sep='\t')
    train_data['Phrase'].fillna("", inplace=True)
    test_data['Phrase'].fillna("", inplace=True)
    return train_data, test_data

# 特征提取（改进：增加 bigram 支持）
def extract_features(train_texts, test_texts, max_features=10000):
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts)
    return X_train_tfidf, X_test_tfidf, tfidf

# 超参数优化（网格搜索）
def optimize_hyperparameters(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=3,
        scoring='accuracy',
        verbose=2
    )
    grid_search.fit(X_train, y_train)
    print("最佳参数组合:", grid_search.best_params_)
    return grid_search.best_estimator_

# 模型训练和评估
def train_and_evaluate(X_train, y_train, X_val, y_val, model):
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 验证集预测
    y_val_pred = model.predict(X_val)
    
    # 输出验证结果
    print("验证集准确率:", accuracy_score(y_val, y_val_pred))
    print("分类报告:\n", classification_report(y_val, y_val_pred))
    
    # 混淆矩阵可视化
    cm = confusion_matrix(y_val, y_val_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    
    print(f"训练和验证总耗时: {time.time() - start_time:.2f} 秒")
    return model

# 特征重要性可视化
def plot_feature_importance(model, tfidf, top_n=20):
    feature_importances = model.feature_importances_
    feature_names = tfidf.get_feature_names_out()
    indices = np.argsort(feature_importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(indices)), feature_importances[indices], align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.title('Top Important Features')
    plt.show()

# 主流程
if __name__ == "__main__":
    # 加载数据
    train_data, test_data = load_data("train.tsv", "test.tsv")
    
    # 提取特征
    X_train_full_tfidf, X_test_tfidf, tfidf = extract_features(train_data['Phrase'], test_data['Phrase'])
    y_train_full = train_data['Sentiment']
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full_tfidf, y_train_full, test_size=0.2, random_state=42
    )
    
    # 超参数优化
    print("开始超参数优化...")
    best_rf = optimize_hyperparameters(X_train, y_train)
    
    # 模型训练和验证
    print("\n开始训练和验证...")
    trained_model = train_and_evaluate(X_train, y_train, X_val, y_val, best_rf)
    
    # 测试集预测
    print("\n对测试集进行预测...")
    test_predictions = trained_model.predict(X_test_tfidf)
    test_data['Sentiment'] = test_predictions
    test_data.to_csv("test_predictions.tsv", sep='\t', index=False)
    print("测试集预测结果已保存至 test_predictions.tsv")
    
    # 特征重要性分析
    print("\n绘制特征重要性...")
    plot_feature_importance(trained_model, tfidf)
