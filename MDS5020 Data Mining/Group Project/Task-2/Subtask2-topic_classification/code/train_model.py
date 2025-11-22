# train_model.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score
import jieba
import re
import joblib

# 数据预处理和清洗 
class NewsTitlePreprocessor(BaseEstimator, TransformerMixin):
    """专门针对新闻标题的预处理"""
    
    def __init__(self):
        self.stop_words = {'公司', '股份', '有限', '责任', '集团', '关于', '公告', '事项'}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return [self._preprocess_text(text) for text in X]
    
    def _preprocess_text(self, text):
        if pd.isna(text):
            return ""
        
        cleaned_text = re.sub(r'^[^:]*:', '', str(text))
        
        # 常见公司名称模式
        company_patterns = [
            r'^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?:',  # 中文公司名+冒号
            r'^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?公告:',  # 公司名+公告+冒号
            r'^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?关于',  # 公司名+关于
        ]
        
        #去掉公司名称
        for pattern in company_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text)
        
        # 去除年份和数字
        cleaned_text = re.sub(r'\d{4}年|\d{1,2}月|\d{1,2}日|\d+', ' ', cleaned_text)
        
        # 去除特殊字符但保留中文
        cleaned_text = re.sub(r'[^\u4e00-\u9fa5]', ' ', cleaned_text)
        
        # 分词
        words = jieba.cut(cleaned_text)
        
        # 过滤
        result = []
        for word in words:
            if len(word) > 1 and word not in self.stop_words:
                result.append(word)
        
        return ' '.join(result)

# 定义主题关键词映射
topic_keywords = {
    '上市保荐书': ['上市保荐书', '上市保荐'],
    '独立董事候选人声明': ['独立董事候选人声明', '独立董事候选人'],
    '独立董事述职报告': ['独立董事述职报告', '独立董事述职'],
    '保荐/核查意见': ['保荐', '核查', '核查意见', '保荐意见', '专项报告'],
    '分配预案': ['分配预案', '利润分配预案'],
    '公司章程': ['公司章程', '章程'],
    '半年度报告全文': ['半年度报告'],
    '高管人员任职变动': ['任职', '高管', '任职变动', '辞职', '聘任', '选举', '监事', '董事'],
    '分配方案实施': ['分配实施', '权益分派实施', '分红派息实施'],
    '独立董事提名人声明': ['独立董事提名人声明', '独立董事提名人'],
    '公司章程修订': ['公司章程修订', '章程修订'],
    '诉讼仲裁': ['诉讼', '仲裁'],
    '年度报告全文': ['年度报告'],
    '关联交易': ['关联交易'],
    '年度报告摘要': ['年度报告摘要'],
    '分配方案决议公告': ['分配方案', '分配方案决议'],
    '股东大会决议公告': ['股东大会决议'],
    '发行保荐书': ['发行保荐书', '发行保荐']
}

def train_news_topic_classifier(df):
    """
    训练新闻主题分类器
    """
    # 数据预处理 - 使用改进的预处理函数
    print("正在进行数据预处理...")
    print("原始标题示例:")
    for i in range(3):
        print(f"  {df['news_title'].iloc[i]}")
    
    # 准备训练数据
    X = df['news_title']  # 使用原始标题，预处理在Pipeline中完成
    y = df['topic']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 创建 Pipeline（包含TF-IDF特征提取和朴素贝叶斯分类器）
    print("正在创建Pipeline...")
    pipeline = Pipeline([
        ('preprocessor', NewsTitlePreprocessor()),
        ('tfidf', TfidfVectorizer(
            max_features=3000,  # 减少特征数量，避免过拟合
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2)
        )),
        ('classifier', MultinomialNB(alpha=0.5))  # 增加平滑参数
    ])
    
    # TF-IDF特征提取和训练朴素贝叶斯分类器（在Pipeline中完成）
    print("正在提取TF-IDF特征和训练朴素贝叶斯分类器...")
    pipeline.fit(X_train, y_train)
    
    # 预测和评估
    y_pred = pipeline.predict(X_test)
    
    # 输出评估结果
    print("\n=== 分类器性能评估 ===")
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred))
    
    return pipeline, X_test, y_test, y_pred

def predict_topic(pipeline, text):
    """
    预测单个文本的主题
    """
    prediction = pipeline.predict([text])[0]
    probabilities = pipeline.predict_proba([text])[0]
    
    # 获取预处理后的文本
    preprocessor = pipeline.named_steps['preprocessor']
    processed_text = preprocessor.transform([text])[0]
    
    return prediction, probabilities, processed_text

def analyze_keyword_importance(pipeline, topic_labels):
    """
    分析每个主题的关键词重要性
    """
    # 从pipeline中获取vectorizer和classifier
    tfidf_vectorizer = pipeline.named_steps['tfidf']
    classifier = pipeline.named_steps['classifier']
    
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    print("\n=== 各主题关键词重要性分析 ===")
    for i, topic in enumerate(topic_labels):
        # 获取该主题的特征对数概率
        feature_log_prob = classifier.feature_log_prob_[i]
        
        # 获取最重要的15个特征
        top_indices = feature_log_prob.argsort()[-15:][::-1]
        top_features = [(feature_names[idx], np.exp(feature_log_prob[idx])) for idx in top_indices]
        
        print(f"\n{topic} 主题的关键词:")
        for feature, prob in top_features:
            print(f"  {feature}: {prob:.4f}")

# 主函数
def main():
    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel('./training_news-topic.xlsx')
    
    # 检查数据
    print(f"数据形状: {df.shape}")
    print(f"主题分布:\n{df['topic'].value_counts()}")
    
    # 训练分类器
    pipeline, X_test, y_test, y_pred = train_news_topic_classifier(df)
    
    # 分析关键词重要性
    topic_labels = df['topic'].unique()
    analyze_keyword_importance(pipeline, topic_labels)
    
    # 保存模型
    model_data = {
        'pipeline': pipeline,
        'topic_labels': topic_labels,
        'topic_keywords': topic_keywords
    }
    
    joblib.dump(model_data, 'news_topic_model.pkl')
    print("模型已保存为 'news_topic_model.pkl'")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()