import os
import re
import collections
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================
# 1. 基础配置与路径设置（请确认路径是否正确）
# ==========================================
TRAIN_PATH = r"D:\新建文件夹 (2)\train_dataset.csv"
TEST_PATH = r"D:\新建文件夹 (2)\test_dataset.csv"
SUB_PATH = r"D:\新建文件夹 (2)\submission.csv"

# ==========================================
# 2. 读取数据（强制 header=None 模式读取，防止丢失首行）
# ==========================================
def load_data_safe(path, is_train=True):
    print(f"正在读取文件: {path}")
    # 针对无表头英文数据，使用逗号或 \t 分隔读取
    try:
        df = pd.read_csv(path, sep='\t', header=None)
        if len(df.columns) < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=',', header=None)
        
    print(f"-> 成功读取，数据形状: {df.shape}，列数: {len(df.columns)}")
    
    if is_train:
        # 假设：0列为索引/ID，1列为正文，2列为摘要
        if len(df.columns) >= 3:
            df.columns = ['Index', 'Text', 'Abstract']
        else:
            df.columns = ['Text', 'Abstract']
            df['Index'] = range(len(df))
    else:
        # 测试集没有摘要列
        if len(df.columns) >= 2:
            df.columns = ['Index', 'Text']
        else:
            df.columns = ['Text']
            df['Index'] = range(len(df))
            
    df['Text'] = df['Text'].fillna('').astype(str)
    if is_train:
        df['Abstract'] = df['Abstract'].fillna('').astype(str)
        
    return df

print("--- 开始加载并解析数据集 ---")
train_df = load_data_safe(TRAIN_PATH, is_train=True)
test_df = load_data_safe(TEST_PATH, is_train=False)

# ==========================================
# 3. 英文分句与文本处理函数
# ==========================================
def split_sentences_en(text):
    """针对英文的分句正则"""
    if not isinstance(text, str) or not text.strip():
        return []
    text = re.sub(r'\s+', ' ', text)  # 合并多余空格
    # 根据英文句号、问号、感叹号分句
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s*', text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]

def get_word_overlap_f1(sent, abstract):
    """计算单句与参考摘要的词级重合度，用于生成训练标签"""
    words_sent = set(re.findall(r'\w+', sent.lower()))
    words_abs = set(re.findall(r'\w+', abstract.lower()))
    if not words_sent or not words_abs:
        return 0.0
    intersection = words_sent.intersection(words_abs)
    if not intersection:
        return 0.0
    precision = len(intersection) / len(words_sent)
    recall = len(intersection) / len(words_abs)
    return 2 * precision * recall / (precision + recall)

def get_jaccard_sim(str1, str2):
    """Jaccard 相似度，用于剔除冗余句"""
    a = set(re.findall(r'\w+', str1.lower()))
    b = set(re.findall(r'\w+', str2.lower()))
    if not a or not b:
        return 0.0
    return len(a.intersection(b)) / len(a.union(b))

# ==========================================
# 4. 提取句子并建立全局 TF-IDF
# ==========================================
print("\n正在进行分句处理...")
train_docs = []
for _, row in train_df.iterrows():
    sents = split_sentences_en(row['Text'])
    if len(sents) > 0:
        train_docs.append({'Index': row['Index'], 'sents': sents, 'Abstract': row['Abstract']})

test_docs = []
for _, row in test_df.iterrows():
    sents = split_sentences_en(row['Text'])
    if len(sents) == 0:
        sents = ["Placeholder sentence for empty text."]
    test_docs.append({'Index': row['Index'], 'sents': sents})

# 收集所有句子用以拟合词频向量
all_sentences = []
for doc in train_docs:
    all_sentences.extend(doc['sents'])
for doc in test_docs:
    all_sentences.extend(doc['sents'])

print("正在构建全局英文词级 TF-IDF 特征...")
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf.fit(all_sentences)

# ==========================================
# 5. 特征工程（已修复 np.matrix 报错）
# ==========================================
def extract_features(docs, is_train=True):
    features_list = []
    labels_list = []
    meta_list = []

    for doc in docs:
        sents = doc['sents']
        M = len(sents)
        
        if is_train:
            abstract = doc['Abstract']
            sims = [get_word_overlap_f1(s, abstract) for s in sents]
            top_indices = np.argsort(sims)[-3:] if M >= 3 else np.arange(M)
            labels = np.zeros(M)
            for idx in top_indices:
                if sims[idx] > 0.05:
                    labels[idx] = 1
        
        sent_vecs = tfidf.transform(sents)
        
        # 【核心修复点】将稀疏矩阵求平均得到的 matrix 强转为普通的 numpy.ndarray
        doc_vec = np.asarray(sent_vecs.mean(axis=0))
        
        tfidf_sums = np.array(sent_vecs.sum(axis=1)).flatten()
        
        # 计算句子与文章的余弦相似度
        if doc_vec.sum() == 0:
            cos_sims = np.zeros(M)
        else:
            # 同样确保传入的 doc_vec 是标准 ndarray
            cos_sims = cosine_similarity(sent_vecs, doc_vec).flatten()
            
        for i in range(M):
            num_words = len(sents[i].split())
            digit_count = sum(1 for c in sents[i] if c.isdigit())
            
            feat = [
                i,                          # 句子绝对位置
                i / M,                      # 相对位置
                1 if i == 0 else 0,         # 是否首句
                1 if i == 1 else 0,         # 是否第二句
                1 if i == M - 1 else 0,     # 是否尾句
                num_words,                  # 句子单词数
                tfidf_sums[i],              # TF-IDF 权重和
                cos_sims[i],                # 与整篇文档的主题相似度
                digit_count                 # 数字字符个数
            ]
            features_list.append(feat)
            if is_train:
                labels_list.append(labels[i])
            else:
                meta_list.append((doc['Index'], i, sents[i]))
                
    features_df = pd.DataFrame(features_list, columns=[
        'pos_abs', 'pos_rel', 'is_first', 'is_second', 'is_last',
        'length_word', 'tfidf_sum', 'doc_similarity', 'digit_count'
    ])
    
    if is_train:
        return features_df, np.array(labels_list)
    else:
        return features_df, meta_list

print("正在构建训练集特征...")
X_train, y_train = extract_features(train_docs, is_train=True)
print("正在构建测试集特征...")
X_test, test_meta = extract_features(test_docs, is_train=False)

# ==========================================
# 6. 模型训练 (LightGBM)
# ==========================================
print(f"\n训练集样本数: {X_train.shape[0]}, 特征数: {X_train.shape[1]}")
print("开始训练 LightGBM 抽取分类器...")
model = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
model.fit(X_train, y_train)

# ==========================================
# 7. 摘要抽取、去冗余与重构
# ==========================================
print("\n正在生成测试集摘要...")
preds = model.predict_proba(X_test)[:, 1]

doc_predictions = collections.defaultdict(list)
for idx, (doc_id, sent_idx, sent_text) in enumerate(test_meta):
    doc_predictions[doc_id].append((sent_idx, preds[idx], sent_text))

results = []
for doc_id, sents_info in doc_predictions.items():
    # 按照概率得分降序排序
    sents_info.sort(key=lambda x: x[1], reverse=True)
    
    selected_sents = []
    for sent_idx, prob, sent_text in sents_info:
        # 去冗余过滤
        is_redundant = False
        for _, _, selected_text in selected_sents:
            if get_jaccard_sim(sent_text, selected_text) > 0.45:
                is_redundant = True
                break
        
        if not is_redundant:
            selected_sents.append((sent_idx, prob, sent_text))
            
        if len(selected_sents) >= 3:
            break
            
    # 还原成在原文中出现的先后顺序（保证可读性）
    selected_sents.sort(key=lambda x: x[0])
    
    # 用空格拼接英文摘要句
    summary = " ".join([item[2] for item in selected_sents])
    results.append((doc_id, summary))

# ==========================================
# 8. 保存提交结果
# ==========================================
sub_df = pd.DataFrame(results, columns=['Index', 'Target'])
sub_df = sub_df.sort_values(by='Index')

# 按照官方要求：使用 \t 分隔符，不保留行索引
sub_df.to_csv(SUB_PATH, sep='\t', index=False)
print(f"\n运行完成！结果已存入：{SUB_PATH}")
