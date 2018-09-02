'''
https://blog.csdn.net/duinodu/article/details/76618638
https://radimrehurek.com/gensim/similarities/docsim.html
涉及算法：tfidf
步骤
    1.读取文档
    2.对要计算的文档进行分词
    3.对文档进行整理成指定格式，方便后续进行计算
    4.计算出词语的频率
    5.[可选]UI频率低的词语进行过滤
    6.通过语料库建立词典
    7.加载要对比的文档
    8.将要对比的文档通过doc2bow转化为稀疏向量
    9.对稀疏向量进行进一步处理得到新的语料库
    10.对新语料库通过tfidf进行处理，得到tfidf
    11.得到token2id的特征数
    12.稀疏矩阵的相似度，从而建立索引
    13.得到最终相似度结果
'''
# 语料库 模型和相似度
from gensim import corpora, models, similarities
import jieba
from collections import defaultdict


# sample和trainging两个都是训练集
# test是测试集
sample = 'sample.txt'
training = "training.txt"
# 读取这两个样本集文档
sampleContent = open(sample, "rb").read().decode("utf-8", 'ignore')
trainingContent = open(training, "rb").read().decode("utf-8", 'ignore')
# 对文档进行分词操作
simpleList = jieba.cut(sampleContent)
trainingList = jieba.cut(trainingContent)

# 在分词后对词语进行处理，方便后面操作
# 单词1 单词2 单词3 ...单词n
simpleWords = ""
for word in simpleList:
    simpleWords += word + " "
# print(simpleWords)

trainingWords = ""
for word in trainingList:
    trainingWords += word + " "
# print(trainingWords)
# 将处理后的词语存放在document的list中
documents = [simpleWords, trainingWords]
texts = [[word for word in document.split()]
         for document in documents]

# 计算分词后出现词语的个数（词频）（仅在筛选时有用）
frequence = defaultdict(int)
for text in texts:
    for token in text:
        frequence[token] += 1
# print(frequence)
# 频率低的词语进行过滤
texts = [[word for word in text if frequence[token] > 3]
         for text in texts]
print(texts)
# 通过语料库建立词典
dictionary = corpora.Dictionary(texts)
dictionary.save("dict.txt")
# 对测试集进行读取
doc = "test.txt"
testContent3 = open(doc, 'rb').read().decode("utf-8", 'ignore')
# 对测试集进行分词
testCut = jieba.cut(testContent3)
testDictStr = ""
for item in testCut:
    testDictStr += item + " "
# 将样本集转换为稀疏向量
new_vec = dictionary.doc2bow(testDictStr.split())
# 将训练集转换成稀疏向量
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
featureNum = len(dictionary.token2id.keys())
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featureNum)
similary = index[tfidf[new_vec]]
string_tfidf = tfidf[new_vec]
print(similary)
print(new_vec)
print(string_tfidf)
