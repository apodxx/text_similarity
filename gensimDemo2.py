'''
training（盗墓笔记 老九门） testing（鬼吹灯）相似度匹配
'''
import jieba
from collections import defaultdict
from gensim import corpora,models,similarities


# 打开文本
daomu = "novel/盗墓笔记.txt"
laojiu = "novel/老九门.txt"

daomuContent = open(daomu, "rb").read().decode("UTF-8", 'ignore')

laojiuContent = open(laojiu, "rb").read().decode("UTF-8", 'ignore')
daomuWords = jieba.cut(daomuContent)
laojiuWords = jieba.cut(laojiuContent)
daomuList = ""
for word in daomuWords:
    daomuList += word + " "

laojiuList = ""
for word in laojiuWords:
    laojiuList += word + " "

documents = [daomuList, laojiuList]
stoplist = set('我 ， 的 了 ：'.split(' '))
texts = [[word for word in document.split() if word not in stoplist]
         for document in documents]

frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
new_text = [[token for token in text if frequency[token] > 10]
            for text in texts]

dictionary=corpora.Dictionary(new_text)
dictionary.save("dict2.txt")


# 开始对测试集进行操作
testing="novel/鬼吹灯.txt"
testContent=open(testing,'rb').read().decode("utf-8",'ignore')
testWords=jieba.cut(testContent)
testList=""
for word in testWords:
    testList+=word+" "
test_bow=dictionary.doc2bow(testList.split())
train_bow=[dictionary.doc2bow(text) for text in texts]

# 构建模型
tfidf=models.TfidfModel(train_bow)

featureNum = len(dictionary.token2id.keys())
index = similarities.SparseMatrixSimilarity(tfidf[train_bow], num_features=featureNum)
similary = index[tfidf[test_bow]]
string_tfidf = tfidf[test_bow]
print(similary)