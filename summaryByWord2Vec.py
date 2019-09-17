##################################################################################################
##################################################################################################
##################################################################################################
### 算法思路如下：
###       1.将句子用Jibe进行分词处理
###       2.利用Word2Vec词向量模型计算计算句子的相似度
###       3.利用类似Google 的PageRank算法迭代计算句子网络的重要程度，并返回前几个句子
###
###
##################################################################################################
##################################################################################################

#encoding=utf-8
import jieba
import math
from string import punctuation
from heapq import nlargest
from itertools import product, count
import gensim
from gensim.models import word2vec
import numpy as np
import re
import codecs
import sentimentlAnalysis
# import types
import time  



# from __future__ import print_function
# from __future__ import unicode_literals
import sys
# print("extract1")
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import sys
sys.path.append("../")
DATA_SWAP = "/data/AI/python/";
DATA_SWAP2 = "/data/AI/python/";
# In[5]:
CURL = "/data/AI/python/"
MAX_SNETENCES_COUNT = 12

# model = word2vec.Word2Vec.load("chinese_model/word2vec_wx")
model = gensim.models.Word2Vec.load(DATA_SWAP+'data/word2vec_wx')
np.seterr(all='warn')
# DATA_SWAP = "/data/AI/python/";

# model


# In[22]:


def cut_sentences(sentence):
    puns = frozenset(u'。！\n§\r')
    tmp = []
    for ch in sentence:
        tmp.append(ch)
        if puns.__contains__(ch):
            if(len(tmp)<7):
                tmp = []
                continue
            line = ''.join(tmp).replace('\n', '').replace('  ', ' ')
            yield line
            tmp = []
    yield ''.join(tmp).replace('\n', '').replace('  ', ' ')

def filter_symbols(sents):
    stopwords = create_stopwords().union(set(['。', '\r','\n', '.', '§']))
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word in stopwords:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


# 句子中的stopwords
def create_stopwords():
    stop_list = set()
    # stop_list =[]
    with open(CURL+'jieba_dict/stopwords.txt','r',encoding = 'utf-8') as sw:
        for line in sw:
            stop_list.add(line.strip('\n'))
    return stop_list





def two_sentences_similarity(sents_1, sents_2):
    '''
    计算两个句子的相似性
    :param sents_1:
    :param sents_2:
    :return:
    '''
    counter = 0
    for sent in sents_1:
        if sent in sents_2:
            counter += 1
    return counter / (math.log(len(sents_1) + len(sents_2)))


def create_graph(word_sent):
    """
    传入句子链表  返回句子之间相似度的图
    :param word_sent:
    :return:
    """
    num = len(word_sent)
    board = [[0.0 for _ in range(num)] for _ in range(num)]

    for i, j in product(range(num), repeat=2):
        if i != j:
            board[i][j] = compute_similarity_by_avg(word_sent[i], word_sent[j])
    return board


def cosine_similarity(vec1, vec2):
    '''
    计算两个向量之间的余弦相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    tx = np.array(vec1)
    ty = np.array(vec2)
    cos1 = np.sum(tx * ty)
    cos21 = np.sqrt(sum(tx ** 2))
    cos22 = np.sqrt(sum(ty ** 2))
    cosine_value = cos1 / float(cos21 * cos22)
    return cosine_value


def compute_similarity_by_avg(sents_1, sents_2):
    '''
    对两个句子求平均词向量
    :param sents_1:
    :param sents_2:
    :return:
    '''
    if len(sents_1) == 0 or len(sents_2) == 0:
        return 0.0
    # vec1 = model[sents_1[0]]
    # for word1 in sents_1[1:]:
    #     vec1 = vec1 + model[word1]

    # vec2 = model[sents_2[0]]
    # for word2 in sents_2[1:]:
    #     vec2 = vec2 + model[word2]

    vec1 = getModel(model,sents_1[0])
    # model[sents_1[0]]
    for word1 in sents_1[1:]:
        vec1 = vec1 + getModel(model,word1)
        # vec1 = vec1 + model[word1]

    # vec2 = model[sents_2[0]]
    vec2 = getModel(model,sents_2[0])
    for word2 in sents_2[1:]:
        # vec2 = vec2 + model[word2]
        vec1 = vec1 + getModel(model,word2)
    
    if isinstance(vec1,(int,float)) and vec1==0:
        return 0.0

    if isinstance(vec2,(int,float)) and vec2==0:
        return 0.0

    if len(vec1) == 0 or len(vec1) == 0:
        return 0.0

    similarity = cosine_similarity(vec1 / len(sents_1), vec2 / len(sents_2))
    return similarity

def  getModel(model,word):
    if(word in model):
        return model[word]
    else:
        return 0

def calculate_score(weight_graph, scores, i):
    """
    计算句子在图中的分数
    :param weight_graph:
    :param scores:
    :param i:
    :return:
    """
    length = len(weight_graph)
    d = 0.85
    added_score = 0.0

    for j in range(length):
        fraction = 0.0
        denominator = 0.0
        # 计算分子
        fraction = weight_graph[j][i] * scores[j]
        # 计算分母
        for k in range(length):
            denominator += weight_graph[j][k]
            if denominator == 0:
                denominator = 1
        added_score += fraction / denominator
    # 算出最终的分数
    weighted_score = (1 - d) + d * added_score
    return weighted_score


def weight_sentences_rank(weight_graph):
    '''
    输入相似度的图（矩阵)
    返回各个句子的分数
    :param weight_graph:
    :return:
    '''
    # 初始分数设置为0.5
    scores = [0.5 for _ in range(len(weight_graph))]
    old_scores = [0.0 for _ in range(len(weight_graph))]

    # 开始迭代
    while different(scores, old_scores):
        for i in range(len(weight_graph)):
            old_scores[i] = scores[i]
        for i in range(len(weight_graph)):
            scores[i] = calculate_score(weight_graph, scores, i)
    return scores


def different(scores, old_scores):
    '''
    判断前后分数有无变化
    :param scores:
    :param old_scores:
    :return:
    '''
    flag = False
    for i in range(len(scores)):
        if math.fabs(scores[i] - old_scores[i]) >= 0.0001:
            flag = True
            break
    return flag




def filter_model(sents):
    _sents = []
    for sentence in sents:
        for word in sentence:
            if word not in model:
                sentence.remove(word)
        if sentence:
            _sents.append(sentence)
    return _sents


def summarize(text, n):
    tokens = cut_sentences(text)
    sentences = []
    sents = []
    keySentIndex =[]
    index = 0
    for sent in tokens:
    # for i in range(len(tokens)) :
        # sent = tokens[i]
        sent = sent.replace(u'  ', ' ')
        sent = sent.replace(u' ', '')
        sent = sent.replace(u'\r', '')
        sent = sent.replace("\u3000", '')

        # u3000 
        sentences.append(sent)
        ret1 = matchKeyPoint(sent)
        if(ret1):
            print(sent)
            print('Key:'+ret1.group())
            keySentIndex.append(index)
        sents.append([word for word in jieba.cut(sent) if word])
        index = index+1

    sents = filter_symbols(sents)
    sents = filter_model(sents)
    graph = create_graph(sents)

    scores = weight_sentences_rank(graph)
    sent_selected = nlargest(n, zip(scores, count()))
    sent_index = []
    for i in range(n):
        if(i>=len(sent_selected)):
            break;
        sent_index.append(sent_selected[i][1])

    for kindex  in keySentIndex:
        if(kindex not in sent_index):
            print("add"+sentences[kindex])
            sent_index.append(kindex)

    # 按照重要程度返回
    # return [sentences[i] for i in sent_index]
    topSentenceBySeq = []
    for i in range(len(sentences)):
        if(i in sent_index):
            ret1 = matchNotValidPoint(sentences[i])
            if(ret1 == None):
                topSentenceBySeq.append(sentences[i])
    if(len(topSentenceBySeq)>=10):
        topSentenceBySeq=topSentenceBySeq[0:10]
    return topSentenceBySeq



def extract(filein,fileout,sa):
    print("extract2")
    fIn = None
    wOut = None
    try:
        fIn = codecs.open(filein, 'r','utf-8')
        wOut = codecs.open(fileout, 'w','utf-8') # 若是'wb'就表示写二进制文件
        all_the_text = fIn.read().replace('\n', '')
        # print(all_the_text.decode('utf-8'))
        ret = summarize(all_the_text,8)
        # print(ret.decode('gbk'))
        for x in ret:
            # print(x.decode('utf-8'))
            x = x.strip()
            x=x.replace('\n', '')
            if(len(x)>=1):
                wOut.write(x+'\r\n')

        if("true"==sa):
            sentimentlAnalysis.predictForFile(all_the_text,wOut)
    finally:
        if fIn:
            fIn.close()
        if wOut:
            wOut.close()


def extractForFileinName(fileName,sa):
    fileIn = DATA_SWAP+"tmp/"+fileName
    fileOut = DATA_SWAP2+"tmp/"+fileName+"_out"
    print(fileIn,fileOut)
    extract(fileIn,fileOut,sa)



def extractContent(content,sa,N =MAX_SNETENCES_COUNT):
    print("content")
    now=time.strftime("%M:%S")
    try:
        all_the_text = content
        # replace('\n', '')
        # print(all_the_text.decode('utf-8'))
        ret = summarize(all_the_text,N)

        # print(ret.decode('gbk'))
        # for x in ret:
        #     # print(x.decode('utf-8'))
        #     x = x.strip()
        #     x=x.replace('\n', '')
        #     if(len(x)>=1):
        #         wOut.write(x+'\r\n')
        rate = None
        if("true"==sa):
            rate = sentimentlAnalysis.predictForConttent(all_the_text)
    except Exception as e:
       print ("Error: unable to fetch data"+e)
    # finally:
    end=time.strftime("%M:%S")
    print(now+","+end)
    return ret,rate

# 我们预计|我们估计|应该关注|归母净利润|扣非净利润|EPS
def matchKeyPoint(sentence):
    searchObj = re.search( u"(.*)(我们认为|我们估计|应该关注|归母净利润|扣非净利润|EPS)+(.*)", sentence, re.M|re.I)

    # 风险提示
    return searchObj

def matchNotValidPoint(sentence):
    searchObj = re.search( u"(.*)(风险提示)+(.*)", sentence, re.M|re.I)

    return searchObj
    # ,searchObj.group(2)

if __name__ == '__main__':
    import sys
    print(sys.path)
    # **************按照文章文件路径抽取规则**************
    # fileName = sys.argv[1] 
    # sa = sys.argv[2] 
    # # fileName = "1517464803380_6031"
    # # sa = "true"

    # # array = fileNameStr.split('###', 1 );
    # # fileName = array[0]
    # # sa = array[1]
    # extractForFileinName (fileName,sa)
    # with open(DATA_SWAP+"1513649943714_7477", "r", encoding='utf-8') as myfile:
    #     text = myfile.read().replace('\n', '')
    #     print(summarize(text, 10))

    # sentence = u"dsa 我们认为 智能化银行战略符合行业长逻辑，将有力支持公司未来走出一条轻型、智能、高效的特色银行道路。 *"
    # ret = matchKeyPoint(sentence)
    # print(ret)

# **************按照文章内容抽取规则**************
    file = open("/Users/daichanglin/Desktop/igoldenbeta/robotSVN/robot/robot-parent/python/sa/test/rp2",
            "r",encoding = 'utf-8')
    article = file.read()
    rankS,rate = extractContent(article,"true2",MAX_SNETENCES_COUNT)
    print(rankS)
    print(rate)
