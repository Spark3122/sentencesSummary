#encoding=utf-8
from __future__ import print_function
from __future__ import unicode_literals
import sys
print("extract1")
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

import sys
sys.path.append("../")


from snownlp import SnowNLP
import re
import codecs



DATA_SWAP = "/data/AI/python/";
DATA_SWAP2 = "/data/AI/python/";
N = 12

def extract(filein,fileout):
    print("extract2")
    try:
        f = codecs.open(filein, 'r','utf-8')
        w = codecs.open(fileout, 'w','utf-8') # 若是'wb'就表示写二进制文件
        all_the_text = f.read()
        # print(all_the_text.decode('utf-8'))
        ret = extractSummary(all_the_text,12)
        # print(ret.encode('gbk'))
        for x in ret:
            # print(x.decode('utf-8'))
            w.write(x+'\r\n')
    finally:
        if f:
            f.close()
        if w:
            w.close()

def extractSummary(sentences,N):
    # print("extractSummary")
    # if(sentences.)
    # print(sentences[0:50])
    s = SnowNLP(sentences)
    ret = s.summary(N)
    # print(ret)
    print("suc")
    return ret

def test(string1,string2):
    return string1+string2


import sys
print(sys.path)
fileName = sys.argv[1]  
DATA_SWAP = DATA_SWAP+fileName
DATA_SWAP2 = DATA_SWAP2+fileName+"_out"
print(DATA_SWAP,DATA_SWAP2)
extract(DATA_SWAP,DATA_SWAP2)

# if __name__=='__main__':
   
