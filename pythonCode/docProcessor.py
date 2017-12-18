import gensim
import jieba
import pandas as pd
word_dict = {}
EMBEDDING_INDEX = gensim.models.KeyedVectors.load_word2vec_format('/Users/grant/Desktop/vectors_w2v_unclean_100d.txt', binary=False)


def wordCount(doc):
    word_dict = {}
    for word in doc.split(' '):
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1
    return word_dict


def topN_mostSimilar(embedding,query,n):
    topn_list = EMBEDDING_INDEX.most_similar(query,topn=n)
    rSet = {}
    for i in range(len(topn_list)):
        rSet[topn_list[i][0]] = topn_list[i][1]
    return rSet

def getSegStr(doc):
    doc_seg = jieba.cut(doc,cut_all=False)
    doc_str = " ".join(doc_seg)
    return doc_str

def getIntersectionSet(embedding,query,n,doc):
    embeddict = topN_mostSimilar(embedding,query,n)
    set1 = embeddict.keys()
    word_dict = wordCount(getSegStr(doc))
    set2 = word_dict.keys()
    return word_dict,embeddict,set1&set2

#optimize---------->
def score(word_dict,embeddict,interset):
    s = 0;
    for word in interset:
        print(word+"------------->"+str(embeddict[word]))
        if(embeddict[word] >= 0.6):
            s = s + embeddict[word]
        if (embeddict[word] < 0.6 and embeddict[word] > 0.45):
            s = s + embeddict[word] *0.5
        if (embeddict[word] <0.45):
            s = s + embeddict[word] * 0.2
    return s

def sortDocs(embedding,query,n,docs):
    docIds = []
    scores = []
    i = 0
    for doc in docs:
        word_dict,embeddict,interset = getIntersectionSet(embedding,query,n,doc)
        print(interset)
        print('---------->>>>>>>>  '+str(i))
        doc_score = score(word_dict,embeddict,interset)
        docIds.append(i)
        scores.append(doc_score)
        i = i + 1
    score_df_data = {'docId':docIds , 'score': scores }
    rank_df = pd.DataFrame.from_dict(score_df_data)
    return rank_df

if __name__ == "__main__":
    word_dict = wordCount("我 我 你 java python");
    w2vSet = EMBEDDING_INDEX.vocab.keys()
    print(word_dict.keys()&w2vSet)