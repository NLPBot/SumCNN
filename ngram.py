
from collections import *

def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

def get_uni_gram_dict(word_list):
    uni_dict = defaultdict(int)
    d = find_ngrams(word_list,1)
    for term in d:
        term = ' '.join(list(term))
        uni_dict[term] = 1 
    return uni_dict

def get_bi_gram_dict(word_list):
    bi_dict = defaultdict(int)
    d = find_ngrams(word_list,2)
    for term in d:
        term = ' '.join(list(term))
        bi_dict[term] = 1
    return bi_dict

def get_tri_gram_dict(word_list):
    tri_dict = defaultdict(int)
    d = find_ngrams(word_list,3)
    for term in d:
        term = ' '.join(list(term))
        tri_dict[term] = 1 
    return tri_dict
    
def get_quad_gram_dict(word_list):
    quad_dict = defaultdict(int)
    d = find_ngrams(word_list,4)
    for term in d:
        term = ' '.join(list(term))
        quad_dict[term] = 1 
    return quad_dict
    
def get_ngram_sim(sent,summary):

    sum_uni = get_uni_gram_dict(summary)
    sum_bi = get_bi_gram_dict(summary)
    sum_tri = get_tri_gram_dict(summary)
    sum_quad = get_quad_gram_dict(summary)
    uni_score, bi_score, tri_score, quad_score = 0., 0., 0., 0.
    
    if len(sent)>0:
        for uni_gram in find_ngrams(sent,1):
            term = ' '.join(list(uni_gram))    
            if term in sum_uni.keys():
                uni_score += 1
        uni_score /= (len(list(find_ngrams(sent,1))))
            
    if len(sent)>1:
        for bi_gram in find_ngrams(sent,2):
            term = ' '.join(list(bi_gram))    
            if term in sum_bi.keys():
                bi_score += 1
        bi_score /= (len(list(find_ngrams(sent,2))))
            
    if len(sent)>2:        
        for tri_gram in find_ngrams(sent,3):
            term = ' '.join(list(tri_gram))    
            if term in sum_tri.keys():
                tri_score += 1    
        tri_score /= (len(list(find_ngrams(sent,3))))
                
    if len(sent)>3:
        for quad_gram in find_ngrams(sent,4):
            term = ' '.join(list(quad_gram))    
            if term in sum_quad.keys():
                quad_score += 1
        quad_score /= (len(list(find_ngrams(sent,4))))
    
    if len(sent)<1: return 0.
    if len(sent)<2: return uni_score
    if len(sent)<3: return bi_score*0.25+uni_score*0.75
    if len(sent)<4: return tri_score*0.25+bi_score*0.30+uni_score*0.45
    return (uni_score*0.10+bi_score*0.50+tri_score*0.25+quad_score*0.15)
    
if __name__=="__main__":
    sent = ['i','am','hungry']
    summary = ['j','sdf','sdf','i','u','a','what','hungry']
    print('score is '+str(get_ngram_sim(sent,summary)))
    
    
    
    
    
    
    

