import spacy
import ast
import re


def word_base( sentence ) :
    nlp = spacy.load('en' , disable=[ 'tagger' , 'parser' , 'ner' , 'textcat' ] )
    doc = nlp( sentence )
    t = tuple()
    for word in doc :
        t += ( word.lemma_ , )
    return t

def word_affix( sentence , n_match ) :
    words = sentence.split(' ')
    with open('./suffix.txt' , 'r') as f1 :
        suf_list = [list(ast.literal_eval(line)) for line in f1][0]
    with open('./prefix.txt' , 'r') as f2 :
        pre_list = [list(ast.literal_eval(line)) for line in f2][0]
    suffix2index = dict( zip( suf_list , range(len(suf_list)) ) )
    prefix2index = dict( zip( pre_list , range(len(pre_list)) ) )
    suf_regex = []
    pre_regex = []
    for suffix in suf_list :
        suf_regex.append(re.compile(suffix+r'$'))
    for prefix in pre_list :
        pre_regex.append(re.compile(r'^'+prefix))
    affix_vec = []
    for i , word in enumerate(words) :
        affix_vec.append([])
        for j , reg in enumerate(suf_regex) :
            if reg.search( word ) :
                if len(affix_vec[i])>=n_match :
                    break
                else :
                    affix_vec[i].append( suffix2index[suf_list[j]]+1 ) #add 1 ,cause 0 for no-match
        while len(affix_vec[i])<n_match :
            affix_vec[i].append(0)
        for j , reg in enumerate(pre_regex) :
            if reg.search( word ) :
                if len(affix_vec[i])>=2*n_match :
                    break
                else :
                    affix_vec[i].append( prefix2index[pre_list[j]]+1 )
        while len(affix_vec[i])<2*n_match :
            affix_vec[i].append(0)
    print(affix_vec)
