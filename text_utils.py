import ast
import re
import os

def make_dict():
    out = dict()
    count = 0
    a_z = [ chr(i) for i in range(96,124) ]
    a_z[0] = '#'
    a_z[-1] = '$'
    for i in a_z[0:-1]: #start with '#' (chr(35))
        for j in a_z[1:-1]:
            for k in a_z[1:]: #end with '$' (chr(36))
                out[i+j+k] = count
                count += 1
                
    return out

def load_base_dict() :
    out = dict()
    with open('infl2ori.txt','r') as f:
        for line in f :
            i , j = line.split()
            out[i] = j
    return out


#input list and output list
def word_base(sentence, dic):
    t = []
    for i, word in enumerate(sentence):
        t.append(dic.get(word, 0)) 
    return t

def load_affix(prefix_path = './prefix.txt', suffix_path = './suffix.txt'):
    with open(prefix_path , 'r') as f1 :
        pre_list = [list(ast.literal_eval(line)) for line in f1][0]
    with open(suffix_path , 'r') as f2 :
        suf_list = [list(ast.literal_eval(line)) for line in f2][0]
    prefix2index = dict(zip(pre_list, range(len(pre_list))))
    suffix2index = dict(zip(suf_list, range(len(suf_list))))
    return pre_list, suf_list, prefix2index, suffix2index


#input list output list
def word_affix(words, a) :
    
    pre_list, suf_list, prefix2index, suffix2index = a
    pre_regex = []
    suf_regex = []
    for prefix in pre_list :
        pre_regex.append(re.compile(r'^'+prefix))
    for suffix in suf_list :
        suf_regex.append(re.compile(suffix+r'$'))
    vec = []
    for i , word in enumerate(words) :
        vec.append([])
        for j , reg in enumerate(pre_regex) :
            if reg.search(word) :
                vec[i].append(prefix2index[pre_list[j]]+1)
                break
        if len(vec[i]) < 1:
            vec[i].append(0)
        for j , reg in enumerate(suf_regex) :
            if reg.search(word) :
                vec[i].append(suffix2index[suf_list[j]]+1) #add 1 ,cause 0 for no-match
                break
        if len(vec[i]) < 2:
            vec[i].append(0)

    return vec

#input list output list
def tri_gram( sentence , dict2index , length = 20 ) :
    output = []

    for word in sentence:

        tmp_word = '#'+word+'$'
        output_tmp = []
        for i in range(len(tmp_word)) :
            if i+3 >= len(tmp_word) :
                break 
            else :
                tmp = dict2index[ tmp_word[i:i+3] ]
            if len(output_tmp) >= 20 :
                break 
            else :
                output_tmp.append( tmp ) 
        app = [0] * ( length - len(output_tmp) ) #append to 20 elements
        output_tmp.extend( app )
        output.append(output_tmp)
    return output
    
