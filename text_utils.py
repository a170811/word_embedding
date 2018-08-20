import spacy
import ast
import re
import os
from tqdm import tqdm

def make_dict( o_fileName = 'dict2index' ):
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
                
    with open(o_fileName,'w') as f:
        f.write(str(out))
    print('dictionary has been constructed !')

def load_dict( i_fileName = 'dict2index' ):
    if not os.path.exists( i_fileName ) :
        make_dict()
    with open( i_fileName , 'r' ) as f:
        d_str = f.read()
        dict2index = ast.literal_eval(d_str)
        index2dict = dict( zip(dict2index.values() , dict2index.keys()) )
    return dict2index , index2dict
    

def load_base_dict() :
    out = dict()
    with open('infl2ori.txt','r') as f:
        for line in f :
            i , j = line.split()
            out[i] = j
    return out


#input list and output list
def word_base( sentence , dic ) :
    t = []
    print('-----------------transfer to base form-------------------')
    for i , word in enumerate( tqdm(sentence , ncols=80 ) )  :
        t.append( dic.get(word , sentence[i] ) ) 
    return t

#input list output list
def word_affix( words , n_match ) :
    
    with open('./prefix.txt' , 'r') as f1 :
        pre_list = [list(ast.literal_eval(line)) for line in f1][0]
    with open('./suffix.txt' , 'r') as f2 :
        suf_list = [list(ast.literal_eval(line)) for line in f2][0]
    prefix2index = dict( zip( pre_list , range(len(pre_list)) ) )
    suffix2index = dict( zip( suf_list , range(len(suf_list)) ) )
    pre_regex = []
    suf_regex = []
    for prefix in pre_list :
        pre_regex.append(re.compile(r'^'+prefix))
    for suffix in suf_list :
        suf_regex.append(re.compile(suffix+r'$'))
    pre_vec = []
    suf_vec = []
    print('-----------------detect word affixes-------------------')
    for i , word in enumerate( tqdm(words , ncols=80) ) :
        pre_vec.append([])
        suf_vec.append([])
        for j , reg in enumerate(pre_regex) :
            if reg.search( word ) :
                if len(pre_vec[i])>=n_match :
                    break
                else :
                    pre_vec[i].append( prefix2index[pre_list[j]]+1 )
        if len(pre_vec[i])<n_match:
            pre_vec[i].extend( [0]*( n_match-len(pre_vec[i]) ) )

        for j , reg in enumerate(suf_regex) :
            if reg.search( word ) :
                if len(suf_vec[i])>=n_match :
                    break
                else :
                    suf_vec[i].append( suffix2index[suf_list[j]]+1 ) #add 1 ,cause 0 for no-match
        if len(suf_vec[i])<n_match:
            suf_vec[i].extend( [0]*( n_match-len(suf_vec[i]) ) )

    return pre_vec , suf_vec

#input list output list
def tri_gram( sentence , dict2index , length = 20 ) :
    output = []

    print('-----------------tri_gram scaning-------------------')
    for word in tqdm( sentence , ncols=80 ) :

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
    
