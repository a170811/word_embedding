import spacy
import ast
import re
import os

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
    

#input sentence and output list
def word_base( sentence ) :
    nlp = spacy.load('en' , disable=[ 'tagger' , 'parser' , 'ner' , 'textcat' ] )
    doc = nlp( sentence )
    t = []
    for word in doc :
        t.append(word.lemma_) 
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
    affix_vec = []
    for i , word in enumerate(words) :
        affix_vec.append([])
        for j , reg in enumerate(pre_regex) :
            if reg.search( word ) :
                if len(affix_vec[i])>=n_match :
                    break
                else :
                    affix_vec[i].append( prefix2index[pre_list[j]]+1 )
        while len(affix_vec[i])<n_match :
            affix_vec[i].append(0)

        for j , reg in enumerate(suf_regex) :
            if reg.search( word ) :
                if len(affix_vec[i])>=2*n_match :
                    break
                else :
                    affix_vec[i].append( suffix2index[suf_list[j]]+1 ) #add 1 ,cause 0 for no-match
        while len(affix_vec[i])<2*n_match :
            affix_vec[i].append(0)
    return affix_vec

#input list output list
def tri_gram( sentence , dict2index , length = 20 ) :
    output = []

    for word in sentence :

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
    
