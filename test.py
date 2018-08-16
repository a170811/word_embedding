
from text_utils import word_base , word_affix , make_dict , load_dict , tri_gram , load_base_dict
import json


#print(word_base('this is a sentence.'))
#ddd = json.load(open('./test.json'))
#print(ddd)

#sentence = "homeless tired disable mislead unhappily" 
#print( word_affix( sentence.split() , 2 ) )

dic = load_base_dict()
sen = 'this is relation apples is very delicious boring'
print(word_base(sen.split() , dic) )

"""
dict2index , index2dict = load_dict()
a = tri_gram( word_base(sen) , dict2index ) 
#b = list(map( lambda x : index2dict[x] , a ) )
print(a)
#print(b)
"""
