
from text_utils import word_base , word_affix , make_dict , load_dict , tri_gram
import json


#print(word_base('this is a sentence.'))
#ddd = json.load(open('./test.json'))
#print(ddd)

sentence = "homeless tired disable mislead unhappily" 
print( word_affix( sentence.split() , 2 ) )

"""
dict2index , index2dict = load_dict()
sen = 'this is relation apple is very delicious'
print(word_base(sen))
a = tri_gram( word_base(sen) , dict2index ) 
#b = list(map( lambda x : index2dict[x] , a ) )
print(a)
#print(b)
"""

