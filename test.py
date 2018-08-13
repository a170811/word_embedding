
from text_utils import word_base , word_affix , make_dict , load_dict
import json

"""
print(word_base('this is a sentence.'))
ddd = json.load(open('./test.json'))
print(ddd)
"""
sentence = "homeless tired disable mislead unhappily" 
print( word_affix( sentence , 2 ) )


