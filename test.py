
from text_utils import word_base , word_affix
import json

"""
print(word_base('this is a sentence.'))
ddd = json.load(open('./test.json'))
print(ddd)
"""
sentence = "this is a sentence absorbtion avalible pretrain ." 
print( word_affix( sentence , 2 ) )


