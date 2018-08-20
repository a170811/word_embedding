import tensorflow as tf
from tqdm import tqdm
from preprocess import Preprocessing
import os

url = 'http://mattmahoney.net/dc/text8.zip' 
data_path = 'text8.zip' 

##load data
if not os.path.exists(data_path):
    print("Downloading the dataset... (It may take some time)")
    filename, _ = urllib.urlretrieve(url, data_path)
    print("Done!")
# Unzip the dataset file. Text has already been processed
#with zipfile.ZipFile('./text8.zip') as f :

#    text_words = f.read(f.namelist()[0]).lower().decode('utf-8').split()
#testing data
text_words = ['moment', 'homeless', 'disable', 'bore', 'frustrate', 'apple', 'milk', 'is', 'good', 'to', 'drink', 'delicious']

data_obj = Preprocessing( text_words , min_occurrence = 0.1 )
print( 'text_words = ' , text_words )
print( 'vocabulary size = ' , data_obj.vocabulary_size )
print( 'data : \n' , data_obj.data )
print( 'data_base_form : \n' , data_obj.base )
print( 'tri_tram : \n' , data_obj.text_gram )
print( 'prefix : \n' , data_obj.prefix )
print( 'suffix : \n' , data_obj.suffix )
