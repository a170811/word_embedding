from six.moves.urllib.request import urlretrieve

url = 'http://mattmahoney.net/dc/'
filename = 'text8.zip'
filename, _ = urlretrieve(url + filename, filename)


