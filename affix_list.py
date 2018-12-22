import ast
import re

def clean( words ):
    words = words.split(',')
    out = []
    for word in words :
        word = re.sub( '[-$\s]' , '' , word )
        out.append(word)
    return out


i = int(input('input 1 to add suffix and 2 to prefix'))
tmp = ''

if i==1:
    print('input suffix or \'Del\' to cancle or \'ctrl+c\' to end')
    with open('suffix.txt') as file:
        get = [set(ast.literal_eval(line)) for line in file][0]
        while True:
            try :
                in_ = input()
                if in_=='Del':
                    for i in clean(tmp) :
                        get.remove(i)
                elif re.match( '[\s]+' , in_ ) or in_==''  :
                    continue
                else :
                    for i in clean(in_) :
                        get.add(i)
                    tmp = in_
            except KeyboardInterrupt:
                out = ''
                for i in get:
                    out += '\'{}\' , '.format(i)
                break

    with open('suffix.txt' , 'w') as file1 :
        file1.write(out)
        print('suffix:\n' , out)
if i==2:
    print('input prefix or \'Del\' to cancle or \'ctrl+c\' to end')
    with open('prefix.txt') as file:
        get = [set(ast.literal_eval(line)) for line in file][0]
        while True:
            try :
                in_ = input()
                if in_=='Del':
                    for i in clean(tmp) :
                        get.remove(i)
                elif re.match( '[\s]+' , in_ ) or in_==''  :
                    continue
                else :
                    for i in clean(in_) :
                        get.add(i)
                    tmp = in_
            except KeyboardInterrupt:
                out = ''
                for i in get:
                    out += '\'{}\' , '.format(i)
                break

    with open('prefix.txt' , 'w') as file1 :
        file1.write(out)
        print('prefix:\n' , out)


"""
with open('prefix.txt') as file:
    get = [set(ast.literal_eval(line)) for line in file][0]
    out = ''
    for i in get:
        out += '\'{}\' , '.format(i)

with open('prefix.txt' , 'w') as file1 :
    file1.write(out)
    print('prefix:\n' , out)
"""
