import os

os.getcwd() # current working directory
user_name = 'ASUS'
path_name = os.path.join('C:\\Users',user_name,'Desktop')
os.chdir(path_name) # change directory

listdir = os.listdir() #list item in () (current directory by default)

if not (os.path.exists('mkdir') or os.path.exists('makedirs/dirs')):
    os.mkdir('mkdir') # make only 1 folder at a time
    os.makedirs('makedirs/dirs') # can make sub folder in one-line

os.rename('mkdir','hitman47')

os.removedirs('makedirs/dirs') # remove the entire dir
os.rmdir('hitman47') #remove specific folder

print(os.stat('kpi.txt')) #show item stat

from datetime import datetime
print(datetime.fromtimestamp(os.stat('kpi.txt').st_atime)) #readable form

from six.moves.urllib.request import urlretrieve
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'

if not os.path.exists(path_name+'\\tt\\tt.data'):
    urlretrieve(url,path_name+'\\tt\\tt.data') #save file from url and store at C:\\Users\\ASUS\\Desktop\\tt
names = ['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
import pandas as pd
os.chdir(".\\tt")
##os.rename('tt.data','tt.data.txt')
df = pd.read_csv('tt.data.txt',names=names)
'''
print('Hello', 'World', 2+3, file=open('file.txt', 'w')) #tips to crate file name file.txt with 'hello world 5' inside
'''
