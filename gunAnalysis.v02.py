<<<<<<< HEAD
runSample = False

import gc, sys
import os, glob, time
import os.path
from os import listdir
import sys
import pandas as pd
from datetime import date, timedelta as td, datetime
import nltk
sys.path.append('.')
import BromanticDensity as bd
stemmer = nltk.stem.snowball.EnglishStemmer()

sys.stdout = open("gunlog.txt", "a")
print(str(datetime.now()))
print('Finished importing modules')
sys.stdout.flush()

#Create date list function
def datelist(d1,d2):
    delta = d2 - d1
    dateList=[]
    for i in range(delta.days + 1):
        dateList.append(d1 + td(days=i))
    return(dateList)

###############################
#####Raw File List Extract#####
###############################

#Get file list for documents
rawPath = './CPD/data'
rawFileList=[]
for dirpath, dirnames, filenames in os.walk(rawPath):
    for filename in [f for f in filenames ]:
        rawFileList.append(os.path.join(dirpath, filename))


#Get tokens
#rawTokens=bd.tokenize(rawFileList)

#Get short meta csv
metaFileDF=pd.read_csv('./CPD/shortMeta.csv')

metaFileDF['datetime']=pd.to_datetime(metaFileDF['date'])
metaFileDF['filePath']=rawPath+'/'+ metaFileDF['year'].map(str)+'/'+metaFileDF['fileName']+'.txt'

######################            
#####Gun analysis#####
######################

#Bring in attack wordlist
wordList=['gun','gunman','shoot','tragedy','background','mental','hate',
'safety','god','death','kill','hatred','control','congress','amendment','right']

wordList=[stemmer.stem(word) for word in wordList]

#Datelist for gun events
preSandyDateList=datelist(date(2012,9,14),date(2012,12,13))
preSandyFiles=metaFileDF[metaFileDF['datetime'].isin(preSandyDateList)].filePath
postSandyDateList=datelist(date(2012,12,14),date(2013,3,14))
postSandyFiles=metaFileDF[metaFileDF['datetime'].isin(postSandyDateList)].filePath


allFiles=list(preSandyFiles)+list(postSandyFiles)

if runSample == True:
    import random
    allFiles = random.sample(allFiles, 200)

print('starting tokenization')
sys.stdout.flush()
tokens = bd.tokenize(allFiles)
# Filter Tokens List to only Documents that contain a Target Word
tokens = bd.retainRelevantDocs(tokens, wordList)
print('finished tokenization')
sys.stdout.flush()
#Subset tokens only for attack dates
#attackTokens={key:value for key, value in rawTokens.items() if key in allFiles}

preSandyFiles = [doc for doc in preSandyFiles if doc in tokens.keys()]
postSandyFiles = [doc for doc in postSandyFiles if doc in tokens.keys()]


#Get word coCo
print('starting word coco')
sys.stdout.flush()
CoCo, TF, docTF = bd.coOccurence(tokens,6)
print('finished word coco!')
print('length of coco dict',len(CoCo.keys()))
sys.stdout.flush()

#Get DSM
print('starting DSM')
sys.stdout.flush()
startTime=time.time()
DSM=bd.DSM(CoCo,50)
endTime=time.time()
print('finished DSM!')
print((endTime-startTime)/3600)
print('dsm shape',len(DSM.keys()))
#print('dsm dim',attackDSM[list(attackDSM.keys())[0]])
sys.stdout.flush()

#Remove coCo
del CoCo
gc.collect()

#Get context vectors
print('starting context vectors')
sys.stdout.flush()
startTime=time.time()
CVDict=bd.contextVectors(tokens, DSM, wordList, 6)
print('finished context vectors!')
endTime=time.time()
print((endTime-startTime)/3600)
print('context vectors dict len',len(CVDict.keys()))
sys.stdout.flush()

#Remove tokens and DSM
del tokens
del DSM
gc.collect()


#Run cosine sim for pre attack files
print('starting get sem density 1')
sys.stdout.flush()
preSandyCosine=bd.averageCosine(CVDict,preSandyFiles,wordList,1000)
pd.DataFrame(preSandyCosine).to_csv('./preSandy_cosine.csv')
print('finished sem density 1')
sys.stdout.flush()

print('starting get sem density 2')
sys.stdout.flush()
postSandyCosine=bd.averageCosine(CVDict,postSandyFiles,wordList,1000)
pd.DataFrame(postSandyCosine).to_csv('./postSandy_cosine.csv')
print('finished sem density 2')
sys.stdout.flush()

#Get TF for each period
preSandyTF=bd.getPeriodTF(docTF,preSandyFiles)
preSandyTF_DF = pd.DataFrame.from_dict(preSandyTF, orient = 'index')
preSandyTF_DF.columns = ['TF']
preSandyTF_DF = preSandyTF_DF.sort('TF',ascending=False)
preSandyTF_DF.to_csv('./preSandy_TF.csv', encoding='utf-8')

#Get TF for each period
postSandyTF=bd.getPeriodTF(docTF,postSandyFiles)
postSandyTF_DF = pd.DataFrame.from_dict(postSandyTF, orient = 'index')
postSandyTF_DF.columns = ['TF']
postSandyTF_DF = postSandyTF_DF.sort('TF',ascending=False)
postSandyTF_DF.to_csv('./postSandy_TF.csv', encoding='utf-8')

#Get ContextList for KNN
tfList=[[k,v] for k,v in TF.items()]
tfList.sort(key=lambda x:x[1], reverse = True)
tfList=[x[0] for x in tfList]
contextList=list(set(tfList[50:250]+wordList))

#Get knn for context vectors in pre attack files
print('starting get knn preSandy')
sys.stdout.flush()
startTime=time.time()
preSandyKNN=bd.knnContextVector(CVDict,preSandyFiles,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn preSandy')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(preSandyKNN).to_csv('./preSandy_knn.csv', index=False, header=False)

print('starting get knn postSandy')
sys.stdout.flush()
startTime=time.time()
postSandyKNN=bd.knnContextVector(CVDict,postSandyFiles,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn postSandy')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(postSandyKNN).to_csv('./postSandy_knn.csv', index=False, header=False)
