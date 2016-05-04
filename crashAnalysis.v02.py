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

sys.stdout = open("crashlog.txt", "a")
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

#########################            
##### Crash analysis#####
#########################

#Bring in attack wordlist
wordList=['business','loan','economy','bank','bailout','stability',
               'stimulus','tax','billion','mortgage','recovery','stock',
               'street','unemployment','jobs','foreclosure','treasury',
               'regulation','greed','recession', 'credit']

wordList=[stemmer.stem(word) for word in wordList]

#Datelist for pre and post crash
preCrashDateList=datelist(date(2005,1,1),date(2006,12,31))
preCrashFiles=metaFileDF[metaFileDF['datetime'].isin(preCrashDateList)].filePath
crashDateList=datelist(date(2007,1,1),date(2008,12,31))
crashFiles=metaFileDF[metaFileDF['datetime'].isin(crashDateList)].filePath
postCrashDateList=datelist(date(2007,1,1),date(2008,12,31))
postCrashFiles=metaFileDF[metaFileDF['datetime'].isin(postCrashDateList)].filePath

allFiles=list(preCrashFiles)+list(crashFiles)+list(postCrashFiles)

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

preCrashFiles = [doc for doc in preCrashFiles if doc in tokens.keys()]
crashFiles = [doc for doc in crashFiles if doc in tokens.keys()]
postCrashFiles = [doc for doc in postCrashFiles if doc in tokens.keys()]


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
preCrashCosine=bd.averageCosine(CVDict,preCrashFiles,wordList,1000)
pd.DataFrame(preCrashCosine).to_csv('./preCrash_cosine.csv')
print('finished sem density 1')
sys.stdout.flush()

print('starting get sem density 2')
sys.stdout.flush()
crashCosine=bd.averageCosine(CVDict,crashFiles,wordList,1000)
pd.DataFrame(crashCosine).to_csv('./crash_cosine.csv')
print('finished sem density 2')
sys.stdout.flush()

print('starting get sem density 3')
sys.stdout.flush()
postCrashCosine=bd.averageCosine(CVDict,postCrashFiles,wordList,1000)
pd.DataFrame(postCrashCosine).to_csv('./postCrash_cosine.csv')
print('finished sem density 3')
sys.stdout.flush()

#Get TF for each period
preCrashTF=bd.getPeriodTF(docTF,preCrashFiles)
preCrashTF_DF = pd.DataFrame.from_dict(preCrashTF, orient = 'index')
preCrashTF_DF.columns = ['TF']
preCrashTF_DF = preCrashTF_DF.sort('TF',ascending=False)
preCrashTF_DF.to_csv('./preCrash_TF.csv')

#Get TF for each period
crashTF=bd.getPeriodTF(docTF,crashFiles)
crashTF_DF = pd.DataFrame.from_dict(crashTF, orient = 'index')
crashTF_DF.columns = ['TF']
crashTF_DF = crashTF_DF.sort('TF',ascending=False)
crashTF_DF.to_csv('./crash_TF.csv')

#Get TF for each period
postCrashTF=bd.getPeriodTF(docTF,postCrashFiles)
postCrashTF_DF = pd.DataFrame.from_dict(postCrashTF, orient = 'index')
postCrashTF_DF.columns = ['TF']
postCrashTF_DF = postCrashTF_DF.sort('TF',ascending=False)
postCrashTF_DF.to_csv('./postCrash_TF.csv')

#Get ContextList for KNN
tfList=[[k,v] for k,v in TF.items()]
tfList.sort(key=lambda x:x[1], reverse = True)
tfList=[x[0] for x in tfList]
contextList=list(set(tfList[50:250]+wordList))

#Get knn for context vectors in pre attack files
print('starting get knn preCrash')
sys.stdout.flush()
startTime=time.time()
preCrashKNN=bd.knnContextVector(CVDict,preCrashFiles,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn preCrash')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(preCrashKNN).to_csv('./preCrash_knn.csv', index=False, header=False)

print('starting get knn crash')
sys.stdout.flush()
startTime=time.time()
crashKNN=bd.knnContextVector(CVDict,crashFiles,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn crash')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(crashKNN).to_csv('./crash_knn.csv', index=False, header=False)

print('starting get knn postCrash')
sys.stdout.flush()
startTime=time.time()
postCrashKNN=bd.knnContextVector(CVDict,postCrashFiles,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn postCrash')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(postCrashKNN).to_csv('./postCrash_knn.csv', index=False, header=False)
