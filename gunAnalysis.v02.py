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

#Get filelist for date metadata
metaFilePath='./CPD/metadata'
metaFileList = listdir(metaFilePath)
metaFileList=[metaFilePath+'/'+fileName for fileName in metaFileList if 'all' in fileName]


#Create pandas dataframe with all metadata files
for fileName in metaFileList:
    df = pd.read_csv(fileName)
    try:
        metaFileDF=metaFileDF.append(df, ignore_index=True)
    except:
        metaFileDF=df

metaFileDF['datetime']=pd.to_datetime(metaFileDF['date'])
metaFileDF['filePath']=rawPath+'/'+ metaFileDF['year'].map(str)+'/'+metaFileDF['fileName']+'.txt'

#################################            
#####Gun analysis#####
#################################

#Bring in attack wordlist
wordList=['gun','gunman','shoot','tragedy','background','mental','hate',
'safety','god','death','kill','hatred','control','congress','amendment','right']

wordList=[stemmer.stem(word) for word in wordList]

#Datelist for gun events
gun1DateList=datelist(date(2009,8,5),date(2010,2,5))
gun1Files=metaFileDF[metaFileDF['datetime'].isin(gun1DateList)].filePath
gun2DateList=datelist(date(2012,4,20),date(2012,10,20))
gun2Files=metaFileDF[metaFileDF['datetime'].isin(gun2DateList)].filePath
gun3DateList=datelist(date(2012,9,14),date(2013,3,14))
gun3Files=metaFileDF[metaFileDF['datetime'].isin(gun3DateList)].filePath
gun4DateList=datelist(date(2015,6,16),date(2015,12,16))
gun4Files=metaFileDF[metaFileDF['datetime'].isin(gun4DateList)].filePath
gun5DateList=datelist(date(2015,9,2),date(2016,3,2))
gun5Files=metaFileDF[metaFileDF['datetime'].isin(gun5DateList)].filePath

allFiles=list(gun1Files)+list(gun2Files)+list(gun3Files)+list(gun4Files)+list(gun5Files)

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

gun1Files = [doc for doc in gun1Files if doc in tokens.keys()]
gun2Files = [doc for doc in gun2Files if doc in tokens.keys()]
gun3Files = [doc for doc in gun3Files if doc in tokens.keys()]
gun4Files = [doc for doc in gun4Files if doc in tokens.keys()]
gun5Files = [doc for doc in gun5Files if doc in tokens.keys()]


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
gun1Cosine=bd.averageCosine(CVDict,gun1Files,wordList,1000)
pd.DataFrame(gun1Cosine).to_csv('./gun1_cosine.csv')
print('finished sem density 1')
sys.stdout.flush()

print('starting get sem density 2')
sys.stdout.flush()
gun2Cosine=bd.averageCosine(CVDict,gun2Files,wordList,1000)
pd.DataFrame(gun2Cosine).to_csv('./gun2_cosine.csv')
print('finished sem density 2')
sys.stdout.flush()

print('starting get sem density 3')
sys.stdout.flush()
gun3Cosine=bd.averageCosine(CVDict,gun3Files,wordList,1000)
pd.DataFrame(gun3Cosine).to_csv('./gun3_cosine.csv')
print('finished sem density 3')
sys.stdout.flush()

print('starting get sem density 4')
sys.stdout.flush()
gun4Cosine=bd.averageCosine(CVDict,gun4Files,wordList,1000)
pd.DataFrame(gun4Cosine).to_csv('./gun4_cosine.csv')
print('finished sem density 4')
sys.stdout.flush()

print('starting get sem density 5')
sys.stdout.flush()
gun5Cosine=bd.averageCosine(CVDict,gun5Files,wordList,1000)
pd.DataFrame(gun5Cosine).to_csv('./gun5_cosine.csv')
print('finished sem density 5')
sys.stdout.flush()



#Get TF for each period
gun1TF=bd.getPeriodTF(docTF,gun1Files)
gun1TF_DF = pd.DataFrame.from_dict(gun1TF, orient = 'index')
gun1TF_DF.columns = ['TF']
gun1TF_DF = gun1TF_DF.sort('TF',ascending=False)
gun1TF_DF.to_csv('./gun1_TF.csv', encoding='utf-8')

#Get TF for each period
gun2TF=bd.getPeriodTF(docTF,gun2Files)
gun2TF_DF = pd.DataFrame.from_dict(gun2TF, orient = 'index')
gun2TF_DF.columns = ['TF']
gun2TF_DF = gun2TF_DF.sort('TF',ascending=False)
gun2TF_DF.to_csv('./gun2_TF.csv', encoding='utf-8')

#Get TF for each period
gun3TF=bd.getPeriodTF(docTF,gun3Files)
gun3TF_DF = pd.DataFrame.from_dict(gun3TF, orient = 'index')
gun3TF_DF.columns = ['TF']
gun3TF_DF = gun3TF_DF.sort('TF',ascending=False)
gun3TF_DF.to_csv('./gun3_TF.csv', encoding='utf-8')

#Get TF for each period
gun4TF=bd.getPeriodTF(docTF,gun4Files)
gun4TF_DF = pd.DataFrame.from_dict(gun4TF, orient = 'index')
gun4TF_DF.columns = ['TF']
gun4TF_DF = gun4TF_DF.sort('TF',ascending=False)
gun4TF_DF.to_csv('./gun4_TF.csv', encoding='utf-8')

#Get TF for each period
gun5TF=bd.getPeriodTF(docTF,gun5Files)
gun5TF_DF = pd.DataFrame.from_dict(gun5TF, orient = 'index')
gun5TF_DF.columns = ['TF']
gun5TF_DF = gun5TF_DF.sort('TF',ascending=False)
gun5TF_DF.to_csv('./gun5_TF.csv', encoding='utf-8')

#Get ContextList for KNN
tfList=[[k,v] for k,v in TF.items()]
tfList.sort(key=lambda x:x[1], reverse = True)
tfList=[x[0] for x in tfList]
contextList=list(set(tfList[50:250]+wordList))

#Get knn for context vectors in pre attack files
print('starting get knn gun1')
sys.stdout.flush()
startTime=time.time()
gun1KNN=bd.knnContextVector(CVDict,gun1Files,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn gun1')
print((endTime-startTime)/3600)
sys.stdout.flush()
gun1KNN=[yList for xList in gun1KNN for yList in xList if xList!=None]
pd.DataFrame(gun1KNN).to_csv('./gun1_knn.csv', index=False, header=False)

print('starting get knn gun2')
sys.stdout.flush()
startTime=time.time()
gun2KNN=bd.knnContextVector(CVDict,gun2Files,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn gun2')
print((endTime-startTime)/3600)
sys.stdout.flush()
gun2KNN=[yList for xList in gun2KNN for yList in xList if xList!=None]
pd.DataFrame(gun2KNN).to_csv('./gun2_knn.csv', index=False, header=False)

print('starting get knn gun3')
sys.stdout.flush()
startTime=time.time()
gun3KNN=bd.knnContextVector(CVDict,gun3Files,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn gun3')
print((endTime-startTime)/3600)
sys.stdout.flush()
gun3KNN=[yList for xList in gun3KNN for yList in xList if xList!=None]
pd.DataFrame(gun3KNN).to_csv('./gun3_knn.csv', index=False, header=False)

print('starting get knn gun4')
sys.stdout.flush()
startTime=time.time()
gun4KNN=bd.knnContextVector(CVDict,gun4Files,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn gun4')
print((endTime-startTime)/3600)
sys.stdout.flush()
gun4KNN=[yList for xList in gun4KNN for yList in xList if xList!=None]
pd.DataFrame(gun4KNN).to_csv('./gun4_knn.csv', index=False, header=False)

print('starting get knn gun5')
sys.stdout.flush()
startTime=time.time()
gun5KNN=bd.knnContextVector(CVDict,gun5Files,contextList,wordList,1000,5)
endTime=time.time()
print('finished knn gun5')
print((endTime-startTime)/3600)
sys.stdout.flush()
gun5KNN=[yList for xList in gun5KNN for yList in xList if xList!=None]
pd.DataFrame(gun5KNN).to_csv('./gun5_knn.csv', index=False, header=False)
