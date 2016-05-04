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

sys.stdout = open("terrorlog.txt", "a")
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

################################            
#####Pre-Post 9/11 analysis#####
################################

#Bring in attack wordlist
attackWordList=['terrorism','laden','qaeda','wmd','attack','homeland',
'security','defend','islam', 'freedom','iraq','afghanistan','peace',
'war','protect','god']
attackWordList=[stemmer.stem(word) for word in attackWordList]

#Datelist for pre and post attack
preAttackDateList=datelist(date(1999,9,11),date(2001,9,10))
preAttackFiles=metaFileDF[metaFileDF['datetime'].isin(preAttackDateList)].filePath
postAttackDateList=datelist(date(2001,9,11),date(2003,9,11))
postAttackFiles=metaFileDF[metaFileDF['datetime'].isin(postAttackDateList)].filePath

print('pre',len(preAttackFiles))
print('post',len(postAttackFiles))
sys.stdout.flush()
allFiles=list(preAttackFiles)+list(postAttackFiles)

if runSample == True:
    import random
    allFiles = random.sample(allFiles, 200)

print('starting tokenization')
sys.stdout.flush()
attackTokens = bd.tokenize(allFiles)
# Filter Tokens List to only Documents that contain a Target Word
attackTokens = bd.retainRelevantDocs(attackTokens, attackWordList)
print('finished tokenization')
sys.stdout.flush()
#Subset tokens only for attack dates
#attackTokens={key:value for key, value in rawTokens.items() if key in allFiles}

preAttackFiles = [doc for doc in preAttackFiles if doc in attackTokens.keys()]
postAttackFiles = [doc for doc in postAttackFiles if doc in attackTokens.keys()]
print('pre',len(preAttackFiles))
print('post',len(postAttackFiles))
print('all',len(allFiles))
print('alltokens',len(attackTokens.keys()))

#Get word coCo
print('starting word coco')
sys.stdout.flush()
attackCoCo, TF, docTF = bd.coOccurence(attackTokens,6)
print('finished word coco!')
print('length of coco dict',len(attackCoCo.keys()))
sys.stdout.flush()

#Get DSM
print('starting DSM')
sys.stdout.flush()
startTime=time.time()
attackDSM=bd.DSM(attackCoCo,50)
endTime=time.time()
print('finished DSM!')
print((endTime-startTime)/3600)
print('dsm shape',len(attackDSM.keys()))
#print('dsm dim',attackDSM[list(attackDSM.keys())[0]])
sys.stdout.flush()

#Remove coCo
del attackCoCo
gc.collect()

#Get context vectors
print('starting context vectors')
sys.stdout.flush()
startTime=time.time()
attackCVDict=bd.contextVectors(attackTokens, attackDSM, attackWordList, 6)
print('finished context vectors!')
endTime=time.time()
print((endTime-startTime)/3600)
print('context vectors dict len',len(attackCVDict.keys()))
sys.stdout.flush()

#Remove tokens and DSM
del attackTokens
del attackDSM
gc.collect()


#Run cosine sim for pre attack files
print('starting get sem density 1')
sys.stdout.flush()
startTime=time.time()
preAttackCosine=bd.averageCosine(attackCVDict,preAttackFiles,attackWordList,1000)
endTime=time.time()
print('finished sem density 1')
print((endTime-startTime)/3600)
sys.stdout.flush()

pd.DataFrame(preAttackCosine).to_csv('./preAttack_cosine.csv')

print('starting sem density 2')
sys.stdout.flush()
startTime=time.time()
postAttackCosine=bd.averageCosine(attackCVDict,postAttackFiles,attackWordList,1000)
endTime=time.time()
print('finished sem density 2')
print((endTime-startTime)/3600)
sys.stdout.flush()
pd.DataFrame(postAttackCosine).to_csv('./postAttack_cosine.csv')

#Get TF for each period
preAttackTF=bd.getPeriodTF(docTF,preAttackFiles)
preAttackTF_DF = pd.DataFrame.from_dict(preAttackTF, orient = 'index')
preAttackTF_DF.columns = ['TF']
preAttackTF_DF = preAttackTF_DF.sort('TF',ascending=False)
preAttackTF_DF.to_csv('./preAttack_TF.csv')

postAttackTF=bd.getPeriodTF(docTF,postAttackFiles)
postAttackTF_DF = pd.DataFrame.from_dict(postAttackTF, orient = 'index')
postAttackTF_DF.columns = ['TF']
postAttackTF_DF = postAttackTF_DF.sort('TF',ascending=False)
postAttackTF_DF.to_csv('./postAttack_TF.csv')

#Get ContextList for KNN
tfList=[[k,v] for k,v in TF.items()]
tfList.sort(key=lambda x:x[1], reverse = True)
tfList=[x[0] for x in tfList]
contextList=list(set(tfList[50:250]+attackWordList))

#Get knn for context vectors in pre attack files
print('starting get knn preAttack')
sys.stdout.flush()
startTime=time.time()
preAttackKNN=bd.knnContextVector(attackCVDict,preAttackFiles,contextList,attackWordList,1000,5)
endTime=time.time()
print('finished knn preAttack')
print((endTime-startTime)/3600)
sys.stdout.flush()

pd.DataFrame(preAttackKNN).to_csv('./preAttack_knn.csv', index=False, header=False)

#Get knn for context vectors in pre attack files
print('starting get knn postAttack')
sys.stdout.flush()
startTime=time.time()
postAttackKNN=bd.knnContextVector(attackCVDict,postAttackFiles,contextList,attackWordList,1000,5)
endTime=time.time()
print('finished knn postAttack')
print((endTime-startTime)/3600)
sys.stdout.flush()

pd.DataFrame(postAttackKNN).to_csv('./postAttack_knn.csv', index=False, header=False)

