# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:08:57 2016

@author: nmvenuti
"""

import os, glob, time
import os.path
from os import listdir
import sys
import pandas as pd
from datetime import date, timedelta as td
import nltk
sys.path.append('.')
import BromanticDensity as bd
stemmer = nltk.stem.snowball.EnglishStemmer()

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
rawTokens=bd.tokenize(rawFileList)

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
#####Pre-Post 9/11 analysis#####
#################################
            
#Datelist for pre and post attack
preAttackDateList=datelist(date(1999,9,11),date(2001,9,10))
preAttackFiles=metaFileDF[metaFileDF['datetime'].isin(preAttackDateList)].filePath
postAttackDateList=datelist(date(2001,9,11),date(2003,9,11))
postAttackFiles=metaFileDF[metaFileDF['datetime'].isin(postAttackDateList)].filePath

allFiles=list(preAttackFiles)+list(postAttackFiles)

#Subset tokens only for attack dates
attackTokens={key:value for key, value in rawTokens.items() if key in allFiles}

            
#Get word coCo
attackCoCo=bd.coOccurence(attackTokens,10)

#Get DSM
startTime=time.time()
attackDSM=bd.DSM(attackCoCo,100)
endTime=time.time()
print(endTime-startTime)

#Remove coCo
del attackCoCo

#Get context vectors
startTime=time.time()
attackCVDict=bd.contextVectors(attackTokens,attackDSM,10)
endTime=time.time()
print(endTime-startTime)

#Remove tokens and DSM
del attackTokens
del attackDSM

#Bring in crash wordlist
attackWordList=['terrorism','laden','qaeda','wmd','attack','homeland',
'security','defend','islam', 'freedom','iraq','afghanistan','peace',
'war','protect','god']

attackWordList=[stemmer.stem(word) for word in attackWordList]

#Run cosine sim for pre attack files
startTime=time.time()
preAttackCosine=bd.averageCosine(attackCVDict,preAttackFiles,attackWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(preAttackCosine).to_csv('./preAttack_cosine.csv')

startTime=time.time()
postAttackCosine=bd.averageCosine(attackCVDict,postAttackFiles,attackWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(postAttackCosine).to_csv('./postAttack_cosine.csv')

