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
#####Pre-Post crash analysis#####
#################################
            
#Datelist for pre and post crash
preCrashDateList=datelist(date(2005,1,1),date(2006,12,31))
preCrashFiles=metaFileDF[metaFileDF['datetime'].isin(preCrashDateList)].filePath
crashDateList=datelist(date(2007,1,1),date(2008,12,31))
crashFiles=metaFileDF[metaFileDF['datetime'].isin(crashDateList)].filePath
postCrashDateList=datelist(date(2007,1,1),date(2008,12,31))
postCrashFiles=metaFileDF[metaFileDF['datetime'].isin(postCrashDateList)].filePath

allFiles=list(preCrashFiles)+list(crashFiles)+list(postCrashFiles)

#Subset tokens only for crash dates
crashTokens={key:value for key, value in rawTokens.items() if key in allFiles}
#Seems some files did not come through will double check
           
#Get word coCo
startTime=time.time()
crashCoCo=bd.coOccurence(crashTokens,10)
endTime=time.time()
print(endTime-startTime)
#2874.17690277 seconds w/ stop words

#Get DSM
startTime=time.time()
crashDSM=bd.DSM(crashCoCo,50)
endTime=time.time()
print(endTime-startTime)

#Remove coCo
del crashCoCo

#Get context vectors
startTime=time.time()
crashCVDict=bd.contextVectors(crashTokens,crashDSM,10)
endTime=time.time()
print(endTime-startTime)

#Remove tokens and DSM
del crashTokens
del crashDSM


#Bring in crash wordlist
crashWordList=['business','loan','economy','bank','bailout','stability',
               'stimulus','tax','billion','mortgage','recovery','stock',
               'street','unemployment','jobs','foreclosure','treasury',
               'regulation','greed','recession', 'credit']
crashWordList=[stemmer.stem(word) for word in crashWordList]

#Run cosine sim for pre crash files
startTime=time.time()
preCrashCosine=bd.averageCosine(crashCVDict,preCrashFiles,crashWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(preCrashCosine).to_csv('./preCrash_cosine.csv')

startTime=time.time()
crashCosine=bd.averageCosine(crashCVDict,crashFiles,crashWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(crashCosine).to_csv('./crash_cosine.csv')

startTime=time.time()
postCrashCosine=bd.averageCosine(crashCVDict,postCrashFiles,crashWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(postCrashCosine).to_csv('./postCrash_cosine.csv')

