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

#Subset tokens only for attack dates
gunTokens={key:value for key, value in rawTokens.items() if key in allFiles}

            
#Get word coCo
gunCoCo=bd.coOccurence(gunTokens,10)

#Get DSM
startTime=time.time()
gunDSM=bd.DSM(gunCoCo,100)
endTime=time.time()
print(endTime-startTime)

#Remove coCo
del gunCoCo

#Get context vectors
startTime=time.time()
gunCVDict=bd.contextVectors(gunTokens,gunDSM,10)
endTime=time.time()
print(endTime-startTime)

#Remove tokens and DSM
del gunTokens
del gunDSM

#Bring in crash wordlist
gunWordList=['gun','gunman','shoot','tragedy','background','mental','hate',
'safety','god','death','kill','hatred','control','congress','amendment','right']

gunWordList=[stemmer.stem(word) for word in gunWordList]

#Run cosine sim
startTime=time.time()
gun1Cosine=bd.averageCosine(gunCVDict,gun1Files,gunWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(gun1Cosine).to_csv('./gun1_cosine.csv')

startTime=time.time()
gun2Cosine=bd.averageCosine(gunCVDict,gun2Files,gunWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(gun2Cosine).to_csv('./gun2_cosine.csv')

startTime=time.time()
gun3Cosine=bd.averageCosine(gunCVDict,gun3Files,gunWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(gun3Cosine).to_csv('./gun3_cosine.csv')

startTime=time.time()
gun4Cosine=bd.averageCosine(gunCVDict,gun4Files,gunWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(gun4Cosine).to_csv('./gun4_cosine.csv')

startTime=time.time()
gun5Cosine=bd.averageCosine(gunCVDict,gun5Files,gunWordList,1000)
endTime=time.time()
print(endTime-startTime)
pd.DataFrame(gun5Cosine).to_csv('./gun5_cosine.csv')