# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:42:33 2016

@author: nmvenuti
Context vector code development
"""

import os, glob, time
import sys
import nltk
import string
import re
import numpy as np
from sklearn.decomposition import TruncatedSVD
import random
import math

###################################
######Set up inital parameters#####
###################################

#Set function parameters for package
tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
stemmer = nltk.stem.snowball.EnglishStemmer()
punctuation = set(string.punctuation)
#stopWords = nltk.corpus.stopwords.words('english')

#Define tokenize function
def tokenize(fileList):
    tokens={}
    
    for fileName in fileList:
        
        #Try to clean files, otherwise note errors
        try:
            #Clean raw text into token list
            rawText=open(fileName).read()
            
            #Update for encoding issues            
            rawText=unicode(rawText, "utf-8", errors="ignore")
            #Tokenize
            textList=tokenizer.tokenize(rawText)
            
            #Convert all text to lower case
            textList=[word.lower() for word in textList]
            
            #Remove punctuation
            textList=[word for word in textList if word not in punctuation]
            textList=["".join(c for c in word if c not in punctuation) for word in textList ]
            
            #convert digits into NUM
            textList=[re.sub("\d+", "NUM", word) for word in textList]  
            
            #Stem words
            textList=[stemmer.stem(word) for word in textList]
#            stemStopwords=[stemmer.stem(word) for word in stopWords]
             
            #Remove stopwords
#            stemStopwords.append("")
#            textList=[word for word in textList if word not in stemStopwords]
                
            #Add to dictionary if textList len greater than zero
            if len(textList)>0:
                tokens[fileName]=textList    
        except:
            pass
    #Return tokens
    return(tokens)

#Define co-occurence function
def coOccurence(tokens,k):
    
    #Define coOccurence dict
    coCoDict={}
    
    #Loop through each file
    for fileName in tokens.keys():
        for i in range(len(tokens[fileName])):
            #Adjust window to contain words k in front or k behind
            lowerBound=max(0,i-k)
            upperBound=min(len(tokens),i+k)
            coCoList=tokens[fileName][lowerBound:i]+tokens[fileName][i+1:upperBound+1]
            window=tokens[fileName][i]
            
            #Add window to coCoDict if not present
            if window not in coCoDict.keys():
                coCoDict[window]={}
            
            #Add words to coCoDict for window
            for word in coCoList:
                try:
                    coCoDict[window][word]+=1
                except:
                    coCoDict[window][word]=1
    
    #Return CoCoDict
    return(coCoDict)


#Define function to perform SVD on co-occurences
def DSM(coCoDict,k):
    vocab=list(coCoDict.keys())
    coCoList=[]
    for row in vocab:
        rowList=[]
        for column in vocab:
            try:
                rowList.append(coCoDict[row][column])
            except:
                rowList.append(0)
        coCoList.append(rowList)
    
    svd= TruncatedSVD(n_components=k, random_state=42)
    svd.fit(coCoList)       
    coCoSVD=svd.transform(coCoList)
    
    #Convert back to dictionary
    svdDict={}
    for i in range(len(vocab)):
        svdDict[vocab[i]]={}
        for j in range(k):
            svdDict[vocab[i]][j]=coCoSVD[i][j]
    
    #Return DSM
    return(svdDict)

#Define function to create context vectors
def contextVectors(tokens,dsm,k):
    
    #Define coOccurence dict
    cvDict={}
    
    #Loop through each file
    for fileName in tokens.keys():
        cvDict[fileName]={}
        for i in range(len(tokens[fileName])):
            #Adjust window to contain words k in front or k behind
            lowerBound=max(0,i-k)
            upperBound=min(len(tokens),i+k)
            cvList=tokens[fileName][lowerBound:i]+tokens[fileName][i+1:upperBound+1]
            window=tokens[fileName][i]
            
            #Add entry for cvDict if window not yet present
            if window not in cvDict[fileName].keys():
                cvDict[fileName][window]={}
            
            #Create context vector            
            contextVector={}
            
            for word in cvList:
                for key in dsm[word].keys():                    
                    #Update context vector
                    try:
                        contextVector[key]=contextVector[key]+dsm[word][key]
                    except:
                        contextVector[key]=dsm[word][key]
            
            #Add context vector to cvDict
            cvIndex=len(cvDict[fileName][window])+1
            cvDict[fileName][window][cvIndex]=contextVector
    
    #Return context vector dictionary
    return(cvDict)

#Define context vector function
def get_cosine(vec1,vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    return float(numerator/denominator)


#Define context vector simulation
def averageCosine(cvDict,fileList,wordList,sim):
    subCV={}
    cosineResults=[]
    for fileName in fileList:
        for word in cvDict[fileName].keys():
            #Add word if not in keys
            if word not in subCV.keys():
                subCV[word]={}
            for i in range(len(cvDict[fileName][word])):
                subCV[word][len(subCV[word])+1]=cvDict[fileName][word][i+1]
    for searchWord in wordList:
        if len(subCV[searchWord])>1:
            consineSim=np.zeros(sim)
            for i in range(sim):
                x=random.randrange(0, len(subCV[searchWord]))
                y=random.randrange(0, len(subCV[searchWord]))
                
                consineSim[i]=get_cosine(subCV[searchWord][x+1],subCV[searchWord][y+1])
            approx_avg_cosine=np.average(consineSim)
        else:
            approx_avg_cosine=-1
        cosineResults.append([searchWord,approx_avg_cosine])
    return cosineResults

##################################
######Testing function on WBC#####
##################################
#
##Get file list
#wbPath = '/home/nmvenuti/Desktop/nmvenuti_sandbox/Capstone/webscraping westboro/sermons'
#wbFileList=[infile for infile in glob.glob( os.path.join(wbPath, '*.*') )]
#
##Get tokens for wbc
#wbTokens=tokenize(wbFileList)
#            
##Get word coCo for wbc
#start=time.time()            
#wbCoCo=coOccurence(wbTokens,10)
#print(time.time()-start)
##10012.3053741 seconds
#
##
###Get DSM
#start=time.time()
##wbDSM=DSM(wbCoCo,10)
#print(time.time()-start)
#
##Remove coCo
#del wbCoCo
#
##Get context vectors
#wbCVDict=contextVectors(wbTokens,wbDSM,100)
#
##Remove tokens and DSM
#del wbTokens
#del wbDSM
#
#fileList=list(wbCVDict.keys())
##Test cosine similairty function on 'God'
#averageCosine(wbCVDict,fileList,['god','home'],1000)
