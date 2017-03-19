#!/usr/bin/python

#The Indexer class

import pandas as pd
import numpy as np
from sklearn import preprocessing
from string import ascii_lowercase
from alignment import wagner_fischer
#import string

class Indexer: 
	
    # Don't put values here that you want to keep separate between instances of Indexer
    
    
    def __init__(self, refFile):
        
        overallList = []
        self.queryMorph = {}
        self.oneBestMorphDict = {}
        self.graphDict = {} # Grapheme confusion dictionary used to obtain substitution costs 
        
        occurenceCounter = 0
        
        with open(refFile, 'r') as f:
            for line in f:
                wordSeq = line.split()
                
                # Remove uppercase letters and punctuation
                token  = wordSeq[4].lower()
                # Python 2
                #token =  token.translate(None, string.punctuation)
                # Python 3
                #token = token.translate(str.maketrans('','',string.punctuation))
                
                newOccurence = {'id':occurenceCounter, 'file':wordSeq[0], 'channel':wordSeq[1], 'start':float(wordSeq[2]), \
                                'duration':float(wordSeq[3]), 'token':token, 'score': float(wordSeq[5]), 'successor':-1}
                overallList.append(newOccurence)
                occurenceCounter = occurenceCounter + 1
                
#                # Create the link between this word and it's predecessor if there is one to be made.
#                # We are indexing by -2 due to the counter being incremented. 1 step back is the current word, 2 for predecessor
#                if(occurenceCounter>=2):
#                    previousFileName = overallList[occurenceCounter-2]['file']
#                    timeLimit = overallList[occurenceCounter-2]['start'] + overallList[occurenceCounter-2]['duration'] + 0.5
#                    
#                    currentFileName = newOccurence['file']
#                    currentStartTime =newOccurence['start']
#                    
#                    if((previousFileName==currentFileName) and currentStartTime<=timeLimit):
#                        overallList[occurenceCounter-2]['successor'] = 1
            
        # Create the links between words
        for i in range(0, len(overallList)-1):
            currentFileName = overallList[i]['file']
            timeLimit = overallList[i]['start'] + overallList[i]['duration'] + 0.5
                                   
            nextFileName = overallList[i+1]['file']
            nextStartTime = overallList[i+1]['start']
           
            if((currentFileName==nextFileName) and nextStartTime<=timeLimit):
                overallList[i]['successor'] = 1
        
        self.oneBest = pd.DataFrame(overallList)
        self.vocab = pd.DataFrame(self.oneBest.token.unique())
        self.vocab.columns = ['token']   
        
    # Similar to __init__ but with morphological decomposition added
    def initWithMorph(self,oneBestFile, morphDict):
        
        self.queryMorphDict(0,morphDict)
        
        overallList = []
        occurenceCounter = 0
        
        with open(oneBestFile, 'r') as f:
            for line in f:
                wordSeq = line.split()
                
                # Remove uppercase letters and punctuation
                token  = wordSeq[4].lower()
                # Python 2
                #token =  token.translate(None, string.punctuation)
                # Python 3
                #token = token.translate(str.maketrans('','',string.punctuation))
                
                if token in self.oneBestMorphDict:
                    tokenSeq = self.oneBestMorphDict[token].split()
                    startTime = float(wordSeq[2])
                    splitDuration = float(wordSeq[3])/len(tokenSeq)
                    
                    for i  in range(0, len(tokenSeq)):                  
                
                        newOccurence = {'id':occurenceCounter, 'file':wordSeq[0], 'channel':wordSeq[1], 'start':startTime+(i*splitDuration), \
                                        'duration':splitDuration, 'token':tokenSeq[i], 'score': float(wordSeq[5]), 'successor':-1}
                        overallList.append(newOccurence)
                        occurenceCounter = occurenceCounter + 1
                else:
                    newOccurence = {'id':occurenceCounter, 'file':wordSeq[0], 'channel':wordSeq[1], 'start':float(wordSeq[2]), \
                                        'duration':float(wordSeq[3]), 'token':wordSeq[4].lower(), 'score': float(wordSeq[5]), 'successor':-1}
                    overallList.append(newOccurence)
                    occurenceCounter = occurenceCounter + 1
            
        # Create the links between words
        for i in range(0, len(overallList)-1):
            currentFileName = overallList[i]['file']
            timeLimit = overallList[i]['start'] + overallList[i]['duration'] + 0.5
                                   
            nextFileName = overallList[i+1]['file']
            nextStartTime = overallList[i+1]['start']
           
            if((currentFileName==nextFileName) and nextStartTime<=timeLimit):
                overallList[i]['successor'] = 1
        
        self.oneBest = pd.DataFrame(overallList)
    
    # Find occurences of a phrase
    def search(self,phrase,normalized):
        wordSeq = phrase.split()
        numberOfWords = len(wordSeq)
        
        occurences = []
        firstWordMatch= self.oneBest[(self.oneBest['token']==wordSeq[0])]
        for i in range(0, len(firstWordMatch)):
            
            matchFound = 0
            if(numberOfWords==1):
                matchFound = 1
            
            pastWordSlice = firstWordMatch.iloc[i]
            startTime = pastWordSlice['start']
            duration = pastWordSlice['duration']
            successorFirst = pastWordSlice['successor']
            score = [pastWordSlice['score']]
            
            j = 1
            
            # Enter this loop only if we are dealing with a phrase 
            # The first term checks to see that we have been asked to look for a phrase
            # The second term checks to see if the hit we have found is the start of a phrase
            while(j<numberOfWords and successorFirst!=-1):
                currentWordSlice = self.oneBest.iloc[pastWordSlice['id'] + 1] 
                
                successor = currentWordSlice['successor']
                
                # If we are not at the last word and there is no successor
                if(successor==-1 and j<numberOfWords-1):
                    break
                
                currentWord = currentWordSlice['token']
                
                if(currentWord!=wordSeq[j]):
                    break
                
                currentScore = currentWordSlice['score']
                score.append(currentScore)
                pastWordSlice = currentWordSlice
                j = j + 1
                
                
                if(j==numberOfWords):
                    matchFound = 1
                    startTimeLastWord = currentWordSlice['start']
                    durationOfLastWord = currentWordSlice['duration']
                    duration = startTimeLastWord + durationOfLastWord- startTime
                    
            
            #score = np.prod(score)
            score = np.average(score)              
                 
            if(matchFound==1):             
                occured = {'id':firstWordMatch.iloc[i]['id'], 'start':startTime, 'duration':duration, 'score':score,\
                            'file':firstWordMatch.iloc[i]['file'], 'channel': firstWordMatch.iloc[i]['channel']}
                occurences.append(occured)
        
        # Once all occurences have been found, apply score normalization if the flag was true
        if(normalized==True and len(occurences)>=1):
            scores = []
            for i in range(0,len(occurences)):
                scores.append(occurences[i]['score'])
            
            scoresNorm = preprocessing.normalize(scores,'l1')[0] # Unsure why the normalization here returns a list of list
            
            for i in range(0,len(occurences)):
                occurences[i]['score'] = scoresNorm[i]
            
        
        occurences = pd.DataFrame(occurences)            
        return occurences
    
    # Give a phrase, is each word of the phrase IV? 
    def wordsIV(self, phrase):
        wordSeq = phrase.split()
        
        for word in wordSeq:
            if not (word in self.vocab.token.values):
                return False
            
        return True
        
    
    # Given a queries.xml file, shift the contents into a dataframe for easier handling
    # TODO: Make this a dictionary instead of a dataframe
    def queries(self, queryFile):
        
        overallList = {}
        kwid= ''
        kwtext = ''
        
        with open(queryFile, 'r') as f:
            next(f)
            for line in f:
                
                if 'kwid' in line: 
                    kwid = line[12:-3]
                elif 'kwtext' in line:
                    kwtext = line[12:-10]
                    
                    # Remove uppercase letters and punctuation
                    kwtext  = kwtext.lower()
                    # Python 2
                    #kwtext=  kwtext.translate(None, string.punctuation)
                    # Python 3
                    #kwtext = kwtext.translate(str.maketrans('','',string.punctuation))
                    
                    # Dataframe approach
                    #overallList.append({'kwid':kwid, 'kwtext':kwtext})
                    
                    # Double dict approach
                    overallList[kwid] = kwtext
                    
        #self.queriesList = pd.DataFrame(overallList)
        self.queriesList = overallList
        return(self.queriesList)
    
    # Given a set of queries, find the matches and write the output to the approrpriate file
    def hitsFile(self,outputFile,normalization):
        
        with open(outputFile, 'a') as f:
            for kwid in self.queriesList:
                #keyWordPhrase = self.queriesList.iloc[i]['kwtext']
                keyWordPhrase = self.queriesList[kwid]
                # First do a simple check to ensure that every word of the phrase is IV
                wordsIV = self.wordsIV(keyWordPhrase)
                
                # If any words of the occurence was OOV, find the replacement phrase where all words are IV
                if(wordsIV==False and len(self.graphDict)!=0):
                    #print('OOV Hit')
                    IVPhrase =''
                    keyWordPhraseItems = keyWordPhrase.split()
                    for word in keyWordPhraseItems:
                        # Check if this word of the phrase is OOV and if so, then find IV neighbour
                        IVWord = ''
                        if not(self.wordsIV(word)):
                            IVWord = self.findIV(word)
                        else:
                            IVWord = word 
                        IVPhrase = IVPhrase + IVWord + ' '
                    #print('OOV: ' + keyWordPhrase + ', IV: ' + IVPhrase[0:-1])
                    
                    keyWordPhrase = IVPhrase[0:-1]
                    
                
                occurences = self.search(keyWordPhrase,normalization)
                
                
                # Header for detection list for single phrase
                #kwid = self.queriesList.iloc[i]['kwid']
                f.write('<detected_kwlist kwid="'+ str(kwid) + '" oov_count="0" search_time="0.0">\n')
                
                for j in range(0, len(occurences)):
                    f.write('<kw file="' + occurences.iloc[j]['file'] + '" channel="' + occurences.iloc[j]['channel'] +\
                            '" tbeg="' + str(occurences.iloc[j]['start']) + '" dur="' + str(occurences.iloc[j]['duration']) + \
                            '" score="' + str(occurences.iloc[j]['score']) + '" decision="YES"/>\n')
                    
                # Footer for detection list for single phrase  
                f.write('</detected_kwlist>\n')
            
            # Footer of entire file
            f.write('</kwslist>')
    
    # Just fills in the header for the hits file
    def hitsHeader(self,outputFile):
        with open(outputFile, 'w+') as f:
            f.write('<kwslist kwlist_filename="IARPA-babel202b-v1.0d_conv-dev.kwlist.xml" language="swahili" system_id="">\n')

            
    # High level function that goes from queries.xml to hits.txt
    def queryToHits(self,queryFile,outputFile,normalization):
        if (normalization=='True'):
            normalization = True
        else:
             normalization = False   
        self.queries(queryFile)
        self.hitsHeader(outputFile)
        self.hitsFile(outputFile,normalization)
        
    # Create a morphological dictionary. The first parameter controls whether we are creating a morphed version of the ASR output or the queries. 
    def queryMorphDict(self, whichDict, morphFile):
        overallDict = {}
        with open(morphFile, 'r') as f:
            for line in f:
                wordSeq = line.split()
                key = wordSeq[0]
                firstWordLen = len(key)
                value = line[firstWordLen:].strip()
                overallDict[key] = value
        if(whichDict==0):
            self.oneBestMorphDict = overallDict
        else:
            self.queryMorph = overallDict
        
        #return self.queryMorph
    
    # Create a graphemic confusion matrix/dictionary
    def makeGraphDict(self, graphFile):
        overallDict = []
        counter = 0
        # Create basic dictionary 
        with open(graphFile, 'r') as f:
            for line in f:
                values = line.split()
                if(values[0]!='sil' and values[1]!='sil'):
                    #newEntry = {'id': counter, 'original':values[0], 'mistaken':values[1], 'frequency':float(values[2])}
                    newEntry = {'id': counter, 'original':values[0], 'mistaken': values[1], 'pair': values[0] + values[1], 'frequency':float(values[2])}
                    overallDict.append(newEntry)
                    counter = counter + 1
        
        overallDict = pd.DataFrame(overallDict)
          
        # Assign different costs per entry depending on frequency for that particular grapheme
        # Using some 'outside' knowledge: we have entries for a-z
        for character in ascii_lowercase:
            characterSet = overallDict[(overallDict['original']==character)]
            frequencySum = (characterSet.sum())['frequency']
            
            # Change the freqency counts to be a substitution cost
            for i in range(0,len(characterSet)):
                counterIndex = characterSet.iloc[i]['id']
                currentFrequency = characterSet.iloc[i]['frequency']
                overallDict.loc[overallDict.id==counterIndex, 'frequency'] = 1/(currentFrequency/frequencySum)
            
        # Convert this information to a Python double nested dictionary. The latter is a lot quicker. 
        grphDictPy= {}
        for i in range(0,len(overallDict)):
            if overallDict.iloc[i]['original'] not in grphDictPy:
                grphDictPy[overallDict.iloc[i]['original']] = {}
            grphDictPy[overallDict.iloc[i]['original']][overallDict.iloc[i]['mistaken']] = overallDict.iloc[i]['frequency']
            
        self.graphDict = grphDictPy
        return self.graphDict
        
    # For a given OOV word, find the closest IV word. Note here we deal with single words, not phrases!
    def findIV(self, OOVWord):
        bestScore = np.Infinity
        IVWord = ''
        for i in range(0, len(self.vocab)):
            score = wagner_fischer(list(self.vocab.token.iloc[i]), list(OOVWord), self.graphDict)
            if score<bestScore:
                bestScore = score
                IVWord = self.vocab.token.iloc[i]
                #print('score: ' + str(bestScore) + ' original: ' + OOVWord + ' candidate: ' + IVWord)

        
        #print('one IV word returned')
        #print(bestScore)
        return IVWord
    
    # Apply morphological decomposition to an exisiting query dataframe
    def morphQueries(self):
        for i in range(0, len(self.queriesList)):
            key = self.queriesList.iloc[i]['kwtext']
            if key in self.queryMorph:
                self.queriesList.iloc[i]['kwtext'] = self.queryMorph[key]
                
    
    # High level function that goes from queries.xml to hits.txt but applied morphological decomposition to the queries first
    def morphQueryToHits(self,queryFile,morphFile, outputFile, normalization):
        self.queryMorphDict(1, morphFile)
        self.queries(queryFile)
        self.morphQueries()
        self.hitsHeader(outputFile)
        self.hitsFile(outputFile,normalization)
