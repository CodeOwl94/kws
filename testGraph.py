# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:04:56 2017

@author: Ashwin
MLSALT 5: Question 5
"""

from indexer import Indexer
from datetime import datetime

startTime = datetime.now()

indexer = Indexer('decode.ctm')
test = indexer.makeGraphDict('grapheme.map')
indexer.queries('queries.xml')
indexer.hitsHeader('decode-grph.xml')
indexer.hitsFile('decode-grph.xml', 'TRUE')


#queryMorpy = indexer.queryMorphDict(0,'morph.kwslist.dct')
#indexer.initWithMorph('decode.ctm','morph.dct')
#indexer.morphQueryToHits('queries.xml', 'morph.kwslist.dct','decode-word-morph.xml', 'TRUE')


print(datetime.now() - startTime)