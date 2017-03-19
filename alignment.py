# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 17:22:41 2017

Implementation of DP for Minimum Edit Distance: Wagner-Fisher algorithm
Obtained from: http://www.giovannicarmantini.com/2016/01/minimum-edit-distance-in-python
"""

import numpy as np
#import tabulate as tb

def wagner_fischer(word_1, word_2, grphDict):
    
    # Insertion and Deletion cost. Set very high to try and choose substitutions if possible
    insCost = 1500
    delCost = 1500
    
    n = len(word_1) + 1  # counting empty string 
    m = len(word_2) + 1  # counting empty string

    # initialize D matrix
    D = np.zeros(shape=(n, m), dtype=np.int)
    D[:,0] = range(n)
    D[0,:] = range(m)

    for i, l_1 in enumerate(word_1, start=1):
        for j, l_2 in enumerate(word_2, start=1):
            deletion = D[i-1,j] + delCost
            insertion = D[i, j-1] + insCost
            
            # Substitution cost is based on the graphemic confusion map. 
            # If there isn't an entry for that particular pair, then the cost is just the same as insertion or deletion
            substitutionCost = delCost
            #substitutionFrame = grphDict[(grphDict['original']==l_1) & (grphDict['mistaken']==l_2)]['frequency']
            
            #substitutionFrame = grphDict[(grphDict['pair']==(l_1 + l_2))]['frequency']
#            if(len(substitutionFrame)!=0):
#                substitutionCost = float(substitutionFrame)

#           substitution = D[i-1,j-1] + substitutionCost
            #print(substitutionCost)
            if(grphDict.get(l_1).get(l_2)):
                substitutionCost = grphDict.get(l_1).get(l_2)
                
            substitution = D[i-1,j-1] + substitutionCost#(0 if l_1==l_2 else substitutionCost)

            mo = np.min([deletion, insertion, substitution])
            D[i,j] = mo
            
    med = D[(D.shape[0])-1,(D.shape[1])-1]
    return med