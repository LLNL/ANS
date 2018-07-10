# ans.py
# Autoencoder node saliency
#
# Input: 
# A - Activation values
# L - Classification labels, 0 or 1
#
# Output:
# NEDAll - NED curve
# NED0All - NED0 curve
# NED1All - NED1 curve
# sns_incr - SNS with increasing probability distribution
# sns_bi - SNS with binary distribution
# g0Count - Histogram counts for label 0
# g1Count - Histogram counts for label 1

import numpy as np


def ans(A,L,numBins=10):
	[n, numNodes] = A.shape

	#numBins = 20

	
        NEDAll = np.zeros((numNodes))
        NED0All = np.zeros((numNodes))
        NED1All = np.zeros((numNodes))
	sns_incr = np.zeros((numNodes))
	sns_bi = np.zeros((numNodes))

	histCount = np.zeros((numNodes,numBins))
	g1Count = np.zeros((numNodes,numBins))
	g0Count = np.zeros((numNodes,numBins))

	label1 = 1
	label0 = 0
	g1 = [i for [i,val] in enumerate(L) if val==label1]
	g0 = [i for [i,val] in enumerate(L) if val==label0]

	g1Total = len(g1)
        g0Total = len(g0)

	#nodeIter = 1
	for nodeIter in range(numNodes):
	#for nodeIter in range(1):

		# Histogram
		binRange = float(1.0/numBins)
		binMids = np.zeros((numBins))
		binMids[0] = binRange/2.0
		for iter in range(1,numBins):
		    binMids[iter] = binRange*(iter+1) - binMids[0]
		#print binMids
		#sumBinMids = np.sum(binMids)
		#print sumBinMids

		for iter1 in range(n):
		    for iter2 in range(numBins):
			if iter2 == 0 and A[iter1,nodeIter] >= binRange*iter2 and A[iter1,nodeIter] <= binRange*(iter2+1):
			    histCount[nodeIter,iter2] = histCount[nodeIter,iter2] + 1.0
			    #if abs(L[iter1]-label1) < 0.001:
			    if L[iter1]==label1:
				g1Count[nodeIter,iter2] = g1Count[nodeIter,iter2] + 1.0
			    #if abs(L[iter1]-label0) < 0.001:
			    if L[iter1]==label0:
				g0Count[nodeIter,iter2] = g0Count[nodeIter,iter2] + 1.0

			elif A[iter1,nodeIter] > binRange*iter2 and A[iter1,nodeIter] <= binRange*(iter2+1):
			    histCount[nodeIter,iter2] = histCount[nodeIter,iter2] + 1.0
			    #if abs(L[iter1]-label1) < 0.001:
			    if L[iter1]==label1:
				g1Count[nodeIter,iter2] = g1Count[nodeIter,iter2] + 1.0
			    #if abs(L[iter1]-label0) < 0.001:
			    if L[iter1]==label0:
				g0Count[nodeIter,iter2] = g0Count[nodeIter,iter2] + 1.0

		#print histCount
		#print g1Count[nodeIter,]
		#print g0Count[nodeIter,]            

                #----------------------------------
		# Normalized entropy difference
                #----------------------------------
		entropy = 0.0
                numOccupiedBins = 0.0
		for iter2 in range(numBins):
		    currentP = float(histCount[nodeIter,iter2]/n)
		    if currentP != 0.0:
			numOccupiedBins = numOccupiedBins + 1.0
			temp = - currentP*np.log2(currentP)
			entropy = entropy + temp
                if numOccupiedBins > 1:
                    NED = 1.0 - entropy/np.log2(numOccupiedBins)
                else:
                    NED = 1.0

		#------------------------------------
		# Class NED
		#------------------------------------
		entropy0 = 0.0
                numOccupiedBins0 = 0.0
		for iter2 in range(numBins):
		    currentP0 = float(g0Count[nodeIter,iter2]/g0Total)
		    if currentP0 != 0.0:
			numOccupiedBins0 = numOccupiedBins0 + 1.0
			temp = - currentP0*np.log2(currentP0)
			entropy0 = entropy0 + temp
                if numOccupiedBins0 > 1:
                    NED0 = 1.0 - entropy0/np.log2(numOccupiedBins0)
                else:
                    NED0 = 1.0

		entropy1 = 0.0
                numOccupiedBins1 = 0.0
		for iter2 in range(numBins):
		    currentP1 = float(g1Count[nodeIter,iter2]/g1Total)
		    if currentP1 != 0.0:
			numOccupiedBins1 = numOccupiedBins1 + 1.0
			temp = - currentP1*np.log2(currentP1)
			entropy1 = entropy1 + temp
                if numOccupiedBins1 > 1:
                    NED1 = 1.0 - entropy1/np.log2(numOccupiedBins1)
                else:
                    NED1 = 1.0
                '''
		#-----------------------------------
		# Cross entropy
		#-----------------------------------
		ce1 = 0.0
		ce2 = 0.0
		currentP = np.zeros((numBins,1))
		for iter2 in range(numBins):
		    groupTotal = g1Count[nodeIter,iter2]+g0Count[nodeIter,iter2]
		    currentP = float(histCount[nodeIter,iter2]/n)
		    if currentP != 0.0:
			if g1Count[nodeIter,iter2] != 0:
			    currentQ1 = float(g1Count[nodeIter,iter2]/groupTotal)
                            ce1 = ce1 - currentP*(currentQ1*np.log2(binMids[iter2])-(1.0-currentQ1)*np.log2(1.0-binMids[iter2]))

			if g0Count[nodeIter,iter2] != 0:
		   	    currentQ2 = float(g0Count[nodeIter,iter2]/groupTotal)
                            ce2 = ce2 - currentP*(currentQ2*np.log2(binMids[iter2])-(1.0-currentQ2)*np.log2(1.0-binMids[iter2]))
                '''
		#-----------------------------------
		# Cross entropy - increasing
		#-----------------------------------
		ce1 = 0.0
		ce2 = 0.0
		currentP = np.zeros((numBins,1))
		for iter2 in range(numBins):
		    groupTotal = g1Count[nodeIter,iter2]+g0Count[nodeIter,iter2]
		    currentP = float(histCount[nodeIter,iter2]/n)
            	    if g1Count[nodeIter,iter2] != 0:
		        currentQ1 = float(g1Count[nodeIter,iter2]/groupTotal)
                        if 1-currentQ1 != 0:
		            ce1 = ce1 - currentP*(binMids[iter2]*np.log2(currentQ1)- (1-binMids[iter2])*np.log2(1-currentQ1))           
 
                    if g0Count[nodeIter,iter2] != 0:
   	               currentQ2 = float(g0Count[nodeIter,iter2]/groupTotal)
                       if 1-currentQ2 != 0:
		            ce2 = ce2 - currentP*(binMids[iter2]*np.log2(currentQ2)- (1-binMids[iter2])*np.log2(1-currentQ2))
 

		#-----------------------------------
		# Cross entropy - binary
		#-----------------------------------
		ceo1 = 0.0
		ceo2 = 0.0
		currentP = np.zeros((numBins,1))
		for iter2 in range(numBins):
		    groupTotal = g1Count[nodeIter,iter2]+g0Count[nodeIter,iter2]
		    currentP = float(histCount[nodeIter,iter2]/n)
            	    if g1Count[nodeIter,iter2] != 0:
		        currentQ1 = float(g1Count[nodeIter,iter2]/groupTotal)
                        if 1-currentQ1 != 0:
		            #ceo1 = ceo1 - currentP*(binMids[iter2]*np.log2(currentQ1)- (1-binMids[iter2])*np.log2(1-currentQ1))                 
                            if iter2 > numBins/2:
		                ceo1 = ceo1 - currentP*(np.log2(currentQ1))
                            else:
				ceo1 = ceo1 - currentP*(np.log2(1-currentQ1))

                    if g0Count[nodeIter,iter2] != 0:
   	               currentQ2 = float(g0Count[nodeIter,iter2]/groupTotal)
                       if 1-currentQ2 != 0:
		            #ceo2 = ceo2 - currentP*(binMids[iter2]*np.log2(currentQ2)- (1-binMids[iter2])*np.log2(1-currentQ2))
                            if iter2 > numBins/2:
		                ceo2 = ceo2 - currentP*(np.log2(currentQ2))
                            else:
				ceo2 = ceo2 - currentP*(np.log2(1-currentQ2))

		
		  
                NEDAll[nodeIter] = NED
                NED0All[nodeIter] = NED0
                NED1All[nodeIter] = NED1
                # Supervised node saliency - increasing probability
		sns_incr[nodeIter] = min([ce1,ce2])
                # Supervised node saliency - binary
                sns_bi[nodeIter] = min([ceo1,ceo2])

	return NEDAll, NED0All, NED1All, sns_incr, sns_bi, g0Count, g1Count



