# plot_sns_mnist_01.py
# 
#
import numpy as np
import matplotlib.pyplot as plt
from ans import ans


#================================
# MNIST: 0 v.s. 1
#================================

A = np.load('A_mnist_01.npy')
L = np.load('L_mnist_01.npy')

NED, NED0, NED1, sns_incr, sns_bi, g0Count, g1Count = ans(A,L)

#--------------
# Sort
#--------------

CE_idx = sns_bi.argsort()
sortedCE = sns_bi[sns_bi.argsort()]

#==============================================
# A stacked bar plot on bin counts - Combined
#==============================================
myblue = '#%02x%02x%02x' % (0,107,164)
myred = '#%02x%02x%02x' % (214,39,40)

width = 0.7 # width of the bars
numBins = 10
label1 = 1
label0 = 0
lNames = ['0','1']

#------------------------------
# Histogram of the best node
#------------------------------
targetNode = CE_idx[0]
ind = np.arange(numBins)

fig = plt.figure(figsize=(5,5))
p1 = plt.bar(ind,g1Count[targetNode,],width,color=myred,bottom=g0Count[targetNode,],hatch='//')
p2 = plt.bar(ind,g0Count[targetNode,],width,color=myblue)

plt.ylabel('Counts',fontsize=17)
plt.xlabel('Activation bin middle value ($10^{-2}$)',fontsize=17)
plt.title('Node '+str(targetNode+1)+', SNS=%.4f' %(sns_bi[targetNode]),fontsize=17)
plt.xticks(ind,(' 5',' 15',' 25',' 35',' 45',' 55',' 65',' 75',' 85',' 95'))
plt.tick_params(labelsize=15)
plt.tight_layout()

lg = plt.legend((p1[0],p2[0]),(lNames[0],lNames[1]),loc='upper center',title='Class label')
plt.setp(lg.get_title(),fontsize=17)


plt.show()
fig.savefig('NMIST_01_bestNode.png')

#------------------------------
# SNS curve
#------------------------------
fig = plt.figure(figsize=(10,3))

snsx = np.arange(len(sns_bi))
plt.plot(snsx,sortedCE,'k', linewidth=2, label='SNS with binary distr.')

plt.legend(loc='lower right')

xlocation = np.arange(0,len(sns_bi),10)
xlabels = CE_idx[xlocation] + 1
plt.xticks(xlocation, xlabels, rotation='vertical')

plt.ylabel('SNS',fontsize=17)
plt.xlabel('Sorted nodes (in original node number)',fontsize=17)
plt.title('0 vs. 1')
plt.xlim([-3,260])
plt.ylim([0,1.05])
plt.tight_layout()
plt.show()
fig.savefig('MNIST_01_sns.png')

#--------------
# Plot NED
#--------------
fig = plt.figure(figsize=(10,3))

nedRed = '#%02x%02x%02x' % (214,39,40)
nedx = np.arange(len(NED))
plt.plot(nedx,NED[CE_idx],'k', linewidth=2, label='NED')
plt.plot(nedx,NED0[CE_idx],':',mfc=myblue, linewidth=2,label='$NED_0$')
plt.plot(nedx,NED1[CE_idx],'--',mfc=nedRed,linewidth=2, label='$NED_1$')

plt.legend(loc='lower right')

plt.ylabel('NED',fontsize=17)
plt.xlabel('Sorted by SNS with Binary Distribution',fontsize=17)
plt.title('0 vs. 1')
plt.xlim([-3,305])
plt.tight_layout()
plt.show()
fig.savefig('MNIST_01_NED.png')



