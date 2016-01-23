__author__ = 'lope'
import mlpModule
import numpy as np
from pylab import meshgrid,cm,imshow,contour,clabel,colorbar,axis,title,show

print("Helly ML!")
myMLP = mlpModule.MLP(2,8,1)
myBack = mlpModule.Backpropagation(myMLP, 0.3, 0.001)

print("Backpropagation: ------------------------------")
for i in range(5000):
    myBack.iterate([[0,0],[0,1],[1,0],[1,1], [0.5, 0.5], [0.75, 0.5], [0.3, 0.5], [0.45, 0.2], [0.2, 0.7]],

                   [[0],[1],[1],[0],[0],[1],[1],[0],[1]])
print(myMLP.compute([0,0]))
print(myMLP.compute([0,1]))
print(myMLP.compute([1,0]))
print(myMLP.compute([1,1])) #tender a 01 11 11 01
print("------------------------------")

x = np.arange(0,1.0,0.1)
y = np.arange(0,1.0,0.1)
X,Y = np.meshgrid(x, y)
Z = X

for i in range(10) :
    for j in range(10):
        Z[i][j] = myMLP.compute2Arg(X[i][j],Y[i][j])

im = imshow(Z,cmap=cm.RdBu) # drawing the function
cset = contour(Z,np.arange(0,1,0.2),linewidths=2,cmap=cm.Set2)
clabel(cset,inline=True,fmt='%1.1f',fontsize=10)
colorbar(im) # adding the colobar on the right
show()