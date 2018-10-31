import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
import random
from random import randrange

data = np.loadtxt("iris.csv", delimiter=",")
h=0.05
# Create color maps
cmap_light = ListedColormap(['#5bc509', '#f28080', '#AAAAFF'])
cmap_bold = ListedColormap(['#16692d', '#b43a3a', '#605a9c'])
#create test data
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
test=np.c_[xx.ravel(), yy.ravel()]

def Euclidean(test,train):
  distance=defaultdict(list)
  for i in range(len(test)):
    for j in range(len(train)):
      distance[i].append(np.sqrt(sum((test[i]-train[j])**2)))
  return distance

def Neighbour(distance):
        neighbour=defaultdict(list)
        for i in range(len(distance)):
                neighbour[i]=sorted(range(len(distance[i])),key=distance[i].__getitem__)
        return neighbour

def majority(neighbour,k,train):
        selected,Majority={},list()
        for i in range(len(neighbour)):
			#print("Neighbour--->"+str(neighbour))
			selected={}
			for j in range(k):
				label=train[neighbour[i][j]][-1]
				selected[label]=selected.get(label,0)+1
			#print("selected-->"+str(selected))
			Majority.append(max(selected,key=selected.get))
	return Majority

def KNN(test,train,K):
  distance=Euclidean(test,train[:,:2])
  neighbour=Neighbour(distance)
  prediction=majority(neighbour,K,train)
  return prediction

m=[0,10,20,30,50]
#adding noise
Error=[]
for noise in m:
  random.seed(1)
  error=0
  random_indices=list()
  data_index=range(150)
  while len(random_indices)<=noise:
	  index = randrange(len(data_index))
	  random_indices.append(data_index.pop(index))
  a = np.empty_like (data[:])
  a[:] = data[:] #flipping the labels
  for i in random_indices:
    if a[i][-1] != 3:
      a[i][-1]+=1
    else:
      a[i][-1]-=1
  Z=np.array(KNN(test,a,3))
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#plot training data
  plt.scatter(a[:, 0], a[:, 1], c=a[:,-1], cmap=cmap_bold)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.title("3-Class classification (k = %i, noise=%s)"
                    % (3, noise))
  plt.show()
  for i in range(len(a)):
	  train_data=range(len(a))
	  train_data_indices=[j for j, k in enumerate(train_data) if j !=i]
	  LOOCV=[a[d] for d in train_data_indices]
	  LOOCV_train=np.array(LOOCV)
	  prediction=KNN(a[i,:2].reshape(1,2),LOOCV_train,3)
	  if a[i][-1]!=prediction:error+=1
  error=float(error)/150
  Error.append(error)
#plot Error
plt.title('Testing Error')
plt.xlabel('Noise in data',fontsize=15)
plt.ylabel('Error',fontsize=15)
plt.plot(m, Error,label='Testing Error')
legend = plt.legend(loc='upper right', shadow=True)
plt.show()
