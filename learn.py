import numpy as np
#import scipy as sp
from sklearn.ensemble import RandomForestClassifier

import Image # for visualizing images of the digits
import matplotlib.pyplot as plt

def loadCSV(filename):
  return np.genfromtxt(open(filename, 'r'), delimiter=',', dtype='f8')[1:]

def loadNpy(filename):
  try:
    return np.load('./'+filename+'.npy')
  except IOError:
    data = loadCSV('./'+filename+'.csv')
    np.save('./'+filename+'.npy',data)
    return data

def rotateImage(image, radius):
  return image.rotate(radius, Image.BILINEAR)

def rescaleImage(image, newsize):
  return image.resize(newsize,Image.ANTIALIAS)

def showCharacter(vector):
  image = vector2image(vector)
  image.show()

def vector2image(vector):
  return Image.fromarray(np.uint8(vector.reshape(28,28)), "L")

def image2vector(image):
  return np.array(image.getdata())
  
def predict(data, rf):
  predictions = rf.predict_proba(data)
  return [ypred.argmax() for index, ypred in enumerate(predictions)]
  
def main():
  performance = {}
  
  for nr_of_trees in range(1000, 1100, 100):
    rf = RandomForestClassifier(n_estimators=nr_of_trees, n_jobs=3)
    rf.fit(Xtrain, ytrain)
    #if Xtest
    #  prediction = np.array(predict(Xtest, rf))
    #
    #  performance[float(nr_of_trees)] = sum(prediction == ytest) * 100 / float(prediction.size)
   
    #for i, test in enumerate(prediction[prediction != ytest]):
    #  showCharacter(Xtest[i])
    
  #if Xtest
  #  print performance
  #  plt.plot(performance.keys(), performance.values(), 'ro')
  #  plt.show()
  
  prediction = predict(Xcompetition, rf)
  predicted_probs = [[index + 1, ycomp] for index, ycomp in enumerate(prediction)]
  np.savetxt('./submission.csv', predicted_probs, delimiter=',', fmt='%d,%d')

def initialize():
  global train, Xcompetition, Xtrain, ytrain, Xtest, ytest
  
  train = loadNpy('train')
  Xcompetition  = loadNpy('test')
  
  Xtrain = train[:,1:]
  ytrain = train[:,0] 
  #Xtrain = train[:train.shape[0] * 0.8,1:]
  #ytrain = train[:train.shape[0] * 0.8,0]
  #Xtest  = train[train.shape[0] * 0.8 + 1:,1:]
  #ytest  = train[train.shape[0] * 0.8 + 1:,0]

def expandSet(X, y, rotation_radius=10):
  Xrr = np.ndarray(X.shape)
  Xrl = np.ndarray(X.shape)
  yrr = np.ndarray(y.shape)
  yrl = np.ndarray(y.shape)
  
  for i in range(Xtrain.shape[0]):
    Xrr[i,:] = image2vector(rotateImage(vector2image(X[i]), rotation_radius))
    Xrl[i,:] = image2vector(rotateImage(vector2image(X[i]), -rotation_radius))
    yrr[i] = y[i]
    yrl[i] = y[i]
  
  Xshape = Xtrain.shape
  
  # below append creates a long vector, so reshape necessary
  return [np.append(X, [Xrr, Xrl]).reshape((Xshape[0] * 3, Xshape[1])), np.append(y, [yrr, yrl])]

def rescaleSet(X, size):
  Xrescaled = np.ndarray([X.shape[0], size[0] * size[1]])
  
  for i in range(X.shape[0]):
    Xrescaled[i] = image2vector(rescaleImage(vector2image(X[i]), size))
    
  return Xrescaled
  
if __name__ == '__main__':
  initialize()
  #image = vector2image(Xtest[1])
  #rotateImage(image, 10).show()
  #image.show()
  #rescaleImage(image, (20,20)).show()
  
  [Xtrain, ytrain] = expandSet(Xtrain, ytrain)
  Xtrain = rescaleSet(Xtrain, (20,20))
  #Xtest  = rescaleSet(Xtest,  (20,20))
  Xcompetition = rescaleSet(Xcompetition,  (20,20))
  main()
