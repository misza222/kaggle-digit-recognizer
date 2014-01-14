import numpy as np
import matplotlib.pyplot as plt

import data
import rf
import lr

def main():
  performance = {}
  
  # Best results so far with rescaling to (9,9) and polynomial features degree=2
  line = lr.LR_learn(data.generatePolynomialFeatures(Xtrain,2), ytrain)
  prediction = lr.LR_predict(line, data.generatePolynomialFeatures(Xtest,2))
  print sum(prediction == ytest) * 100 / float(prediction.size)
  
#  for nr_of_trees in range(100, 200, 100):
    
#    forest = rf.RF_learn(Xtrain, ytrain, nr_of_trees, -1)
    #if Xtest
    #  prediction = np.array(rf.RF_predict(forest, Xtest))
    #
    #  performance[float(nr_of_trees)] = sum(prediction == ytest) * 100 / float(prediction.size)
   
    #for i, test in enumerate(prediction[prediction != ytest]):
    #  showCharacter(Xtest[i])
    
  #if Xtest
  #  print performance
  #  plt.plot(performance.keys(), performance.values(), 'ro')
  #  plt.show()
  
#  prediction = rf.RF_predict(forest, Xcompetition)
#  predicted_probs = [[index + 1, ycomp] for index, ycomp in enumerate(prediction)]
#  np.savetxt('./submission.csv', predicted_probs, delimiter=',', fmt='%d,%d')

def initialize():
  global train, Xcompetition, Xtrain, ytrain, Xtest, ytest
  
  train = data.loadData('train')
  Xcompetition  = data.loadData('test')
  
  #Xtrain = train[:,1:]
  #ytrain = train[:,0] 
  Xtrain = train[:train.shape[0] * 0.8,1:]
  ytrain = train[:train.shape[0] * 0.8,0]
  Xtest  = train[train.shape[0] * 0.8 + 1:,1:]
  ytest  = train[train.shape[0] * 0.8 + 1:,0]


if __name__ == '__main__':
  initialize()
  #image = data.vector2image(Xtest[1])
  #data.rotateImage(image, 10).show()
  #data.image.show()
  #data.rescaleImage(image, (20,20)).show()
  
  #[Xtrain, ytrain] = data.expandSet(Xtrain, ytrain)
  Xtrain = data.rescaleSet(Xtrain, (9,9))
  Xtest  = data.rescaleSet(Xtest,  (9,9))
  #Xcompetition = data.rescaleSet(Xcompetition,  (20,20))
  main()
