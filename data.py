import numpy as np
import Image # for visualizing images of the digits
import itertools

def loadCSV(filename):
  return np.genfromtxt(open(filename, 'r'), delimiter=',', dtype='f8')[1:]

def loadData(filename):
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
  
def generatePolynomialFeatures(X, degree=2):
  polysize = 0
  for i in itertools.combinations_with_replacement(range(X.shape[1]), degree):
    polysize += 1
    
  Xout = np.ndarray((X.shape[0], polysize))
    
  for i in range(X.shape[0]):
    for j, combinations in enumerate(itertools.combinations_with_replacement(range(X.shape[1]), degree)):
      for index in combinations:
        if Xout[i][j] == 0:
          Xout[i][j] = X[i][index]
        else:
          Xout[i][j] *= X[i][index]
    
  return Xout

