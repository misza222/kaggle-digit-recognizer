from sklearn.ensemble import RandomForestClassifier

def RF_predict(rf, data):
  predictions = rf.predict_proba(data)
  return [ypred.argmax() for index, ypred in enumerate(predictions)]
  
def RF_learn(Xtrain, ytrain, nr_of_trees=10, n_jobs=-1):
  rf = RandomForestClassifier(n_estimators=nr_of_trees, n_jobs=n_jobs)
  rf.fit(Xtrain, ytrain)
  return rf

