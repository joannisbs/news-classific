
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, plot_confusion_matrix
from mlxtend.plotting import plot_decision_regions
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image

class SKLRandomForestClassifier:
  def fit(self, trainX, trainY):
    self.__model_RF = RandomForestClassifier()
    self.__model_RF.fit(trainX, trainY)
    
  def test(self, testX, testY):
    predictions = self.__model_RF.predict(testX)
    score = self.__model_RF.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    
  def getConfusionMatrix(self, testX, testY):
    plot_confusion_matrix(self.__model_RF, testX, testY)  
    plt.show()

  def getDecisionTree(self, vocabulary):
    estimator = self.__model_RF.estimators_[5]
    export_graphviz(estimator, out_file='tree.dot',
                class_names = ['outros', 'decisao', 'sentença', 'audiencia'],
                feature_names = list(vocabulary),
                max_depth = 15,
                rounded = True, proportion = False, 
                precision = 2, filled = True)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    


class SKLLogisticRegression:
  def fit(self, trainX, trainY):
    self.__model_LogR = LogisticRegression()
    self.__model_LogR.fit(trainX, trainY)
    
  def test(self, testX, testY):
    predictions = self.__model_LogR.predict(testX)
    score = self.__model_LogR.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    
  def getConfusionMatrix(self, testX, testY):
    plot_confusion_matrix(self.__model_LogR, testX, testY)  
    plt.show()
    
class SKLExtraTreesClassifier:  
  def fit(self, trainX, trainY):
    self.__model_ETC = ExtraTreesClassifier()
    self.__model_ETC.fit(trainX, trainY)
    
  def test(self, testX, testY):
    predictions = self.__model_ETC.predict(testX)
    score = self.__model_ETC.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    
  def getConfusionMatrix(self, testX, testY):
    plot_confusion_matrix(self.__model_ETC, testX, testY)  
    plt.show()
    

class SKLDecisionTreeClassifier: 
  def fit(self, trainX, trainY):
    self.__model_DTC = DecisionTreeClassifier()
    self.__model_DTC.fit(trainX, trainY)
    
  def test(self, testX, testY):
    predictions = self.__model_DTC.predict(testX)
    score = self.__model_DTC.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')

    
  def getConfusionMatrix(self, testX, testY):
    plot_confusion_matrix(self.__model_DTC, testX, testY)  
    plt.show()
    

    
class SKLKNeighborsClassifier:    
  def fit(self, trainX, trainY):
    self.__model_KNN = KNeighborsClassifier()
    self.__model_KNN.fit(trainX, trainY)
    
  def test(self, testX, testY):
    predictions = self.__model_KNN.predict(testX)
    score = self.__model_KNN.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')


  def getConfusionMatrix(self, testX, testY):
    plot_confusion_matrix(self.__model_KNN, testX, testY)  
    plt.show()
    

    
    
    
    
    
    
    
