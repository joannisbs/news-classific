
import sklearn as skl
#from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
#from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import classification_report, plot_confusion_matrix

# Predict

# Confusion Matrix


class SklClass:
  def fitLinearRegretion(self, trainX, trainY):
    self.__model_LR = skl.linear_model.LogisticRegression()
    self.__model_LR.fit(trainX, trainY)
    
  def testLinearRegretion(self, testX, testY):
    predictions = self.__model_LR.predict(testX)
    score = self.__model_LR.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_LR, testX, testY)  
    plt.show()
    
  def fitRandomForest(self, trainX, trainY):
    self.__model_RF = RandomForestClassifier()
    self.__model_RF.fit(trainX, trainY)
    
  def testRandomForest(self, testX, testY):
    predictions = self.__model_RF.predict(testX)
    score = self.__model_RF.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_RF, testX, testY)  
    plt.show()
    
        
  def fitLogisticRegression(self, trainX, trainY):
    self.__model_LogR = LogisticRegression()
    self.__model_LogR.fit(trainX, trainY)
    
  def testLogisticRegression(self, testX, testY):
    predictions = self.__model_LogR.predict(testX)
    score = self.__model_LogR.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_LogR, testX, testY)  
    plt.show()
    
  def fitExtraTreesClassifier(self, trainX, trainY):
    self.__model_ETC = ExtraTreesClassifier()
    self.__model_ETC.fit(trainX, trainY)
    
  def testExtraTreesClassifier(self, testX, testY):
    predictions = self.__model_ETC.predict(testX)
    score = self.__model_ETC.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_ETC, testX, testY)  
    plt.show()
    
  def fitDecisionTreeClassifier(self, trainX, trainY):
    self.__model_DTC = DecisionTreeClassifier()
    self.__model_DTC.fit(trainX, trainY)
    
  def testDecisionTreeClassifier(self, testX, testY):
    predictions = self.__model_DTC.predict(testX)
    score = self.__model_DTC.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_DTC, testX, testY)  
    plt.show()
    
  def fitKNeighborsClassifier(self, trainX, trainY):
    self.__model_KNN = KNeighborsClassifier()
    self.__model_KNN.fit(trainX, trainY)
    
  def testKNeighborsClassifier(self, testX, testY):
    predictions = self.__model_KNN.predict(testX)
    score = self.__model_KNN.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_KNN, testX, testY)  
    plt.show()
    
  def fitBaggingClassifier(self, trainX, trainY):
    self.__model_Bag = BaggingClassifier(KNeighborsClassifier(),
      max_samples=0.5, max_features=0.5)
    
    self.__model_Bag.fit(trainX, trainY)
    
  def testBaggingClassifier(self, testX, testY):
    predictions = self.__model_Bag.predict(testX)
    score = self.__model_Bag.score(testX, testY)
    print(f'score:      { score*100:.3f}%')
  
    # precision, recall, support and f1-score
    print(f'{classification_report(testY, predictions)}')
    plot_confusion_matrix(self.__model_Bag, testX, testY)  
    plt.show() 
    
    
    
    
    
    
    
    
    
