
import sklearn as skl
#from sklearn import model_selection
from sklearn import linear_model
import matplotlib.pyplot as plt
#from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import classification_report, plot_confusion_matrix

# Predict

# Confusion Matrix


class SklBoW:
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