
import re
import string
import nltk
# from nltk.stem import WordNetLemmatizer

stopwords = nltk.corpus.stopwords.words('portuguese')
ponctuation = string.punctuation


class Utils():
    
  def getBagOfSentencesByRegex(self, text):
    return re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s", text)

    # for i in range (len(sentences)):
      

    # print (sentences)
    
  def tokenize(self, text):
    textFormated = text.lower()
    tokens = []
    for token in nltk.word_tokenize(textFormated):
      tokens.append(token.lower())
    tokens = [word for word in tokens if word not in stopwords and word not in ponctuation]
    return tokens
  
  def getPropoFreq(self, tokens):
    freq = nltk.FreqDist(tokens)
    maxKey = freq.max()
    maxFreq = freq[maxKey]
    
    for word in freq.keys():
      freq[word] = freq[word]/maxFreq
    return (list(freq))
    
    
  def lematize(self, text):
    textFormated = text.lower()
    sents = nltk.sent_tokenize(textFormated)
    # print(sents)
    return sents
    
    
    
