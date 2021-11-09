import pandas as pd
dataPath = 'data/Pasta3.csv'
df_uf = pd.read_csv(dataPath, sep=';')

import random

dictDataframe = {
  "tag": [],
  "text": [], 
}

formatTag = {
  "Publicação - Despacho / Decisão": "Publicação - Despacho / Decisão",
  "Publicação - Sentença": "Publicação - Sentença",
  "Publicação - Audiência": "Publicação - Audiência",
}

formatTagValue = {
  "Publicação - Despacho / Decisão": 3,
  "Publicação - Sentença": 7,
  "Publicação - Audiência": 24,
}

for i in range (102345):
  tag = df_uf.loc[i].iloc[4]
  tag = formatTag.get(tag, "outros")
  
  valueRandom = random.randint(1,100)
  
  text = df_uf.loc[i].iloc[3]
  
  if text == 'Err:510':
    print('removendo')
    
  elif valueRandom < formatTagValue.get(tag, 200):
  
    dictDataframe['tag'].append(tag)
    dictDataframe['text'].append(text)
  
df = pd.DataFrame(dictDataframe)
  
dictValue = {}

for i in range (len(df.index)):
  if df.loc[i].iloc[0] in dictValue:
    dictValue[df.loc[i].iloc[0]] = dictValue[df.loc[i].iloc[0]] + 1;
  else:
    dictValue[df.loc[i].iloc[0]] = 1;
  
df.to_csv("data/shortCorpus2.csv", sep=';')

for i in dictValue:
  if(dictValue[i] > 300):
    print(i, dictValue[i])
