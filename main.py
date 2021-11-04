import pandas as pd
dataPath = 'data/shortCorpus.csv'
df_uf = pd.read_csv(dataPath, sep=';')

from utilsPtBr import Utils

util = Utils()

dictDataframe = {
  "tag": [],
  "tokens": [],
  "types": [], 
  "freq": []
}

formatTag = {
  "Publicação - Despacho / Decisão": "Publicação - Despacho / Decisão",
  "Publicação - Sentença": "Publicação - Sentença",
  "Publicação - Audiência": "Publicação - Audiência",
}

# pre formatando
for i in range (len(df.index):
  tag = df_uf.loc[i].iloc[4]
  tag = formatTag.get(tag, "outros")
  
  text = df_uf.loc[i].iloc[3]
  # tokens = util.tokenize(text)
  # freq = util.getPropoFreq(tokens)
  
  dictDataframe['tag'].append(tag)
  # dictDataframe['text'].append(text)
  # dictDataframe['tokens'].append(tokens)
  # dictDataframe['freq'].append(freq)
  
  
df = pd.DataFrame(dictDataframe)
  
  

np.array([sample_text])
