import pythainlp
import tqdm
import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_columns', 1700)



def under_sample(df, label, low_label, high_label, size):

    df_under_sample = df.loc[
        np.concatenate([
            np.random.choice(
                df.loc[df[label] == low_label].index,
                size
            ),
            np.random.choice(
                df.loc[df[label] == high_label].index,
                size
            )
        ])
    ]
    return df_under_sample

def text_cleaner(text):
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z' ]|^'|'$|''")
    replaced = re.sub(pattern, '', text)
    return replaced

df = pd.read_csv('./corpus/sentiment/all.csv')

cols = ['text','type']
df = df[cols]
df = df.replace({0:-1})

print(df.shape[0])

df3 = pd.read_csv("./corpus/sentiment/IRresult.csv")
df3 = df3.drop(['id','roomid'],axis=1)

print(df3.shape[0])

testText = []
df2 = pd.read_csv("./corpus/sentiment/gadgetResult.csv")
df2 = df2.drop(['id','roomid'],axis=1)

print(df2.shape[0])

frames = [df,df3,df2]
resultDf = pd.concat(frames)




entry = []

for i, row in tqdm.tqdm(resultDf.iterrows()):
    cleaned_text = text_cleaner(row.text)
    entry.append({
        'tokens': cleaned_text,
        'type': row.type
    })

resultDf = pd.DataFrame(entry)
resultDf = resultDf.sample(frac=1).reset_index(drop=True)
vectorizer = sklearn.feature_extraction.text.CountVectorizer(tokenizer=pythainlp.tokenize.word_tokenize)


cv = StratifiedKFold(n_splits=10,shuffle=True)

for train_index, test_index in cv.split(resultDf['tokens'], resultDf['type']):
    # print("TRAIN:", train_index, "\nTEST:", test_index)
    X_train, X_test = resultDf['tokens'][train_index], resultDf['tokens'][test_index]
    y_train, y_test = resultDf['type'][train_index], resultDf['type'][test_index]
    vectorize = sklearn.feature_extraction.text.CountVectorizer(tokenizer=pythainlp.tokenize.word_tokenize)
    bagOfWords_train = vectorize.fit_transform(X_train)
    X_train = bagOfWords_train.toarray()

    bagOfWords_test = vectorize.transform(X_test)
    X_test = bagOfWords_test.toarray()

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(nb.score(X_test, y_test))



