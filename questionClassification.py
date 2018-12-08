import pythainlp
import tqdm
import re
import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
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
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z1-9]")
    replaced = re.sub(pattern, '', text)
    tokens = pythainlp.tokenize.word_tokenize(replaced, engine='newmm')
    tokens = [token.replace(' ', '') for token in tokens]
    tokens = list(filter(lambda token: token != '', tokens))
    clean = ""
    for token in tokens:
        clean+=token
    return clean

df = pd.read_csv('./corpus/question/pantip.csv')

df['type'] = np.where(df['type']=='question', 1, 0)
df.drop(['date','tags'], axis=1, inplace=True)

cols = ['id','title','type']
df = df[cols]
df = under_sample(df, label='type', low_label=0, high_label=1, size=30000)

entry = []

for i, row in tqdm.tqdm(df.iterrows()):
    cleaned_text = text_cleaner(row.title)
    entry.append({
        'id': row.id,
        'tokens': cleaned_text,
        'type': row.type
    })

df = pd.DataFrame(entry)
print(df.shape[0])

cv = StratifiedKFold(n_splits=10,shuffle=True)

for train_index, test_index in cv.split(df['tokens'], df['type']):

    X_train, X_test = df['tokens'][train_index], df['tokens'][test_index]
    y_train, y_test = df['type'][train_index], df['type'][test_index]


    vectorize = sklearn.feature_extraction.text.CountVectorizer(tokenizer=pythainlp.tokenize.word_tokenize)
    bagOfWords_train = vectorize.fit_transform(X_train)
    X_train = bagOfWords_train.toarray()

    bagOfWords_test = vectorize.transform(X_test)
    X_test = bagOfWords_test.toarray()

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    print(nb.score(X_test, y_test))
