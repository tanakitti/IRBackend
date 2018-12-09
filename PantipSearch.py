import traceback
import string
import requests
from bs4 import BeautifulSoup
import re
import time
import pickle
import pythainlp
from pythainlp.sentiment import sentiment
import json

posts = []
filename = "./clawler/model/qusModel.sav"
loaded_model = pickle.load(open(filename, 'rb'))
filename2 = "./clawler/model/senModel.sav"
loaded_model2 = pickle.load(open(filename2, 'rb'))

vec_filename = "./clawler/vec/qusVec.plk"
with open(vec_filename, 'rb') as file:
    pickle_vector = pickle.load(file)

vec_filename2 = "./clawler/vec/senVec.plk"
with open(vec_filename2, 'rb') as file2:
    pickle_vector2 = pickle.load(file2)


def text_cleaner(text):
    pattern = re.compile(r"[^\u0E00-\u0E7Fa-zA-Z1-9]")
    replaced = re.sub(pattern, '', text)
    tokens = pythainlp.tokenize.word_tokenize(replaced, engine='newmm')
    tokens = [token.replace(' ', '') for token in tokens]
    tokens = list(filter(lambda token: token != '', tokens))
    clean = ""
    for token in tokens:
        clean += token
    return clean


def cleanhtml(raw_html):
    raw_html = raw_html.replace("</em>", "")
    raw_html = raw_html.replace("<em>", "")
    return raw_html


def getPage(id, keyword):
    comments = requests.get(
        "https://pantip.com/forum/topic/render_comments?tid=" + id + "&type=3",
        headers={"X-Requested-With": "XMLHttpRequest"}
    )

    comments = json.loads(comments.text.replace("ï»¿", ""))
    # comments = comments['comments']

    if ("comments" in comments):

        for comment in comments['comments']:

            type1 = ""
            cleaned_text = text_cleaner(comment['message'])

            bagOfWords = pickle_vector.transform([cleaned_text])
            Test = bagOfWords.toarray()

            if loaded_model.predict(Test) == 0:
                bagOfWords2 = pickle_vector2.transform([cleaned_text])
                Test2 = bagOfWords2.toarray()

                if loaded_model2.predict(Test2) == 1:
                    type1 = "pos"
                elif loaded_model2.predict(Test2) == 0:
                    type1 = "nue"
                else:
                    type1 = "neg"

            else:
                type1 = "ques"

            if comment['message'].strip() != "":
                posts.append({
                    "tag": "comment",
                    "id": id,
                    "text": comment['message'],
                    "type": type1
                })
            else:
                print("sad")


def get_stores_info(page, keyword):
    global posts
    posts = []

    data = {
        "inputtext": keyword,
        "page": page
    }
    response = requests.post(
        "https://pantip.com/search/search/get_search", data=data)

    # position 0 is topic_id position 1 text
    for post in response.json()['data']:
        title = cleanhtml(post['title'])
        id = post['topic_id']

        type1 = ""
        cleaned_text = text_cleaner(title)

        bagOfWords = pickle_vector.transform([cleaned_text])
        Test = bagOfWords.toarray()
        # print(loaded_model.predict_proba(Test))
        if loaded_model.predict(Test) == 0:
            bagOfWords2 = pickle_vector2.transform([cleaned_text])
            Test2 = bagOfWords2.toarray()

            # print(loaded_model2.predict_proba(Test2))
            if loaded_model2.predict(Test2) == 1:
                type1 = "pos"
            elif loaded_model2.predict(Test2) == 0:
                type1 = "nue"
            else:
                type1 = "neg"

        else:

            type1 = "ques"

        if title.strip() != "":
            posts.append({
                "tag": "title",
                "id": id,
                "room": id,
                "text": title,
                "type": type1
            })
        else:
            print("sad")

        getPage(id, keyword)

    return posts