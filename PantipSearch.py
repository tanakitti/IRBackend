import traceback
import string
import requests
from bs4 import BeautifulSoup
import re
import time
import pickle
import pythainlp
from pythainlp.sentiment import sentiment
posts = []
filename = "./model/qestion.sav"
loaded_model = pickle.load(open(filename, 'rb'))
filename2 = "./model/sentiment.sav"
loaded_model2 = pickle.load(open(filename2, 'rb'))

vec_filename ="./vec/qesVec.pkl"
with open(vec_filename,'rb') as file:
    pickle_vector=pickle.load(file)

vec_filename2 = "./vec/sentiment.pkl"
with open(vec_filename2, 'rb') as file2:
    pickle_vector2 = pickle.load(file2)

def text_cleaner(text):
    table = str.maketrans('', '', string.punctuation.replace('?', ''))
    tmp =  re.sub(r'\d+', '', text)
    tmp = tmp.translate(table).lower()
    return tmp


def tokenizer(text):
    tokens = pythainlp.tokenize.word_tokenize(text, engine='newmm')
    tokens = [token.replace(' ','') for token in tokens]
    tokens = list(filter(lambda token: token != '', tokens))
    stringToken = ""
    for token in tokens:
        stringToken += token+" "
    return stringToken

def cleanhtml(raw_html):

    raw_html = raw_html.replace("</em>","")
    raw_html = raw_html.replace("<em>", "")
    return raw_html

def getPage(id):

    page = requests.get("https://pantip.com/topic/" + id)
    pageHtml = BeautifulSoup(page.content, 'html.parser')
    typehtml = pageHtml.find("input", {"id": "topic-type"})

    params = {"tid": id,
              "type": typehtml}

    with requests.Session() as s:
        s.headers.update({
                             "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
                             "X-Requested-With": "XMLHttpRequest"})
        r = (s.get("http://pantip.com/forum/topic/render_comments", params=params))
        data = r.json()
        try:


            comments = data['comments']



            for comment in comments:

                type1 = ""
                cleaned_text = text_cleaner(comment['message'])
                tokens = tokenizer(cleaned_text)

                bagOfWords = pickle_vector.transform([tokens])
                Test = bagOfWords.toarray()

                if loaded_model.predict(Test) == 0:
                    bagOfWords2 = pickle_vector2.transform([tokens])
                    Test2 = bagOfWords2.toarray()

                    if loaded_model2.predict(Test2) == 1:
                        type1 = "pos"
                    elif loaded_model2.predict(Test2) == 0:
                        type1 = "nue"
                    else:
                        type1 = "neg"


                else:
                    type1 = "ques"



                posts.append({
                    "tag": "comment",
                    "id": comment['_id'],
                    "room": id,
                    "text": comment['message'],
                    "type": type1
                })
        except Exception:
            print(traceback.print_exc())
            print("no")


def get_stores_info(page):

    global posts
    data = {
        "inputtext": "nestle",
        "page": page
    }
    response = requests.post("https://pantip.com/search/search/get_search", data=data)

    # position 0 is topic_id position 1 text
    for post in response.json()['data']:
        title = cleanhtml(post['title'])
        id = post['topic_id']

        type1 = ""
        cleaned_text = text_cleaner(title)
        tokens = tokenizer(cleaned_text)

        bagOfWords = pickle_vector.transform([tokens])
        Test = bagOfWords.toarray()

        if loaded_model.predict(Test) == 0:
            bagOfWords2 = pickle_vector2.transform([tokens])
            Test2 = bagOfWords2.toarray()

            if loaded_model2.predict(Test2) == 1:
                type1 = "pos"
            elif loaded_model2.predict(Test2) == 0:
                type1 = "nue"
            else:
                type1 = "neg"


        else:
            type1 = "ques"
        posts.append({
            "tag":"title",
            "id":id,
            "room":id,
            "text":title,
            "type":type1
        })
        print(post['topic_id'])
        getPage(post['topic_id'])
    return len(posts)


def main():

    i = 0;
    while len(posts) < 20:
        i+=1
        time.sleep(1)
        print(get_stores_info(i))
    print(posts)


if __name__ == '__main__':
    main()