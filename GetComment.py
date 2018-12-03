import requests
from bs4 import BeautifulSoup
import re

import html

Room = 'food'

BASEURL = "https://pantip.com/"
STARTURL = "https://pantip.com/forum/"+Room
processQueue = []
dataReturn = []

def getComment(id,type):
    params = {"tid": id,
              "type": type}

    with requests.Session() as s:
        s.headers.update({
                             "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/53.0.2785.143 Safari/537.36",
                             "X-Requested-With": "XMLHttpRequest"})
        r = (s.get("http://pantip.com/forum/topic/render_comments", params=params))
        data = r.json()
    try:
        return data["comments"]
    except:
        return []

def cleanlink(raw_html):
    cleanr = re.compile('<a.*?>.*?</a>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def cleanhtml(raw_html):
  raw_html = cleanlink(raw_html)
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext

def removeWhitespace(text):
    pattern = re.compile(r'\s+')
    sentence = re.sub(pattern, '', text)
    return sentence

def removeSpecialCharacter(text):
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text

def parse_list_page(url):
    global  data
    data = requests.get(url)
    soup = BeautifulSoup(data.content,'html.parser')

    # add next page link in to the queue
    links = soup.select('a[rel="next"]')
    nextlink = links[0].attrs['href']
    processQueue.append(
        (parse_list_page, BASEURL + nextlink)
    )

    TitleCols = soup.find("div", {"class": "post-list-wrapper"})
    titles = TitleCols.find_all("div", {"class": "post-item-title"})

    f = open("food.csv", "a+",encoding="utf-8")
    for title in titles:
        titleLink = title.find("a", href=True)
        Id = titleLink['href'].replace("/topic/","")

        page = requests.get("https://pantip.com/topic/"+Id)
        pageHtml = BeautifulSoup(page.content, 'html.parser')
        typehtml = pageHtml.find("input",{"id": "topic-type"})

        comments = getComment(Id,typehtml['value'])
        print(Id)
        for comment in comments:
            # print(html.unescape(cleanhtml(comment["message"])))
            if html.unescape(removeSpecialCharacter(removeWhitespace(cleanhtml(comment["message"])))):
                f.write(str(comment['_id'])+","+str(Id)+","+html.unescape(removeSpecialCharacter(removeWhitespace(cleanhtml(comment["message"]))+"\n")))
                dataReturn.append({
                    'id':comment['_id'],
                    'roomId': Id,
                    'message': comment['message']
                })
                print(len(dataReturn))

    f.close()






def main():

    processQueue.append(
        (parse_list_page, STARTURL)
    )

    while len(dataReturn) < 10000000:
        call_back, url = processQueue.pop(0)
        call_back(url)



if __name__ == '__main__':
    main()