from urllib.request import urlopen
from bs4 import BeautifulSoup
import re
import datetime
import random 
import pymysql

conn = pymysql.connect(host='locahost',unix_socket='/tmp/mysql.sock',user='root',passwd=None, db='mysql', charset='utf8')

cur = conn.cursor()
cur.execute("USE wikipedia")

def getUrl(pageId):
    cur.execute("SELECT url FROM pages WHERE id = %s", (int(pageId)))
    if cur.rowcount == 0:
        return None
    return cur.fetchone()[0]

def getLinks(fromPageId):
    cur.execute("SELECT toPageId FROM links WHERE fromPageId = %s", (int(fromPageId)))
    if cur.rowcount == 0:
        return None
    return [x[0] for x in cur.fetchall()]

def searchBreadth(targetPageId, currentPageId, depth, nodes): #breadth=width
    if nodes is None or len(nodes) == 0:
        return None
    if depth <= 0:
        for node in nodes:
            if node == targetPageId:
                return [node]
        return None
    #depth is bigger 0, deeper!
    for node in nodes:
        found = searchBreadth(targetPageId, node, depth-1, getLinks(node))
        if found is not None:
            return found.append(currentPageId)
    return None

nodes = getLinks(2)
targetPageId = 200436 
print(nodes)

for i in range(0,5):
    found = searchBreadth(targetPageId, 2, i, nodes)
    if found is not None:
        for node in found:
            print(getUrl(node))
        break

print('-------supreme---------')

'''#ALTER TABLE pages CONVERT TO CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci; 
#ALTER TABLE pages CONVERT TO CHARACTER SET = utf8mb4 COLLATE = utf8mb4_unicode_ci;


def insertPageIfNotExists(url): #write links that does not exists
    cur.execute("SELECT * FROM pages WHERE url = %s", (url))
    if cur.rowcount==0:
        cur.execute("INSERT INTO pages (url) VALUES (%s)",(url))
        conn.commit()
        return cur.lastrowid
    else:
        return cur.fetchone()[0]
        
def insertLink(fromPageId, toPageId): #group links
    cur.execute("SELECT * FROM links WHERE fromPageId = %s AND toPageId = %s",(int(fromPageId), int(toPageId))) 
    if cur.rowcount==0:
        cur.execute("INSERT INTO links (fromPageId, toPageId) VALUES (%s, %s)",(int(fromPageId), int(toPageId))) 
        conn.commit()

pages = set()
def getLinks(pageUrl, recursionLevel): 
    global pages
    if recursionLevel > 4:
        return;
    pageId = insertPageIfNotExists(pageUrl)
    html = urlopen("https://en.wikipedia.org"+pageUrl)
    bsObj = BeautifulSoup(html, 'lxml')
    for link in bsObj.findAll("a", href=re.compile("^(/wiki/)((?!:).)*$")):
        insertLink(pageId, insertPageIfNotExists(link.attrs['href'])) 
        if link.attrs['href'] not in pages:
            #find new page , add & find links 
            newpage = link.attrs['href']
            
            pages.add(newpage)
            recursionLevel+1
            getLinks(newpage, recursionLevel+1)

pageUrl="/wiki/Rick_Owens"
getLinks(pageUrl, 4)
cur.close()
conn.close()

#cur.execute("SELECT * FROM pages WHERE id=5")
#print(cur.fetchone())'''
print('-------supreme!---------')
