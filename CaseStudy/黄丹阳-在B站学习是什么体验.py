# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:17:42 2019

@author: 58454
"""
import requests
import pandas as pd
import random
import re
import os 
import sys
import json
import time
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool as ThreadPool

url = 'http://api.bilibili.com/x/web-interface/newlist?rid=39&pn={pn}&ps=50'

headers={
"Host": "api.bilibili.com",
"Accept-Encoding": "gzip, deflate, br",
"Accept-Language": "zh-CN,zh;q=0.9",
"Connection": "keep-alive",
"Cookie":'l=v; CURRENT_FNVAL=16; buvid3=4DED1779-B0C8-4A01-9CD2-8D494D34D9EF65971infoc; stardustvideo=1; gr_user_id=db8d2080-4445-4f46-83a0-58d4a2586f68; grwng_uid=0c255412-6555-447d-971e-7f4bfe6c3a5c; fts=1554527042; LIVE_BUVID=AUTO5215545270503560; UM_distinctid=169f1089bcb53f-05dce500c9052e-b781636-e1000-169f1089bcc518; sid=i24lr5pt; rpdid=|(JYYRR~mRlY0J\'ullYlY|Yu~; _uuid=50232BAF-AD7B-6F94-177F-D91B8A09D9A848024infoc; finger=17c9e5f5; im_notify_type_5000649=0; CURRENT_QUALITY=32; DedeUserID=5000649; DedeUserID__ckMd5=cd1ff8f7f773bcbe; SESSDATA=db39c93d%2C1559728900%2Cf3047551; bili_jct=547dd707c646fd58869a64b32d5e8d2d; bp_t_offset_5000649=251719683803694025; arrange=list; _dfcaptcha=4d46cf504d36a7ea881ed261b7d7c54f',
"User-Agent": "Mozilla/5.0 (Windows NT 10.0;"\
"Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36"
}
#urllist =[]
#for i in range(1,51):
#    urlnew = url+str(i)
#    urllist.append(urlnew)
#    
#for i in range(10420,10000,-1):
#   urlnew = url.format(pn=i)
#   urllist.append(urlnew)
#   
 
## 爬取排名信息  
#%%
def get_data():
    interval = random.random()*5 #时间间隔
    time.sleep(interval)
    aids = []
    dists = []
    titles = []
    describes = []
    danmus = []
    watches = []
    times = []
    ups = []
    for i in range(1,51):
        # totalrank,stow,clik,dm
        url = 'https://search.bilibili.com/all?keyword=%E8%AF%BE%E7%A8%8B&from_source=banner_search&spm_id_from=333.334.b_62616e6e65725f6c696e6b.1&order=stow&duration=4&tids_1=36&single_column=1&page='+str(i)
        req = requests.get(url) #一页
        if req.status_code == 200:
            data = BeautifulSoup(req.text)#解析,提取里面的aid值
            avs = data.find_all('span','type avid')
            distpage = data.find_all('span','type hide') #分区
            titlepage = data.find_all('a','title') #题目名字
            despage = data.find_all('div','des hide') #视频描述
            watchpage = data.find_all('span','so-icon watch-num') #观看
            danmupage = data.find_all('span','so-icon hide') #弹幕
            timepage = data.find_all('span','so-icon time') #上传时间
            uppage = data.find_all('span',attrs={'title':'up主'}) #up主
        for i in range(20):
            aids+=re.findall('\d+',avs[i].text)
            dists.append(distpage[i].text)
            titles.append(titlepage[i].text)
            describes.append(despage[i].text)
            watches.append(watchpage[i].text)
            danmus.append(danmupage[i].text)
            times.append(timepage[i].text)
            ups.append(uppage[i].text)
    df = pd.DataFrame([aids,dists,titles,describes,watches,danmus,times,ups])
    return df
#%%
#爬取静态信息
def get_data(aids):
    for aid in aids:
        interval = random.random()*5 #时间间隔
        time.sleep(interval)
        url = 'http://api.bilibili.com/archive_stat/stat?aid=' + str(aid)
        req = requests.get(url) #一页aid 52931428

        if req.status_code == 200:
            data = json.loads(req.text)
            if data is not None:
                content = {}
                content['aid'] = data['data']['aid']
                content['copyright'] = data['data']['copyright']
                content['his_rank'] = data['data']['his_rank'] #历史排名
                content['now_rank'] = data['data']['now_rank'] #目前排名
                content['share'] = data['data']['share'] #分享
                content['reply'] = data['data']['reply']
                content['favorite'] = data['data']['favorite']
                content['coin'] = data['data']['coin'] #硬币数
                content['like'] = data['data']['like'] #点赞
                study.append(content) #一个视频的信息
        
#%%
#def write_to_file(comic_list):
#    with open(r'..\result\bilibili-comic.csv', 'w', newline='', encoding='utf-8') as f:
#        fieldnames = ['aid', 'title', 'view', 'danmaku', 'reply', 'favorite', 'coin']
#        writer = csv.DictWriter(f, fieldnames=fieldnames)
#        writer.writeheader()
#        try:
#            writer.writerows(comic_list)
#        except Exception as e:
#            print(e)
#%%          
#aids = get_aids()           
#study = []
#pool = ThreadPool(4)
#pool.map(get_data,aids)
#pool.close()
#a = []       
#for diction in study:
#    if diction['code'] == 0:
#        a.append(diction)     
        
df = get_data()
df = df.T
df.columns = ['aid','分区','题目','描述','观看次数','弹幕数','上传时间','up主']
df = df.drop_duplicates()
#df['排名'] = df.index
df.to_excel('收藏排名.xlsx')
#%%
# 读取数据
import numpy as np
tot = pd.read_excel('综合排名.xlsx')
click = pd.read_excel('点击排名.xlsx')
#dm = pd.read_excel('弹幕排名.xlsx')

# 增加新列
click['排名'] = np.zeros(len(click))
df['排名'] = np.zeros(len(df))
dm['排名'] = np.zeros(len(dm))
#%%
# 手动对data进行去重
#data = pd.concat([tot,click,df,dm])
#rank = []
#for i in range (1000):
#    rank[i] = rank[i] +1
#
#data = data.drop_duplicates(['aid'])
#for i in range(len(data)-len(rank)):
#    rank.append(0)
#data['排名'] = rank
#data['aid'] = pd.to_numeric(data['aid'])
#data = pd.merge(data,study_df, how= 'inner',on = 'aid')
data = data.drop_duplicates(['aid'])
#%%
aids = data['aid']
# 爬取并存储静态课程信息
get_data(aids)
study_df = pd.DataFrame(study)
study_df = study_df.drop_duplicates('aid')
study_df.to_excel('课程信息.xlsx',index = True,encoding = 'utf-8' )
#%%

for i in range(2819):
    if str(aids[i]) != str(study_df['aid'][i]):
        print('======'+str(i)+'=======')
        print('aid:'+str(aids[i]))
        print('df:'+str(study_df['aid'][i]))
        break

#%%
# merge表格 
#data = pd.read_excel('B站课程.xlsx')
import numpy as np

def find_course(x):
    return (str(x).find('大学'))


#find('人民',data['题目'])
def nplog(x):
    if x <= 0:
        return 0
    else:
        return log(x)
log = np.vectorize(nplog)
import math
math.log

def find_course(x,wordl):
    for word in wordl:
        if str(x).find(word):
            return 1
    return 0
#data['深度学习'] = data['题目'].map(lambda x: find_course(x,['深度学习','机器学习']))


import matplotlib.pyplot as plt
a = np.array([[1,2,3,4],[2,3,4,5]])
a2 = np.dot(a,a.T)
np.diag(a2)
def varimax(Phi, gamma = 1.0, q =20, tol = 1e-6): #定义方差最大旋转函数
    p,k = Phi.shape #给出矩阵Phi的总行数，总列数
    R = np.eye(k) #给定一个k*k的单位矩阵
    d=0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u,s,vh = np.linalg.svd(np.dot(Phi.T,np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T,Lambda)))))) 
        R = np.dot(u,vh)#构造正交矩阵R
        d = np.sum(s)#奇异值求和
        if d_old!=0 and d/d_old < 1+tol:
            break
    return np.dot(Phi, R)#返回旋转矩阵Phi*R

varimax(a)