import os
import requests
from splinter import Browser
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd

def scrape():
    url="https://money.cnn.com/news/specials/jobs/"
    executable_path = {'executable_path': ChromeDriverManager().install()}
    browser = Browser('chrome', **executable_path, headless=False)
    browser.visit(url)
    html=browser.html
    soup = BeautifulSoup(html, 'html.parser')
    result=soup.find_all('div',class_='cnnHeadline')
    img=soup.find_all('a',class_='summaryImg')[0].find('img')['src']
    latest=result[0].find('a').text
    url=result[0].find('a')['href']
    browser.quit()

    return{'Title':latest,'Image':img,"Url":"https://money.cnn.com/"+url}