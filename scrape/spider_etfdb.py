import pickle

import selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import os, datetime, sys
import click
import numpy as np
import copy
import pandas as pd
import time

# logger = Logger()
# logger.init_log(os.path.join(Paths.log, 'taobao_log_{}'.format(datetime.date.today())))

next_btn = '//*[@class="page-next"]/a'
header = '//*[@id="mobile_table_pills"]/div[1]/div/div[1]/table/thead'
path_chrome = r'C:\Software\webdriver\chromedriver_win32_42\chromedriver.exe'


class Spider:

    start_url = r'https://etfdb.com/screener/'
    driver = None
    scraper_name = 'taobao'
    max_listing_pages = 4000

    def __init__(s):
        s.results = []

    @staticmethod
    def get_q_id(url, token='id='):
        id_start = url.find(token)
        if id_start == -1:
            return url
        id_start += len(token)
        id_end = url[id_start:].find('&ns')
        if id_end == -1:
            return url
        return url[id_start:id_start+id_end]

    def scrape(s):
        for i in range(100000000000):
            try:
                tabs = s.driver.find_elements_by_xpath(xpath='//*[@id="screener"]/div[2]/div[2]/div/div[1]/ul/li/a')
                for tab in tabs:
                    tab.click()
                    time.sleep(2)
                    s.parse_site(tab.text)
                    with open(r'C:\repos\trade\data\etfs\list', 'wb') as f:
                        pickle.dump(s.results, f)
                tabs[0].click()
                time.sleep(2)
                s.click_next()
            except Exception as e:
                print('Error processing word {}. Traceback: {}'.format(e))
            print(i)

    def click_next(s):
        next_button = s.elem_xpath(next_btn)
        actions = ActionChains(s.driver)
        actions.pause(0.1)
        actions.click(next_button)
        actions.pause(0.3)
        actions.perform()

    def elem_xpath(s, xpath):
        try:
            return s.driver.find_element_by_xpath(xpath=xpath)
        except selenium.common.exceptions.NoSuchElementException:
            return False

    def parse_site(s, tab=None):
        r = {}
        try:
            r['header'] = [el.text for el in s.driver.find_elements_by_xpath(xpath='//*[@id="mobile_table_pills"]/div[1]/div/div[1]/table/thead/tr/th')]
            r['rows'] = []
            rows = s.driver.find_elements_by_xpath(xpath='//*[@id="mobile_table_pills"]/div[1]/div/div[1]/table/tbody/tr')
            for i in range(1, len(rows) + 1):
                r['rows'].append(
                    [el.text for el in s.driver.find_elements_by_xpath(xpath=f'//*[@id="mobile_table_pills"]/div[1]/div/div[1]/table/tbody/tr[{i}]/td')]
                )
            [el.text for el in s.driver.find_elements_by_xpath(xpath=f'//*[@id="mobile_table_pills"]/div[1]/div/div[1]/table/tbody/tr/td[1]')]
            pdf = pd.DataFrame(r['rows'], columns=r['header'])
            pdf['tab'] = tab
            s.results.append(pdf)
        except Exception as e:
            print(e)

    def start_browser(s):
        options = webdriver.ChromeOptions()
        # options.add_argument(f'load-extension={Paths.extension_directory}')
        options.add_argument("--start-maximized")
        options.add_argument('--log-level=3')
        # chrome_options.add_argument('--disable-extensions')
        # chrome_options.add_argument('--profile-directory=Default')
        # options.add_argument("--incognito")
        options.add_argument("--disable-plugins-discovery")
        # # chrome_options.add_experimental_option("excludeSwitches",
        # #                                        ["ignore-certificate-errors",
        # #                                         "safebrowsing-disable-download-protection",
        # #                                         "safebrowsing-disable-auto-update",
        # #                                         "disable-client-side-phishing-detection"]
        # #                                        )
        s.driver = webdriver.Chrome(path_chrome, chrome_options=options)
        s.driver.get(s.start_url)


def run():
    inst = Spider()
    inst.start_browser()
    # inst.manual_login_confirm()
    inst.scrape()
    inst.driver.close()


if __name__ == '__main__':
    run()

