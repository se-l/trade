from selenium import webdriver
import ctypes
# from selenium.webdriver.chrome.options import Options
import time
import datetime
import os
import shutil
import numpy as np
from bs4 import BeautifulSoup as BSoup
from utils.utilFunc import createDir
from scrape.bitfinex_currencies import currencies

# Bitfinex notes. When e.g. July 30 is selected in the calendar, the received file contains trade data
# from July 29. Hence uploading up to today's date. The format for QC is different. QC file of July 30 contains trade
# data from julty 30

class NoSuchElementException(Exception):
    def __init__(self, message, errors):

        # Call the base class constructor with the parameters it needs
        super().__init__(message)

def get_times(driver):
    time.sleep(.5)
    bs_obj = BSoup(driver.page_source, 'html.parser').findAll("div", {"class": "date-picker-wrapper two-months"})
    bs_obj = BSoup(driver.page_source, 'html.parser').find_all("div", {"class": ["date-picker-wrapper", "two-months"]})
    # if type(bs_obj) is not list:
    #     bs_obj = [bs_obj]
    bs_obj = [o for o in bs_obj if 'display: none' not in o['style']][0]
    return [int(t['time']) for t in bs_obj.find_all(time=True)]

def test_exec(func, val, reps=10, timeout=1):
    for i in np.arange(reps):
        res = func(val)
        if len(res) == 0:
            print('slept {}'.format(i*timeout))
            time.sleep(timeout)
        elif len(res) > 1:
            print('multiple options to click')
        else:
            return res[0].click()
    func(val).click()


def timed_exec(func, reps=3, timeout=.5):
    time.sleep(timeout)
    for i in np.arange(reps):
        try:
            a = func()
            return a
        except NoSuchElementException:
            print('sleeping')
            time.sleep(timeout)
    return

def timed_path_check(fn, reps=3, timeout=.5):
    for i in np.arange(reps):
        a = os.path.exists(fn)
        if a:
            return a
        time.sleep(timeout)
    return a

driver = webdriver.Chrome(r'C:\Users\seb\chromedriver_win32\chromedriver.exe')
driver.get(r'https://www.bitfinex.com')

init_start = (datetime.datetime(2018, 8, 1) - datetime.datetime(1970, 1, 1)).total_seconds() * 1000 - 28800
# -28800 accounts for the browser showing me timestamps in Hong Kong timezone...I think. 1000 for ms
# init_start = 1522512000 * 1000  # 01.04.2018
d1 = 86400 * 1000
step = 1  # either 0 or 1. with 2, file checking breaks
eod = int(time.time()) * 1000

ctypes.windll.user32.MessageBoxW(0, "Login complete?", "ALERT", 1)

for curr in ['ETHUSD']: # ['BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'NEOUSD','XMRUSD']:  # currencies:
    createDir(r"C:\Users\seb\Desktop\bitfin_data\{}".format(curr))
    options = webdriver.ChromeOptions()
    options.add_experimental_option("prefs", {
        "download.default_directory": r"C:\Users\seb\Desktop\bitfin_data\{}".format(curr),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    driver.get(r'https://www.bitfinex.com/trade_history/{}'.format(curr))
    for start in np.arange(init_start, eod, (step+1) * d1):
        print(datetime.datetime.fromtimestamp(start//1000).strftime('%Y-%m-%d'))
        dt_start = datetime.datetime.fromtimestamp(start // 1000).strftime('%Y-%m-%d')
        if start + step * d1 <= eod:
            dt_end = datetime.datetime.fromtimestamp((start + step * d1) // 1000).strftime('%Y-%m-%d')
        else:  # for step==1 the browser may have end-date of today which is icomplete
            dt_end = datetime.datetime.fromtimestamp((start + d1) // 1000).strftime('%Y-%m-%d')

        target_dir = r"C:\Users\seb\Desktop\bitfin_data\{}".format(curr)
        target_fn = '{}-{}-trades-{}.csv'.format(dt_start, dt_end, curr)
        a = list(os.walk(target_dir))[0][2]
        dates = [o[0:10] for o in a] + [o[11:21] for o in a]
        if dt_start in dates:
            continue
        # print(datetime.datetime.fromtimestamp(1522454401).strftime('%Y-%m-%d'))
        test_exec(driver.find_elements_by_css_selector, val='a.right.csv-export-link')
        test_exec(driver.find_elements_by_xpath, val='// *[ @ for = "range_custom"]')
        time.sleep(1)
        # test_exec(driver.find_elements_by_xpath, val='//div[@id="export-date-range" and not(contains(@style, "display: none"))]//input[@name="custom_range"]')
        test_exec(driver.find_elements_by_xpath, val='//*[@id="export-date-range" and not(contains(@style, "display: none"))]')
        # test_exec(driver.find_elements_by_xpath, val='//*[@id="export-date-range"]')
        times = get_times(driver)
        # driver.execute_script('''
        # var xhr = new XMLHttpRequest();
        # xhr.open("POST", "https://www.bitfinex.com/reports/export/trades-BTCUSD?range=1522512000-1522684800", true);
        # xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        # xhr.send('authenticity_token=doBoFQAsK3R5oXJc9Ty6jPMS7jVvpV3Ve6bQWFnR5Civv550aWfytg1DYIBJESod0j7v0p7CKzZwWDzTjIsfUA==');
        # ''')
        while (min(times) + 5*d1) > start:
            prev = driver.find_element_by_xpath('//div[contains(@class, "date-picker-wrapper") and contains(@unselectable, "on") and not(contains(@style,"display: none"))]/div[@class="month-wrapper"]/table[@class="month1"]/thead/tr[1]/th[1]/span').click()
            prev = driver.find_element_by_xpath('//div[contains(@class, "date-picker-wrapper") and contains(@unselectable, "on") and not(contains(@style,"display: none"))]/div[@class="month-wrapper"]/table[@class="month2"]/thead/tr[1]/th[1]/span').click()
            times = get_times(driver)
        target_time = [t for t in times if t > start and t < (start + d1)][0]
        test_exec(driver.find_elements_by_xpath, val='//div[contains(@class, "date-picker-wrapper") and not(contains(@style,"display: none"))]//div[@time="{}" and not(contains(@class,"lastMonth")) and not(contains(@class,"nextMonth"))]'.format(target_time))
        test_exec(driver.find_elements_by_xpath, val='//div[contains(@class, "date-picker-wrapper") and not(contains(@style,"display: none"))]//div[@time="{}" and not(contains(@class,"lastMonth")) and not(contains(@class,"nextMonth"))]'.format(target_time + step * d1))
        test_exec(driver.find_elements_by_xpath, val='//div[contains(@class, "date-picker-wrapper") and not(contains(@style,"display: none"))]//input[@class="apply-btn"]')
        test_exec(driver.find_elements_by_id, val='download-report-btn')

        down_fn = '{}-trades-{}.csv'.format(datetime.datetime.today().strftime('%Y-%m-%d'), curr)
        down_path = r'C:\Users\seb\Downloads'
        if not timed_path_check(os.path.join(down_path, down_fn), timeout=5, reps=100):
            print('Couldnt download file: {}'.format(target_fn))

        for i in np.arange(50):
            try:
                # filename = max([os.path.join(dirpath, f) for f in os.listdir(dirpath)], key=os.path.getctime)
                # if 'crdownload' in filename:
                #     time.sleep(5)
                #     continue
                shutil.move(os.path.join(down_path, down_fn), os.path.join(target_dir, target_fn))
                break
            except (PermissionError, FileNotFoundError):
                time.sleep(5)
