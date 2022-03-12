import selenium
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
import os, datetime
import numpy as np
import importlib.util
import pickle
import ctypes
import pandas as pd
import time

class XPath:
    pass

class BaseSpider():

    extension_directory = Paths.extension_directory
    path_chrome = Paths.path_chrome
    start_url = r''
    driver = None
    target = None
    scraper_name = ''
    max_listing_pages = 100
    dataStore = None
    XPaths = None
    txt_sep = ' ; '
    current_url = ''   # for logging debugging and tracing missing elements and problematic extractions
    copy_current = False
    copy_last_folder = False
    df_prev_scraped = pd.DataFrame()

    def start_browser(s):
        options = webdriver.ChromeOptions()
        options.add_argument(f'load-extension={Paths.extension_directory}')
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
        s.driver = webdriver.Chrome(s.path_chrome, chrome_options=options)
        # go taobao main page
        s.driver.get(s.start_url)

    @staticmethod
    def last_folder_out():
        num_folder = ''
        for _, dirs, filenames in os.walk(Paths.raw_data):
            num_folder = [int(el) for el in dirs]
            break
        print('Referencing previous scrapes from {}'.format(max(num_folder)))
        return str(max(num_folder))

    def load_previous_scrape_results(s) -> pd.DataFrame():
        if s.copy_last_folder:
            transformer = Transformer(s.target)
            transformer.load_raw(s.target, r'\{}'.format(BaseSpider.last_folder_out()))
            s.df_prev_scraped = pd.concat([s.df_prev_scraped, transformer.transform()], axis=0)
        if s.copy_current:
            transformer_curr = Transformer(s.target)
            transformer_curr.load_raw(s.target)
            df_prev_scraped_curr = transformer_curr.transform()
            s.df_prev_scraped = pd.concat([s.df_prev_scraped, df_prev_scraped_curr], axis=0)
        return s.df_prev_scraped


    @staticmethod
    def manual_login_confirm():
        ctypes.windll.user32.MessageBoxW(0, "Login complete?", "ALERT", 1)

    def scrape(s):
        # loop over key words
        for word in KEYWORDS:
            s.dataStore = DataStore(word, s.scraper_name)
            try:
                # enter search search
                cnt_tabs = s.driver.window_handles
                s.search_from_main(word)
                # identify new page
                ActionChains(s.driver).pause(1)
                if s.driver.window_handles > cnt_tabs:
                    # close main search bar where we come from. now only use listing search
                    s.driver.switch_to.window(s.driver.window_handles[0])
                    s.close_tab()
                    # s.driver.switch_to.window(s.driver.window_handles[-1])
                page = s.identify_page()
                if page == Pages.login:
                    s.tb_login()
                    s.parse_listings(word)
                elif page == Pages.tbmain:
                    s.scrape()  # if somehow falling into this -> infinite loop
                elif page == Pages.listings:
                    s.parse_listings(word)
                elif page == Pages.listings_wo_results:
                    Logger.info('Keyword {}: No results'.format(word))
                    continue
                else:
                    Logger.info('Lost our way at {}'.format(word))
                    raise ('Dont know where we are')
                    # send a webhook to Slack in case of this
                s.dataStore.store_record()
            except Exception as e:
                Logger.debug('Error processing word {}. Traceback: {}'.format(word, e.__traceback__))
                continue

    def parse_listings(s, word):
        # product_urls = s.extract('//*[contains(@class, "m-itemlist")]//*[contains(@class, "pic")]/@data-href')
        # product_urls = [s.prepend_url_schema(url) for url in product_urls]

        next_button = True
        listing_page = 0
        while next_button and listing_page < s.max_listing_pages:
            product_elem = s.driver.find_elements_by_xpath('//*[contains(@class, "m-itemlist")]//div/div/a[contains(@class, "pic") and not(contains(@href, "tmall"))]')
            for product in product_elem:
                cnt_tabs = len(s.driver.window_handles)
                actions = ActionChains(s.driver)
                actions.pause(np.random.randint(500, 1600) / 1000)
                # actions.perform()
                # s.move_w_delay_to(product, 5, 5)
                # actions = ActionChains(s.driver)
                actions.click(product)
                actions.pause(np.random.randint(200, 600) / 1000)
                # actions.click()
                actions.perform()
                # load post
                s.driver.implicitly_wait(2)
                # did a new tab open?
                if len(s.driver.window_handles) > cnt_tabs:
                    s.driver.switch_to.window(s.driver.window_handles[-1])
                # check it's a product site
                page = s.identify_page()
                if page == Pages.product:
                    s.parse_product(word)
                    s.close_tab()
                elif page == Pages.login:
                    s.tb_login()
                    if len(s.driver.window_handles) > cnt_tabs:
                        s.driver.switch_to.window(s.driver.window_handles[-1])
                    s.parse_product(word)
                    s.close_tab()
            next_button = s.elem_xpath('//*[@id="mainsrp-pager"]//li[@class="item next"]/a')
            if next_button:
                ok = s.click_next()
                s.driver.implicitly_wait(2)
                listing_page += 1
            # ActionChains(s.driver).pause(1)
            # else means while loop stops because next button is false condition in while

    def click_next(s):
        next_button = s.elem_xpath(s.XPaths.itemlist_next)
        if next_button is False:
            return False
        actions = ActionChains(s.driver)
        actions.pause(0.1)
        actions.click(next_button)
        # actions.move_to_element_with_offset(next_button, 5, 5).click()
        # s.move_w_delay_to(next_button, 2, 2)
        # actions = ActionChains(s.driver)
        actions.pause(0.5)
        actions.perform()
        return True

    def move_w_delay_to(s, elem, off_x, off_y):
        html_00 = s.driver.find_element_by_xpath('//*[@id]')
        # html_00 = s.driver.find_element_by_xpath('/html')
        tx = elem.location['x'] + off_x + 10
        ty = elem.location['y'] + off_y + 10
        i_dx = int(tx / 2)
        i_dy = int(ty / 2)
        steps_xy = min(i_dx, i_dy)
        steps_single = max(i_dx, i_dy) - steps_xy
        actions = ActionChains(s.driver)
        actions.move_to_element_with_offset(html_00, -html_00.location['x'], -html_00.location['y']).perform()
        actions = ActionChains(s.driver)
        actions.pause(0.005)
        # go to 0,0
        for i in range(0, steps_xy):
            actions.move_by_offset(2, 2)
            actions.pause(0.002)
        actions.perform()
        # ActionChains(s.driver).context_click().perform()
        actions = ActionChains(s.driver)
        for i in range(0, steps_single):
            if i_dx > i_dy:
                actions.move_by_offset(2, 0)
            else:
                actions.move_by_offset(0, 2)
            actions.pause(0.002)
        actions.perform()

    def move_to_element(s, el):
        ActionChains(s.driver).move_to_element(el).perform()

    def scroll_down(s, n=30):
        for i in range(n):
            actions = ActionChains(s.driver)
            # actions.pause(np.random.randint(500, 1600) / 1000)
            actions.send_keys(Keys.PAGE_DOWN)
            actions.pause(np.random.randint(200, 600) / 1000)
            actions.perform()

    def scroll_up(s, n=30):
        for i in range(n):
            actions = ActionChains(s.driver)
            # actions.pause(np.random.randint(500, 1600) / 1000)
            actions.send_keys(Keys.PAGE_UP)
            actions.pause(np.random.randint(200, 600) / 1000)
            actions.perform()

    def close_tab(s):
        s.driver.close()
        # ActionChains(s.driver).send_keys(Keys.LEFT_CONTROL + 'w').perform()
        s.driver.switch_to.window(s.driver.window_handles[-1])
        s.current_url = s.driver.current_url

    def elem_id(s, id):
        return '//li[@id="{}"]'.format(id)

    def parse_product(s):
        pass

    def select_last_tab(s):
        if len(s.driver.window_handles) > 1:
            s.driver.switch_to.window(s.driver.window_handles[-1])
        s.current_url = s.driver.current_url

    def ensure_first_tab_only(s):
        while len(s.driver.window_handles) > 1:
            s.driver.switch_to.window(s.driver.window_handles[-1])
            s.close_tab()
        s.current_url = s.driver.current_url

    def search_from_main(s, word):
        main_search_bar = s.elem_xpath(s.XPaths.main_search_bar)
        if not main_search_bar:
            main_search_bar = s.elem_xpath(s.XPaths.listing_search_bar)
            if not main_search_bar:
                s.driver.get(s.start_url)
                s.driver.implicitly_wait(1)
                main_search_bar = s.elem_xpath(s.XPaths.main_search_bar)
                # raise ("Couldnt find search bar")
        # s.move_w_delay_to(tbmain_search_bar, 100, 5)
        actions = ActionChains(s.driver)
        # move_to_element_with_offset(tbmain_search_bar, 5, 5).
        actions.click(main_search_bar)
        actions.pause(np.random.randint(400, 800) / 1000)
        actions.send_keys(Keys.END)
        actions.pause(np.random.randint(90, 100) / 1000)
        for i in range(20):
            actions.send_keys(Keys.BACKSPACE)
            actions.pause(np.random.randint(40, 80) / 1000)
        for key in word:
            actions.send_keys(key)
            actions.pause(np.random.randint(400, 800) / 1000)
        actions.send_keys(Keys.ENTER)
        # load time - but combine with check for elements or addListener document.onload, but tb has some long running scripts even
        # when page appears fully loaded
        actions.pause(1)
        actions.perform()

    @staticmethod
    def get_floats_in_list(text: list) -> list:
        r = []
        for t in text:
            try:
                r.append(float(t))
            except TypeError:
                continue
        return r

    @staticmethod
    def min_number(numbers: list):
        try:
            return min(numbers)
        except ValueError:
            return None

    def elem_xpath(s, xpath):
        try:
            return s.driver.find_element_by_xpath(xpath=xpath)
        except selenium.common.exceptions.NoSuchElementException:
            return False

    def elem_xpath_w_attr(s, xpath, attr):
        try:
            elem = s.driver.find_element_by_xpath(xpath=xpath)
            try:
                return elem.get_attribute(attr)
            except AttributeError:
                Logger.info('Didnt find attribute {} at {}'.format(attr, s.current_url))
        except selenium.common.exceptions.NoSuchElementException:
            Logger.info('Didnt find element for {} at {}'.format(xpath, s.current_url))
            return False

    def elems_xpath_w_attr(s, xpath, attr):
        try:
            elements = s.driver.find_elements_by_xpath(xpath=xpath)
            try:
                return [el for el in [elem.get_attribute(attr) for elem in elements] if el not in ['', None]]
            except AttributeError:
                Logger.info('Didnt find attribute {} at {}'.format(attr, s.current_url))
        except selenium.common.exceptions.NoSuchElementException:
            Logger.info('Didnt find element for {} at {}'.format(xpath, s.current_url))
            return False

    def elem_xpath_w_text(s, xpath):
        try:
            elem = s.driver.find_element_by_xpath(xpath=xpath)
            try:
                return elem.text
            except AttributeError:
                Logger.info('Didnt find text in element at {}'.format(s.current_url))
                return ''
        except selenium.common.exceptions.NoSuchElementException:
            Logger.info('Didnt find element for {} at {}'.format(xpath, s.current_url))
            return False

    def elems_xpath_w_text(s, xpath):
        try:
            elems = s.driver.find_elements_by_xpath(xpath=xpath)
            try:
                return [el for el in [elem.text for elem in elems] if el not in ['', None]]
            except AttributeError:
                Logger.info('Didnt find text at {}'.format(s.current_url))
                return ['']
        except selenium.common.exceptions.NoSuchElementException:
            Logger.info('Didnt find element for {} at {}'.format(xpath, s.current_url))
            return False

    @staticmethod
    def get_int_str_if_present(char):
        try:
            return str(int(char))
        except ValueError:
            return ''

    def press_login(s):
        submit = s.driver.find_element_by_xpath("//*[@id='J_SubmitStatic']")
        actions = ActionChains(s.driver)
        actions.move_to_element_with_offset(submit, 5, 5).pause(0.5).click().perform()
        # s.driver.save_screenshot(
        #     os.path.join(Paths.img_path, 'screen_{}.png'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))

    def close_all_tabs(s):
        for handle in range(0, s.driver.window_handles, -1):
            if handle > 0:
                s.driver.switch_to.window(s.driver.window_handles[0])
                s.close_tab()

    @staticmethod
    def prepend_url_schema(url):
        if url is None:
            return url
        elif url[:2] == r"//":
            return "https:" + url
        else:
            return url

    def solve_captcha(s):
        """this works (2018.10.14) but copy pasted from trial code. needs some refactoring
        to fit class structure and references"""
        while True:
            try:
                # 定位滑块元素,如果不存在，则跳出循环
                show = s.driver.find_element_by_xpath("//*[@id='nocaptcha']")
                if not show.is_displayed():
                    break
                source = s.driver.find_element_by_xpath("//*[@id='nc_1_n1z']")
                s.driver.implicitly_wait(3)
                # 定义鼠标拖放动作
                # ActionChains(driver).drag_and_drop_by_offset(source,400,0).perform()
                # driver.save_screenshot('login-screeshot-11.png')
            except:
                pass
            action = ActionChains(s.driver)
            s.driver.implicitly_wait(1)
            action.click_and_hold(source).perform()
            for index in range(10):
                try:
                    action.move_by_offset(4, 0).perform()  # 平行移动鼠标
                    # driver.save_screenshot('login-screeshot-i-'+str(index)+'.png')
                except Exception as e:
                    print(e)
                    break
                if index == 9:
                    action.release()
                    s.driver.implicitly_wait(0.05)
                    # s.driver.save_screenshot(os.path.join(Paths.img_path, 'screen_{}.png'.format(
                    #     datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
                else:
                    s.driver.implicitly_wait(0.01)  # 等待停顿时间
                s.driver.implicitly_wait(.5)
                # s.driver.save_screenshot(
                #     os.path.join(Paths.img_path, 'screen_{}.png'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))
                text = s.driver.find_element_by_xpath("//*[@id='nc_1__scale_text']/span")
                if text.text.startswith(u'验证通过'):
                    print('成功滑动')
                    break
                if text.text.startswith(u'请点击'):
                    print('成功滑动')
                    break
                if text.text.startswith(u'请按住'):
                    continue
                # except Exception as e:
                #     print(e)
                #     driver.find_element_by_xpath("//div[@id='nocaptcha']/div/span/a").click()
        # s.driver.save_screenshot(
        #     os.path.join(Paths.img_path, 'screen_{}.png'.format(datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))))

    def autologin(s):
        # res = (3840, 2160)
        # win = s.driver.get_window_size()
        # xf = res[0] / win['width']
        # xy = res[1] / win['height']
        # b_user = s.driver.find_element_by_xpath('//*[@id="TPL_username_1"]')
        # u = pi.Point(x=int(b_user.location['x'] * xf) + 5, y=int(b_user.location['y'] * xy) + 2)
        # b_pass = s.driver.find_element_by_xpath('//*[@id="TPL_password_1"]')
        # p = pi.Point(x=b_pass.location['x'] - 10, y=b_pass.location['y'] - 2)
        # b_sub = s.driver.find_element_by_xpath('//*[@id="J_SubmitStatic"]')
        # l = pi.Point(x=b_sub.location['x'] + 15, y=b_sub.location['y'] + 1)
        u = pi.Point(x=2987, y=1058)
        p = pi.Point(x=2629, y=1214)
        l = pi.Point(x=2751, y=1371)
        e_p = []
        e_p.append((pi.moveTo, [u.x, u.y, 2]))
        e_p.append((pi.click, []))
        e_p.append((pi.typewrite, ['390475proton']))
        e_p.append((pi.moveTo, [p.x, p.y, 2]))
        e_p.append((pi.click, []))
        e_p.append((pi.typewrite, ['33CtRZvVimZJ8P6']))
        e_p.append((pi.moveTo, [l.x, l.y, 2]))
        e_p.append((pi.click, []))
        for el in e_p:
            el[0](*el[1])
        time.sleep(1)