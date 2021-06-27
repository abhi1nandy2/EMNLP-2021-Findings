from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains 
import time
from collections import OrderedDict
import json
from tqdm import tqdm
import os

chromedriver=os.path.abspath("chromedriver.exe")
browser = webdriver.Chrome(executable_path=chromedriver)

folder_dict = {
	'phones': 'phones',
	'tvs': 'tvs',
	'televisions': "tvs",
	'refrigerators': "refrigerators",
	'washers': 'washers',
	'tablets': 'tablets',
	'wearables': 'wearables'
}

def get_manual_links(link):
	dict_ = OrderedDict()
	dict_['source_link'] = link
	dict_['device'] = link.split("/")[-2]
	dict_['series'] = link.split("/")[-1]
	browser.get(link)

	tmp_list = []
	tmp_dict = OrderedDict()
	i = 1

	folder = "../data/all_manuals/{}".format(folder_dict[dict_['device']])

	if os.path.exists(folder) == False:
		os.mkdir(folder)

	with open(os.path.join(folder, '{}_{}.txt'.format(dict_['device'], dict_['series'])), 'w') as f:

		while(1):
			try:
				time.sleep(3)
				element1 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__group two-dropdown']/ul/div[@class='icon-down-carat down-arrow']")
				element1.click()
		
				time.sleep(3)
				option1 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__container']/li[" + str(i) + "]")
				# option1.click()
				text1 = option1.text
				browser.execute_script("arguments[0].click();", option1)

				j = 1
				while(1):
					try:
						print(i)
						if j > 1:
							time.sleep(3)
							element1 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__group two-dropdown']/ul/div[@class='icon-down-carat down-arrow']")
							element1.click()
					
							time.sleep(3)
							option1 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__container']/li[" + str(i) + "]")
							# option1.click()
							text1 = option1.text
							browser.execute_script("arguments[0].click();", option1)
						time.sleep(3)
						element2 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__group__sec']/ul/li[1]")
						element2.click()

						time.sleep(3)
						option2 = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection__dropdown__group__sec']/ul/li[" + str(j+1) + "]")
						text2 = option2.text
						option2.click()

						tmp_dict = OrderedDict()
						tmp_dict['model'] = text1
						tmp_dict['carrier'] = text2

						btn = browser.find_element_by_xpath("//div[@class='sp-g-manuals-and-downloads__container--edit__right__selection selection-active']/button[@class='white-button sp-g-manuals-and-downloads__container--edit__right__selection__button']")
						browser.execute_script("arguments[0].click();", btn)

						time.sleep(3)
						try:
							# element = browser.find_element_by_xpath("//*[contains(text(), '0.00 MB')]/following-sibling::div[@class='sp-g-manuals-and-downloads__container--device__section__detail__list__detail__download defaultLanguage']/button/a[@class='sp-g-manuals-and-downloads__container--device__section__detail__download__link']")
							element = browser.find_element_by_xpath("//*[contains(text(), '0.00 MB')]/following-sibling::div/button/a")							
							tmp_dict['manual_link'] = element.get_attribute('href')
							f.write(text1 + " " + text2 + "\n")
							f.write(tmp_dict['manual_link'] + "\n\n")
							tmp_list.append(tmp_dict)
						except Exception as E:
							print(E)
						# print(element.text)

						browser.get(link)
						j+=1
					except:
						browser.get(link)
						break
				i+=1
			except:
				break

	dict_['manual_info'] = tmp_list
	return dict_

	# element = browser.find_element_by_xpath("//div[contains(., '0.00 MB')]/parent::*//div[@class='sp-g-manuals-and-downloads__container--device__section__detail__list__detail__size']")
	# print(element.text)
	# btn.click()

# dummy = "https://www.samsung.com/us/support/mobile/phones/galaxy-s"
# get_manual_links(dummy)

links_file = "../data/all_manuals/links_1.txt"

links_list = []

with open(links_file, 'r') as f:
	for line in f:
		links_list.append(line.strip())

final_dict = OrderedDict()

final_dict['All_Manuals'] = []

for link in tqdm(links_list):
	dict_ = get_manual_links(link)
	final_dict['All_Manuals'].append(dict_)

# with open("../data/all_manuals/All_Samsung_HTML_manuals.json", 'w') as f:
# 	json.dump(final_dict, f, indent = 2)