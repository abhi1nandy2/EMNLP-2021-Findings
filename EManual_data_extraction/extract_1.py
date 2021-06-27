from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains 
import time
chromedriver=os.path.abspath("chromedriver.exe")
browser = webdriver.Chrome(executable_path=chromedriver)

url = "https://www.samsung.com/us/support/#get_to_know_your_product"
browser.get(url)
time.sleep(4)
# i=1
# print(i)

# element1.click()
# browser.get(url)

links = []
# i=1
# while(1):
	# try:
for i in range(1,7):
	j=1
	print(i, j)
	element1 = browser.find_element_by_xpath("//div[@class='sp-g-product-finder__content__cat']/ul/li[" + str(i) + "]/a[@href='javascript:void(0)']")
	print(element1.is_displayed())
	element1.click()
	time.sleep(2)
	ul_ = browser.find_elements_by_xpath("//ul")
	flag = 0
	for ul_tmp in ul_:	
		# print('hello')	
		
		# break
		li_list = ul_tmp.find_elements_by_tag_name("li")
		item_list = []
		for item in li_list:
			# if flag==1:
			if len(item.text) > 0 and len(item.find_elements_by_tag_name('a')) > 0 and item.is_displayed():
				flag = 1
				# print(ul_tmp.get_property('id'))
				print(item.text)
				item_list.append(item.text)
				# item.find_element_by_tag_name('a').click()
				
				# browser.get(url)
				# time.sleep(2)
				# print(item.is_displayed())
		if flag == 1:
			# j+=1
			break
	for j in range(len(item_list)):
		if j > 0:
			element1 = browser.find_element_by_xpath("//div[@class='sp-g-product-finder__content__cat']/ul/li[" + str(i) + "]/a[@href='javascript:void(0)']")
			element1.click()			
			time.sleep(3)
		browser.find_element_by_link_text(item_list[j]).click()
		links.append(browser.current_url)
		browser.get(url)
		time.sleep(2)

	# i+=1
	time.sleep(1)
	print(links)
	# except:
	# 	browser.get(url)
	# 	break

if os.path.exists("../data/all_manuals/") == False:
	os.mkdir("../data/all_manuals/")

with open("../data/all_manuals/links_1.txt", "w") as f:
	for item in links:
		f.write(item + "\n")