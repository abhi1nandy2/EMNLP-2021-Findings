import os
from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains 
from selenium.webdriver.common.keys import Keys
import argparse

from bs4 import BeautifulSoup, NavigableString
import requests
import os
import sys
import pandas as pd
import urllib.parse
import time
from collections import OrderedDict
import json
from urllib.parse import urlparse
from urllib.parse import urldefrag
from textblob import TextBlob
from tqdm import tqdm

# parser = argparse.ArgumentParser()

# parser.add_argument("--save_path", type = str, required=True, help = "path to save the sections of the E-Manual")
# parser.add_argument("--url", type=str, required=True, help = "E-Manual url")

# args = parser.parse_args()

proxy_dict={"http":"", "https":"", "ftp":""}

def rem_sp(text):
	'''
	remove spaces and unicode escape chars
	'''
	text = " ".join(text.split())
	return text.encode('ascii', 'ignore').decode('unicode_escape')	


# sent_list = []

def getText(parent):
	'''
	For a <li> element having some nested tags (e.g. other ol, ul), or when there is a <b> tag, .get_text returns all the text
	within the parent tag including the ones in the nested tags. In order to filter out the text exclusive 
	to the parent tag, this function is used. Also removes unicode escape chars.
	'''
	parts = []
	for element in parent:
		if isinstance(element, NavigableString):
			parts.append(element)
		if(element.name == 'b' and element.parent.parent.parent != 'li'):
			parts.append(element.get_text())
	text  = ''.join(parts)
	text = " ".join(text.split())
	text =  text.encode('ascii', 'ignore').decode('unicode_escape')
	text_list = [sent_.raw for sent_ in TextBlob(text).sentences]
	return text_list

def append_text_recursive(tag, sent_list, first_time = True):	
	if first_time == True:		
		# sent_list.append(tag.find(text=True, recursive=False))
		sent_list.extend(getText(tag))
	immediate_children = tag.findChildren(recursive = False)
	if len(immediate_children) > 0:
		for child in immediate_children:
			# sent_list.append(child.find(text=True, recursive=False))
			if (child.name != "b"):
				sent_list.extend(getText(child))
				append_text_recursive(child, sent_list, first_time = False)
	return sent_list

def traverse(manual_url, soup, dict_):
	# print("hi")
	for ul1 in soup.findAll("ul", recursive = False):
		# print("loop1")
		for li1 in ul1.findAll("li", recursive = False):
			# print("hello")
			link1 = li1.find('a', href=True)
			link1_text = urllib.parse.urljoin(manual_url, link1['href']) + "|" + rem_sp(link1.text)
			tmp1 = li1.findAll('ul', recursive = False)
			if len(tmp1) == 0:
				dict_[link1_text] =  OrderedDict()
			else:
				dict_[link1_text] = OrderedDict()
				dict_[link1_text] = traverse(manual_url, li1, dict_[link1_text])
				# print(dict_)
	return dict_

def get_section_path(key, dic):
	ch = list(dic.keys())
	if key in ch:
		return [key]
	for c in ch:
		pp = get_section_path(key, dic[c])
		if pp != None:
			return [c] + pp

def preorder(dict_, list_, init_dict_):
	if len(dict_) > 0:
		for key in dict_:
			section_url = key.split("|")[0]
			# if "start_here.html" not in section_url:
			# 	rel_url = section_url.split("/")[-1]
			# 	rel_url = "toc.html#" + rel_url.replace(".html", "")
			# 	section_url = urllib.parse.urljoin(section_url, rel_url)
			topic = key.split("|")[1]
			section_path = get_section_path(key, init_dict_)
			tmp_list = get_contents(section_url, ">".join(section_path))
			# print(section_path, tmp_list)
			if "start_here.html" in section_url and section_path[0].split("|")[1] != "Welcome":
				tmp_list = []
				# print("hello")
			elif tmp_list == None:
				continue
			list_.extend(tmp_list)
			preorder(dict_[key], list_, init_dict_)

	return list_

def get_contents(section_url, sec_hier):
	html = requests.get(section_url).text
	soup = BeautifulSoup(html)
	# if soup.find("ul", class_ = "toc-nav") == None and "start_here.html" not in section_url:
	# 	rel_url = section_url.split("/")[-1]
	# 	rel_url = "toc.html#" + rel_url.replace(".html", "")
	# 	new_url = urllib.parse.urljoin(section_url, rel_url)
	# 	print(new_url)
	# 	html = requests.get(new_url).text
	# 	soup = BeautifulSoup(html)
	# 	print(soup)

		# assert(1==0)

	o = urlparse(section_url)
	frag = o.fragment

	if frag == '':
		id_ = section_url.split("/")[-1]
		id_ = id_.split("_")[-1]
		id_ = id_.replace(".html", "")
		start = soup.find(["h1", "h2", "h3"], id = id_)
		if(start == None):
			id_ = section_url.split("/")[-1]
			id_ = "_".join(id_.split("_")[:-1])
			start = soup.find(["h1", "h2", "h3"], id = id_)
			# assert(start!=None)
	else:
		id_ = frag
		start = soup.find(["h1", "h2", "h3"], id = id_)

	if start == None:
		return None

	list_ = []
	iter_ = start

	text_ = []
	fragment = ""
	html_ = []
	flag = 0
	# none_flag = 0

	while(1):
		# if iter_.name == None:
		# print(iter_.next_sibling)
		if iter_ == None:
			iter_name_ = "_"
		elif iter_.name == None:
			iter_ = iter_.next_sibling
			continue
		else:
			iter_name_ = iter_.name
		if (iter_name_ in "".join(["h4", "h5", "h6"])) or (((iter_name_ in "".join(["h1", "h2", "h3"])) or (iter_name_ == "_")) and iter_!=start):
			#add the previous chunk's info as a dictionary
			temp_dict_= OrderedDict()
			# temp_dict_['section_id'] = section_id
			temp_dict_['title'] = title
			temp_dict_['text'] = [it_ for it_ in text_ if it_!=""]
			temp_dict_['html'] = html_
			if fragment != "":
				temp_dict_['section_url'] = urldefrag(section_url).url + "#" + fragment
			else:
				temp_dict_['section_url'] = section_url
			if temp_dict_['section_url'] + "|" + title != sec_hier.split(">")[-1]:
				temp_dict_['section_hierarchy'] = sec_hier + ">" + temp_dict_['section_url'] + "|" + title
			else:
				temp_dict_['section_hierarchy'] = sec_hier
			list_.append(temp_dict_)

			if (((iter_name_ in "".join(["h1", "h2", "h3"])) or (iter_name_ == "_")) and iter_!=start):
				#you have reached the end of the section, break it here.
				break

			#Now, use the present chunk
			# section_id = section_id.split("_")[0] + "_" + str(int(section_id.split("_")[1]) + 1)
			fragment = iter_['id']
			text_ = []
			html_ = []
			flag = 0

		if flag == 0: #either main heading or [h4,h5,h6]
			title = rem_sp(iter_.text)
		sent_list = []
		# print(iter_)
		sent_list = append_text_recursive(iter_, sent_list)
		text_.extend(sent_list)
		html_.append(iter_.prettify())
		flag = 1
		# prev_iter_ = iter_
		iter_ = iter_.next_sibling #incrementing condition	

	# with open("section_temp_json.json", 'w') as f:
	# 	json.dump(list_, f, indent = 2)
	return list_#, section_id

def get_sections_data(manual_url, save_path):
	# print("Getting the data")
	content = requests.get(manual_url, proxies = proxy_dict)
	soup = BeautifulSoup(content.text, 'html.parser')

	#We get the second link on the nav-bar in order to get all possible links in the nav-bar
	nav_bar = soup.find("ul", class_ = "main-nav")
	rel_url = nav_bar.findAll("li")[1].a['href']

	# print(rel_url)
	# url = urllib.parse.urljoin(manual_url, rel_url)
	url = manual_url.replace("start_here.html", rel_url)
	# print(url)

	temp_file = "temp.html"

	chromedriver="/mnt/c/Users/user/Documents/chromedriver.exe"
	# chromedriver='chromedriver.exe'
	browser = webdriver.Chrome(executable_path=chromedriver)

	browser.get(url)
	current_url = browser.current_url
	browser.get(current_url)

	# print("Still getting the data")

	with open(temp_file, 'w') as f:
		f.write(browser.page_source)

	# ActionChains(browser).key_down(Keys.CONTROL).send_keys('a').key_up(Keys.CONTROL).perform()

	browser.close()

	print("Saved temporarily in an HTML file")

	with open(temp_file) as fp:
		soup = BeautifulSoup(fp, 'html.parser')

	# print(soup.findAll("ul", recursive = False))

	# nav_bar_ul = soup.find("ul", class_ = "toc-nav")
	nav_bar = soup.find("div", class_ = "toc")

	dict_ = OrderedDict()
	dict_ = traverse(urllib.parse.urljoin(current_url, "start_here.html"), nav_bar, dict_)

	# print(dict_)

	with open("temp_json.json", 'w') as f:
		json.dump(dict_, f, indent = 2)

	print("Saved sections in nested json form.")

	
	sections_list = []
	sections_list = preorder(dict_, sections_list, dict_)

	# print(sections_list)

	section_corpus = OrderedDict()

	urltitle_to_section_dict = OrderedDict()

	for idx, dict_item in enumerate(sections_list):
		dict_item['id'] = 'section_' + str(idx)
		urltitle_to_section_dict[dict_item['section_url'] + "|" + dict_item["title"]] = dict_item['id']
		sections_list[idx] = dict_item

	for idx, dict_item in enumerate(sections_list):
		tmp_hier_0 = dict_item['section_hierarchy'].split(">")
		tmp_hier = [urltitle_to_section_dict[key_] for key_ in tmp_hier_0 if key_ in urltitle_to_section_dict]
		if len(tmp_hier_0) != len(tmp_hier):
			section_corpus['section_' + str(idx)] = {}	
		dict_item['section_hierarchy'] = ">".join(tmp_hier)
		section_corpus['section_' + str(idx)] = dict_item

	with open(save_path, 'w') as f:
		json.dump(section_corpus, f, indent = 2)

	# all_paths = get_section_path("https://downloadcenter.samsung.com/content/PM/202005/20200506022945248/EB/UNL_G973U_G975U_EN_FINAL_200110_WAC/charge_the_battery_d1e5484.html|Charge the battery", dict_)
	# print(all_paths)

# get_sections_data("https://downloadcenter.samsung.com/content/PM/202001/20200128062847846/EB/ATT_G970U_G973U_G975U_EN_FINAL_200110/start_here.html")

# get_sections_data("https://org.downloadcenter.samsung.com/downloadfile/ContentsFile.aspx?CDSite=US&CttFileID=7127979&CDCttType=PM&ModelType=C&ModelName=SM-G891AZAAATT&VPath=PM/201809/20180922044543027/EB/ATT_G891A_EN_FINAL_180921/start_here.html", "../../data/other_emanuals/dummy.json")
#for tablet

# '''
num_lines = 0
lines = []

with open("../data/Amazon_links.txt", 'r') as f:
	for line in f:
		lines.append(line.strip())
		num_lines += 1

num_products = int(num_lines/3)

save_dir = "../data/other_emanuals"

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

for i in tqdm(range(num_products)):
	# try:
	product_name = "_".join(lines[3*i].split(" "))
	save_path = os.path.join(save_dir, product_name + ".json")
	url = lines[3*i + 1]
	url = url.replace("http:", "https:")

	print(save_path, url)
	try:
		get_sections_data(url, save_path)
	except Exception as E:
		print(E)
# '''
#for smartwatch

# get_contents("https://downloadcenter.samsung.com/content/PM/202005/20200506022945248/EB/UNL_G973U_G975U_EN_FINAL_200110_WAC/navigation_bar_d1e5997.html", "dummy2")