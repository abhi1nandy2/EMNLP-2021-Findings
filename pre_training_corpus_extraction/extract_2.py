from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import json
from collections import OrderedDict
import math

save_dir = "../data/new_pretrain_manuals"

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

save_file = "links_2.json"

save_path = os.path.join(save_dir, save_file)

read_dir = save_dir
read_file = "links_1.txt"

init_links = []

with open(os.path.join(read_dir, read_file), "r") as f:
	for line in f:
		init_links.append(line.strip())

def get_category_data(url):
	brand = url.split("/")[-1]
	categories = []
	brand_category_urls = []

	counter = 0
	max_count = 100000
	while(1):	
		counter+=1
		if counter > max_count:
			break
		content = requests.get(url + "?p=" + str(counter))
		soup = BeautifulSoup(content.text, 'html.parser')

		if counter == 1:
			text_ = soup.find("div", class_="pagination pull-right").span.span.text
			text_ = text_.strip()
			try:
				per_page_cnt = int(text_.split(" ")[-3])
			except Exception as E:
				print(url)
				print(E)
				return "","",""

			total = int(text_.split(" ")[-1])
			max_count = math.ceil(total/per_page_cnt)
			# print(per_page_cnt, total, max_count)
		table = soup.find("div", class_="words-list")
		list_ = table.findAll("div", class_="letter-content")

		for item in list_:
			a_ = item.find("div", class_ = "col-md-5 col-sm-5 col-xs-10").h5.a
			categories.append(a_.text)
			brand_category_urls.append(a_['href'])

	return brand, categories, brand_category_urls

dict_ = OrderedDict()
dict_['All_Manuals'] = []

for init_link in tqdm(init_links):
	brand, categories, brand_category_urls = get_category_data(init_link)
	if categories == "":
		continue
	for iter_ in range(len(categories)):
		tmp_dict = OrderedDict()
		tmp_dict['brand'] = brand
		tmp_dict['category'] = categories[iter_]
		tmp_dict['brand_category_url'] = brand_category_urls[iter_]
		tmp_dict['models'] = []
		dict_['All_Manuals'].append(tmp_dict)

with open(save_path, "w") as f:
	json.dump(dict_, f, indent = 2)