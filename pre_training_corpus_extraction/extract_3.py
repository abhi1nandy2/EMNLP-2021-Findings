from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import json
from collections import OrderedDict
import math
import urllib.parse

save_dir = "../data/new_pretrain_manuals"

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

save_file = "links_2.json"

save_path = os.path.join(save_dir, save_file)

read_path = save_path

with open(read_path, 'r') as f:
	dict_ = json.load(f)

def get_model_data(brand_category_url):
	base_url = brand_category_url.split('manuals/')[0]
	# print(brand_category_url)
	list_of_model_dict = []
	counter = 0
	max_count = 100000
	while(1):	
		counter+=1
		if counter > max_count:
			break
		content = requests.get(brand_category_url + "?p=" + str(counter))
		if str(content) != "<Response [200]>":
			print(str(content))
			print(brand_category_url)
			return ""
		soup = BeautifulSoup(content.text, 'html.parser')

		if counter == 1:
			try:
				text_ = soup.find("div", class_="pagination pull-right").span.span.text
			except Exception as E:
				print(brand_category_url)
				print(E)
				return ""
			text_ = text_.strip()
			try:
				per_page_cnt = int(text_.split(" ")[-3])
			except Exception as E:
				print(brand_category_url)
				print(E)
				return ""

			total = int(text_.split(" ")[-1])
			max_count = math.ceil(total/per_page_cnt)

		table = soup.find("div", class_="words-list product-list")
		list_ = table.findAll("div", class_="col-md-8 col-sm-8 col-xs-7")
		for item in list_:
			a_ = item.h5.a			
			model_dict = OrderedDict()
			model_dict['title'] = a_.text
			model_dict['desc'] = item.p.text
			model_dict['num_pages'] = item.div.span.text
			model_dict['model_url'] = urllib.parse.urljoin(base_url, a_['href'])
			list_of_model_dict.append(model_dict)

	return list_of_model_dict

list_ = dict_['All_Manuals']

for idx, item_ in enumerate(tqdm(list_)):
	list_[idx]['models'] = []
	brand_category_url = item_['brand_category_url']
	temp_ = get_model_data(brand_category_url)
	if temp_ == "":		
		list_[idx]['models'] = []
	else:
		list_[idx]['models'] = temp_
	dict_['All_Manuals'] = list_
	with open(save_path, 'w') as f:
		json.dump(dict_, f, indent=2)
