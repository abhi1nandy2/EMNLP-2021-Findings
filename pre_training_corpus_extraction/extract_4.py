from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import json
from collections import OrderedDict
import urllib.parse

save_dir = "../data/new_pretrain_manuals"

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

save_file = "links_2.json"

save_path = os.path.join(save_dir, save_file)

read_path = save_path

def get_download_link(model_url):
	list_of_model_dict = []
	try:
		content = requests.get(model_url)
	except Exception as E:
		print(model_url)
		if "exceeded" in str(E).lower() and "redirects" in str(E).lower():
			print("This link has TooManyRedirects")
		else:
			print(E)
		return ""
	if str(content) != "<Response [200]>":
		print(str(content))
		print(model_url)
		return ""
	soup = BeautifulSoup(content.text, 'html.parser')
	a_ = soup.find(lambda tag:tag.name=="a" and "Open as PDF" in tag.text)
	if a_ == None:
		tmp = soup.find("object", type = "application/pdf")
		if tmp == None:
			print(model_url)
			return ""
		else:
			link = tmp['data']
	else:
		link = a_['href']

	return link

# print(get_download_link("http://marine.manualsonline.com/manuals/mfg/acr_electronics/y1030204.html"))

with open(read_path, 'r') as f:
	dict_ = json.load(f)

start_idx = 0

print(len(dict_['All_Manuals']))

for list_idx, list_item in enumerate(tqdm(dict_['All_Manuals'][start_idx:])):
	if list_item['models'] == []:
		continue
	for idx_2, item_2 in enumerate(list_item['models']):
		# print(item_2['model_url'])
		temp_return = get_download_link(item_2['model_url']) 
		if temp_return == "":
			continue
		else:
			dict_['All_Manuals'][start_idx + list_idx]['models'][idx_2]['download_link'] = temp_return

	if ((start_idx + list_idx + 1)%1000==0) or ((start_idx + list_idx) == len(dict_['All_Manuals'])):
		print("Saving output")
		with open(save_path, 'w') as f:
			json.dump(dict_, f, indent=2)
