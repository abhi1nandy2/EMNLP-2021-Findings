from bs4 import BeautifulSoup
import requests
from tqdm import tqdm
import os
import urllib.parse

save_dir = "../data/new_pretrain_manuals"

if os.path.exists(save_dir) == False:
	os.mkdir(save_dir)

save_file = "links_1.txt"

save_path = os.path.join(save_dir, save_file)

def get_data(url):
	base_url= "http://www.manualsonline.com/"
	content = requests.get(url)
	soup = BeautifulSoup(content.text, 'html.parser')

	table = soup.find("div", class_="words-list")
	list_ = table.findAll("div", class_="letter-content")

	rel_links = [item.h5.a['href'] for item in list_]
	brand_links = [urllib.parse.urljoin(base_url, rel_url) for rel_url in rel_links]

	return brand_links

with open(save_path, 'w') as f:
	for i in tqdm(range(1, 83)):
		url = "http://www.manualsonline.com/brands/?p=" + str(i)
		brand_links = get_data(url)
		for item in brand_links:
			f.write(item + "\n")