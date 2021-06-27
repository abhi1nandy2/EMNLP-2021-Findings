import json
import os
from collections import OrderedDict

folderpath = "../data/all_manuals"

dict_ = OrderedDict()

device_list = []

for item in os.listdir(folderpath):
	item_path = os.path.join(folderpath, item)
	if os.path.isdir(item_path) == False:
		continue
	item_dict = OrderedDict()
	item_dict['device'] = item
	list_ = []	
	
	
	for file in os.listdir(item_path):
		series = file.replace(".txt", "").split("_")[-1]
		temp_dict = OrderedDict()
		temp_dict['series'] = series
		file_path = os.path.join(item_path, file)
		list_2 = []	
		with open(file_path, "r") as f:
			str_ = f.read()
		entries = str_.split("\n\n")[:-1]
		for temp in entries:
			temp_2 = temp.split("\n")
			model_carrier = temp_2[0]
			manual_link = temp_2[1]
			temp_2_dict = OrderedDict()
			temp_2_dict['model_carrier'] = model_carrier
			temp_2_dict['manual_link'] = manual_link
			list_2.append(temp_2_dict)
		temp_dict['manuals'] = list_2
		list_.append(temp_dict)
	item_dict['device_manuals'] = list_
	device_list.append(item_dict)

dict_['all_manuals'] = device_list

with open("../data/all_manuals/all_Samsung_manuals.json", 'w') as f:
 	json.dump(dict_, f, indent = 2)