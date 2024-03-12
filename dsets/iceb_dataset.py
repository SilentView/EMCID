import csv
import json
from pathlib import Path
from typing import List, Dict, Literal
import os
import random

import torch
from torchvision.datasets import ImageNet
from torch.utils.data import Dataset
import numpy as np
from transformers import (
	CLIPTextModel,
	CLIPTokenizer
)
import pandas as pd
from tqdm import tqdm

from util.globals import *


class ImageNetMendRequestDataset(Dataset):
	def __init__(
		self,
		class_summary_file="data/iceb_data/imgnet_prompts_cls.json",
		prompt_file="data/iceb_data/imgnet_aug_full.json",
		type: Literal["edit", "val"]="edit",
		no_extra_knowledge=True,   # select classes that have at least one name with high score
		use_imgnet_imgs=False,
		imgs_per_class=3,		# number of images for training when using imgnet images for training
		class_score_threshold=0.5,
		name_score_threshold=0.1,
		prompts_per_request=3,
		use_simple_train_prompt=True,
		verbose=False
	):
		"""
		This dataset is used to mend the wrong class names in ImageNet dataset.
		The class names are selected by the following criteria:
		1. The class has at least one name with high score (class_score_threshold)
		2. The class has at least one name with low score (name_score_threshold)
		3. The class has its prompts in the prompt_file
		"""

		# load the class summary file
		with open(class_summary_file, "r") as f:
			summary = json.load(f)
		
		wrong_classes = []
		wrong_names = []
		self.requests = []
		self.no_extra_knowledge = no_extra_knowledge
		# find the classes that has low scores
		for class_id, class_data in summary.items():
			highest_score = 0
			for class_name, class_score in class_data.items():
				highest_score = max(highest_score, class_score["mean"])
				if class_score["mean"] < name_score_threshold:
					wrong_names.append((class_id, class_name))
			if highest_score < class_score_threshold:
				wrong_classes.append(class_id)
			
		print(f"Found {len(wrong_classes)} low score classes.")
		if verbose:
			print(wrong_classes)

		print(f"Found {len(wrong_names)} wrong names.")
		if verbose:
			print(wrong_names)
		
		with open(prompt_file, "r") as file:
			prompt_data = json.load(file)

		if use_simple_train_prompt:	
			edit_prompts_tmp = [
				"An image of {}",
				"A photo of {}",
				"{}"
			]
		else:
			edit_prompts_tmp = [
				"A visually captivating image, showcasing the detialsd of {}",
				"A photograph showcasing {}",
				"A remarkable image, presenting a detailed view of {}"
			]

		if type == "edit":
			prompt_slice = slice(0, EDITING_PROMPTS_CNT)
		elif type == "val":
			prompt_slice = slice(None)
		else:
			raise ValueError(f"Invalid type {type}.")
		
		# load real imgnet images using pytorch dataset
		# imgnet_dataset = ImageNet(root=DATA_DIR, split="train")
		if use_imgnet_imgs:
			print("Loading ImageNet dataset...")
			dataset = ImageNet(root="data/ImageNet", split="val")
			print("Done.")

		random.seed(2023)
		seeds_train = random.sample(range(10000), len(wrong_names))
		# create requests to mend all the wrong name.
		for idx, (class_id, wrong_name) in tqdm(enumerate(wrong_names), total=len(wrong_names)):
			if no_extra_knowledge and class_id in wrong_classes:
				continue
			request = {}
			if class_id in wrong_classes:
				request["txt_align"] = False
				request["use_real_noise"] = True
			elif use_imgnet_imgs:
				request["txt_align"] = True
				request["use_real_noise"] = True
			else:
				request["txt_align"] = True
				request["use_real_noise"] = False
			# find the class name with highest score
			highest_score = 0
			highest_name = None
			for class_name, class_score in summary[class_id].items():
				highest_score = max(highest_score, class_score["mean"])
				highest_name = class_name if highest_score == class_score["mean"] \
											else highest_name
			if use_imgnet_imgs:
				request["dest"] = "real " + highest_name
			else:
				request["dest"] = highest_name
		
			request["source"] = wrong_name
			request["source id"] = class_id
			request["dest id"] = class_id

			# find the prompt and seeds for this request
			if type == "val":
				prompts = []
				seeds = []
				indices = []
				for item in prompt_data:
					if int(item["class id"]) == int(class_id):
						# lower the prompt and replace the class name with {}
						prompt = item["text prompt"].lower()
						prompt = prompt.replace(item["class name"].lower(), "{}")
						prompts.append(prompt)
						seeds.append(item["random seed"])
						indices.append(item["idx"])

				if len(prompts) == 0:
					# print(f"Cannot find prompts for class {class_id}.")
					continue
				request["prompts"] = prompts[prompt_slice]
				request["seeds"] = seeds[prompt_slice]
				request["indices"] = indices[prompt_slice]

				if use_imgnet_imgs:
					success = False
					request["images"] = []
					for img, label in dataset:
						if label == int(class_id):
							request["images"].append(img)
						if len(request["images"]) == imgs_per_class:
							success = True
							break
					if not success:
						raise ValueError(f"Cannot find enough images for class {class_id}")
				self.requests.append(request)
			else:
				seeds = []
				for item in prompt_data:
					if int(item["class id"]) == int(class_id):
						# lower the prompt and replace the class name with {}
						seeds.append(item["random seed"])
				if len(seeds) == 0:
					# print(f"Cannot find prompts for class {class_id}.")
					continue
				request["prompts"] = edit_prompts_tmp[:prompts_per_request]
				request["seed_train"] = seeds_train[idx]
				request["seeds"] = seeds[:prompts_per_request]
				self.requests.append(request)
		
		if type == "edit":
			# set indices for requests
			for idx, request in enumerate(self.requests):
				request["indices"] = [idx * prompts_per_request + i 
						  			  for i in range(prompts_per_request)]
		
		# remove conflict requests
		# conflict means the source name is the same but the dest name is different
		# or some request's dest is the same as another request's source
		to_remove = []
		for idx, request in enumerate(self.requests[:]):
			for other_request in self.requests[:]:
				if request["source"] == other_request["source"] \
						and request["dest"] != other_request["dest"]:
					to_remove.append(request)
				elif request["dest"] == other_request["source"]:
					to_remove.append(other_request)
		for item in to_remove:
			try:
				self.requests.remove(item)
			except ValueError:
				pass
		
		# set indices for requests
		print(f"Created {len(self.requests)} requests.")

	def __len__(self):
		return len(self.requests)
	
	def __getitem__(self, idx):
		return self.requests[idx]
	

class CustomObjectRequestDataset(Dataset):
	def __init__(
			self, 
			data_dir: str=DATA_DIR, 
			dataset_dir: str="dream_booth_dataset",
			batch_size: int=3
	):
		self.dataset_dir = Path(data_dir) / dataset_dir

		if not self.dataset_dir.exists():
			raise FileNotFoundError(f"File {self.dataset_dir} does not exist.")
		
		self.requests: List[Dict] = []

		for name in os.listdir(self.dataset_dir):
			if os.path.isdir(self.dataset_dir / name):
				# add new request
				request = {}
				request["source"] = self.find_source_name(name)
				request["dest"] = f"{name}"
				print(name)
				request["prompts"] = [
					"an image of {}", 
					"a photo of {}",
					"{}, a picture"
				]
				request["seed"] = 37
				request["training_img_paths"] = []
				request["txt_img_align"] = True
				for idx, file in enumerate(os.listdir(self.dataset_dir / name)):
					if idx == batch_size:
						break
					request["training_img_paths"].append(
						str(self.dataset_dir / name / file))
				self.requests.append(request)
	
	def __len__(self):
		return len(self.requests)
	
	def __getitem__(self, idx):
		return self.requests[idx]
	
	def find_source_name(self, dest_name):
		record_file = self.dataset_dir / "prompts_and_classes.txt"
		with open(record_file, "r") as f:
			lines = f.readlines()
		for line in lines:
			if dest_name in line:
				return line.split(",")[1].strip()
		
		raise ValueError(f"Cannot find source name for dest {dest_name}.")



class ObjectPromptDataset(Dataset):
	def __init__(
			self, 
			data_dir: str=DATA_DIR, 
			dataset_name: str="imgnet_prompts",
			file_name: str="imgnet_prompts.json"):
		self.file_loc = Path(data_dir) / dataset_name / file_name
		self.data = []
		with open(self.file_loc, "r") as file:
			data = json.load(file)
			for row in data:
				self.data.append(row)

		print(f"Loaded {len(self.data)} prompts from {self.file_loc}")

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		return self.data[idx]
	

class RequestDataset(Dataset):
	def __init__(
			self, 
			data_dir: str=DATA_DIR, 
			dataset_dir: str="iceb_data",
			file_name: str="imgnet_aug_edit.json", 
			type: Literal["edit", "val"]="edit",
			num_negative_prompts: int=0,
			txt_img_align: bool=False):
		self.file_loc = Path(data_dir) / dataset_dir / file_name
		if not self.file_loc.exists():
			raise FileNotFoundError(f"File {self.file_loc} does not exist.")
		
		if num_negative_prompts > 0:
			# fetch negative prompts from ccs_filtered_sub.json
			with open("data/ccs_filtered_sub.json", "r") as f:
				ccs_data = json.load(f)
			ccs_data = random.sample(ccs_data, num_negative_prompts)
			negative_prompts = []
			for item in ccs_data:
				negative_prompts.append(item["caption"])
		else:
			negative_prompts = []
		
		self.requests: List[Dict] = []
		with open(self.file_loc, "r") as file:
			data = json.load(file)
		
		if type == "edit":
			prompt_slice = slice(0, EDITING_PROMPTS_CNT)
		elif type == "val":
			# collect all the prompts
			prompt_slice = slice(None)
		else:
			raise ValueError(f"Invalid type {type}.")

		edit_prompts_tmp = [
			"An image of {}",
			"A photo of {}",
			"{}"
		]   
		random.seed(2023)
		# add training seeds
		training_seeds = random.sample(range(10000), len(data) // 5)

		last_class = None
		last_class_id = None
		last_dest = None
		last_dest_id = None
		prompts = []
		indices = []
		seeds = []
		for idx, row in enumerate(data):
			class_name = row["class name"]
			if (last_class and class_name != last_class) or idx == len(data) - 1:
				if idx == len(data) - 1:
					prompts.append(row["text prompt"])
					indices.append(row["idx"])
					seeds.append(row["random seed"])
				if type == "edit":
					prompts = edit_prompts_tmp[prompt_slice]
				else:
					prompts = prompts[prompt_slice]
				self.requests.append(
					{"prompts": prompts, 
					 "source": last_class,
					 "seeds": seeds[prompt_slice], 
					 "seed_train": training_seeds[idx // 5],
					 "indices": indices[prompt_slice], 
					 "source id": last_class_id, 
					 "dest": last_dest, 
					 "dest id": last_dest_id,
					 "negative prompts": negative_prompts,
					 "txt_img_align": txt_img_align})
				prompts = []
				indices = []
				seeds = []
				prompts.append(row["text prompt"])
				indices.append(idx)
				seeds.append(row["random seed"])

				last_class = row["class name"]
				last_class_id = row["class id"]
				last_dest = row["dest"]
				last_dest_id = row["dest id"]
				continue
			prompts.append(row["text prompt"])
			indices.append(row["idx"])
			seeds.append(row["random seed"])
			last_class = row["class name"]
			last_class_id = row["class id"]
			last_dest = row["dest"]
			last_dest_id = row["dest id"]
		
		print(f"Loaded {len(self.requests)} requests from {self.file_loc}")
	
	def __len__(self):
		return len(self.requests)
	
	def __getitem__(self, idx):
		return self.requests[idx]
	
	def sample(self, num: int, seed: int=None):
		if seed:
			torch.manual_seed(seed)
		indices = torch.randperm(len(self.requests))[:num]
		return [self.requests[idx] for idx in indices]


def requests_to_csv(requests: List[Dict], out_file: str):

	# create a new df based on requests
	keys = requests[0].keys()
	keys_to_use = [
		"source",
		"dest",
		"prompts",
		"seed_train",
		"seeds",
		"source id",
		"dest id"
	]
	if "val" in out_file:
		keys_to_use.remove("seed_train")

	df = pd.DataFrame(columns=keys_to_use)

	for request in requests:
		new_row = {}
		for key in keys_to_use:
			new_row[key] = request[key]
		new_df = pd.DataFrame([new_row])
		df = pd.concat([df, new_df], ignore_index=True)
	
	# save the df to csv
	df.to_csv(out_file, index=False)


def compose_alias_test_requests(val_requests: List[Dict]):
	"""
	Given a list of val_resquests, we find the classes that has multiple labels,
	and create a new request for each alias of the class using the same prompts
	"""

	with open("data/iceb_data/vit_classifier_config.json") as f:
		vit_classifier_config = json.load(f)
		id2label = vit_classifier_config["id2label"]

	# find the classes that has multiple labels
	alias_idxs = []
	for request in val_requests:
		if len(id2label[str(request["source id"])].split(",")) > 1:
			new_labels = id2label[str(request["source id"])].split(",")
			new_labels.remove(request["source"])
			for new_label in new_labels:
				alias_idxs.append((new_label, request["source id"]))
	print("number of alias prompts: ", len(alias_idxs))

	# for every different name of a class, we create a new request
	# with the same prompts and dest, but changing the source to its alias
	new_requests = []
	for item in alias_idxs:
		for request in val_requests:
			if request["source id"] == item[1]:
				new_request = request.copy()
				new_request["source"] = item[0]
				new_requests.append(new_request)
	return new_requests


def edit_test_split(
		num_edit=300, 
		k_nb=5,
		filtered_file="imgnet_prompts_filtered.json",
		edit_file="imgnet_small_edit_aug.json",
		test_file="objects_test_aug.json"):
	"""
	Load filtered dataset and split it into edit and test set.
	"""
	filtered_dataset = ObjectPromptDataset(file_name=filtered_file)

	edit_set = []
	test_set = []
	all_class_ids = list(set([item["class id"] for item in filtered_dataset]))
	edit_class_indices = np.random.choice(all_class_ids, num_edit, replace=False)

	dest_set = find_dest(all_class_ids, edit_class_indices, k_nb=k_nb)

	for item in filtered_dataset:
		if item["class id"] in edit_class_indices:
			# find this item's dest
			for dest in dest_set:
				if dest["class id"] == item["class id"]:
					item["dest"] = dest["dest"]
					item["dest id"] = dest["dest id"]
					break
			item["text prompt"] = item["text prompt"].lower()
			item["text prompt"] = item["text prompt"].replace(item["class name"], "{}")
			edit_set.append(item)
		else:
			test_set.append(item)
	assert len(edit_set) == num_edit * 5, f"len(edit_set) = {len(edit_set)}"

	# save the edit set
	with open("./data/iceb_data/" + edit_file, "w") as f:
		json.dump(edit_set, f, indent=4)

	# save the test set
	with open("./data/iceb_data/" + test_file, "w") as f:
		json.dump(test_set, f, indent=4)


def find_dest(
		all_class_indices: List[int], 
		edit_class_indices: List[int],
		k_nb: int,
		device: str="cuda:0"):
	"""
	We sample 1 class from the k nearest classes to the edit class, the distance
	is measured by clip text encoder.
	"""
	model_id = "openai/clip-vit-large-patch14"
	tokenizer = CLIPTokenizer.from_pretrained(model_id)
	model = CLIPTextModel.from_pretrained(model_id)
	model.to(device)
	model.eval()

	# load the id2label mapping
	with open("data/iceb_data/vit_classifier_config.json", "r") as f:
		id2label = json.load(f)["id2label"]
	
	template = "an image of {}"
	dest_set = []

	def _cal_similarity(name1, name2):
		text = [template.format(name1), template.format(name2)]
		inp = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
		inp = {k: v.to(device) for k, v in inp.items()}
		with torch.no_grad():
			outputs = model(**inp)
			text_reprs = outputs["pooler_output"]   # (2, hidden_size)
		# compute cosine similarity
		return torch.cosine_similarity(text_reprs[0], text_reprs[1], dim=0).item()
	
	# remove edit_class_indices from all_class_indices to get test_class_indices
	test_class_indices = list(set(all_class_indices) - set(edit_class_indices))
	
	for edit_class_idx in tqdm(edit_class_indices):
		# find the nearest k classes
		distances = []
		for test_class_idx in test_class_indices:
			test_name = id2label[str(test_class_idx)].split(",")[0]
			edit_name = id2label[str(edit_class_idx)].split(",")[0]
			distances.append(
				(test_class_idx, 
				 test_name, 
				 _cal_similarity(test_name, edit_name)))

		distances.sort(key=lambda x: x[-1], reverse=True)
		nearest_items = [item for item in distances[:k_nb]]

		# sample 1 class from the nearest classes
		random.seed(2023)
		dest_item = random.choice(nearest_items)
		dest_set.append({
			"class name": test_name, 
			"class id": edit_class_idx,
			"dest": dest_item[1],
			"dest id": dest_item[0]})

	return dest_set 


def generate_class2id(outpath="./data/iceb_data/class2id.json"):
	with open("./data/iceb_data/imgnet_prompts.json", "r") as f:
		data = json.load(f)

	class2id = {}
	for item in data:
		class2id[item["class name"]] = item["class id"]
	
	with open(outpath, "w") as f:
		json.dump(class2id, f, indent=4)

	return class2id


def get_filtered_dataset(
	data_file="./data/iceb_data/imgnet_prompts_aug.json",
	out_file="./data/iceb_data/imgnet_prompts_filtered.json",
):
	"""
	only keep the classes that have 5 prompts checked.
	"""
	object_dataset = ObjectPromptDataset(file_name=data_file.split("/")[-1]) 
	class_cnt = np.full((1000,), 5)
	for idx, item in enumerate(object_dataset):
		if item.get("checked", False):
			pass
		else:
			class_idx = idx // 5
			class_cnt[class_idx] -= 1
	full_class_indices = [i for i in range(1000) if class_cnt[i] == 5]
	full_class_cnt = len(full_class_indices)
	print(full_class_cnt)
	# make a new dataset use the full_class_indices
	new_dataset = []
	for idx, item in enumerate(object_dataset):
		if idx // 5 in full_class_indices:
			new_dataset.append(item)
	
	assert len(new_dataset) == full_class_cnt * 5
	# save the new dataset
	with open(out_file, "w") as f:
		json.dump(new_dataset, f, indent=4)


if __name__ == "__main__":
	requests = ImageNetMendRequestDataset(type="edit")
	requests_to_csv(requests, "data/iceb_data/rectification_train.csv")
	requests = ImageNetMendRequestDataset(type="val")
	requests_to_csv(requests, "data/iceb_data/rectification_val.csv")

	# print out all the sources of the requests
	sources = []

	for request in requests:
		sources.append(request["source"])
	print(set(sources))