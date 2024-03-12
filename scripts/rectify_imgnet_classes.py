import json
from typing import List, Dict
from argparse import ArgumentParser

import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import (
	AutoProcessor,
	ViTForImageClassification
)

from util.evaluate import calculate_single_cls_score
from util.globals import *
from emcid.emcid_hparams import EMCIDHyperParams, UNetEMCIDHyperParams
from emcid.emcid_main import apply_emcid_to_text_encoder, apply_emcid_to_cross_attn
from experiments.emcid_test import set_weights

from emcid.uce_train import edit_model_uce, edit_model_uce_modified, edit_text_encoder_uce
from dsets.iceb_dataset import *
from dsets.artist_requests import ArtistRequestsDataset
from dsets.global_concepts import get_i2p_editing_requests, NSFWEditRequestDataset

class ImageNetMendRequestDataset_simple(Dataset):
	def __init__(
		self,
		class_summary_file="data/iceb_data/imgnet_prompts_cls.json",
		prompt_file="data/iceb_data/imgnet_aug_full.json",
		type: Literal["edit", "val"]="edit",
		no_extra_knowledge=True,   # select classes that have at least one name with high score
		use_imgnet_imgs=False,
		imgs_per_class=3,
		class_score_threshold=0.5,
		name_score_threshold=0.1,
		prompts_per_request=3,
		use_simple_train_prompt=True,
		verbose=False,
		aliases_to_mend: List[str]=None,
		max_loading_num=None
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

			if aliases_to_mend is not None and wrong_name.lower() not in aliases_to_mend:
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
			
			if wrong_name.lower() == "appenzeller":
				wrong_name = "appenzeller sennenhund"
		
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
					print(f"Cannot find prompts for class {class_id}.")
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
				request["prompts"] = edit_prompts_tmp[:prompts_per_request]
				request["seed_train"] = seeds_train[idx]
				request["seeds"] = seeds[:prompts_per_request]
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
				if max_loading_num and len(self.requests) >= max_loading_num:
					break
		
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


def execute_imgnet_mend_noise_only(
		aliases_to_mend,
		hparam_name,
		mom2_weight=4000,
		edit_weight=0.6,
		eval_sample_per_prompt=6,
		device="cuda:0"):
	# get the ediiting requests
	
	hparams = EMCIDHyperParams.from_json(f"hparams/{hparam_name}.json")

	hparams = set_weights(hparams, mom2_weight, edit_weight)
	mom2_weight = hparams.mom2_update_weight
	edit_weight = hparams.edit_weight

	aliases_to_mend_lower = [alias.lower() for alias in aliases_to_mend]
	requests = \
		ImageNetMendRequestDataset_simple(
			type="edit",
			no_extra_knowledge=False,
			use_imgnet_imgs=True,
			imgs_per_class=eval_sample_per_prompt,
			class_score_threshold=0.02,
			aliases_to_mend=aliases_to_mend_lower,
			)
	print(len(requests))
	demo_requests = requests
	# demo_requests = [] if aliases_to_mend is not None else requests
	
	# for request in requests:
	# 	if request["source"].lower() in aliases_to_mend_lower:
	# 		demo_requests.append(request)
	# print(len(demo_requests))
	
	pipe = StableDiffusionPipeline.from_pretrained(
		"CompVis/stable-diffusion-v1-4",
		torch_dtype=torch.float32,
		safety_checker=None,
		requires_safety_checker=False,
	).to(device)

	pipe.set_progress_bar_config(disable=True)

	cache_name = f"cache/{hparam_name}/imgnet_mend/"

	# generate pre edit images

	# save images
	save_dir = f"{RESULTS_DIR}/emcid/{hparam_name}/imgnet_mend_hard/visual"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	with torch.no_grad():
		# generate pre edit source images
		for request in demo_requests:
			if not os.path.exists(f"{save_dir}/pre_edit_source_{request['source']}"):
				os.makedirs(f"{save_dir}/pre_edit_source_{request['source']}")

			for idx in range(eval_sample_per_prompt):
				if os.path.exists(f"{save_dir}/pre_edit_source_{request['source']}/{idx}.png"):
					continue
				generator = torch.Generator(pipe.device).manual_seed(int(idx)) if request["seed_train"] is not None else None
				img = pipe(
					[request["prompts"][0].format(request["source"])], 
					generator=generator, 
					guidance_scale=7.5).images[0]
				
				# save the image
				img.save(f"{save_dir}/pre_edit_source_{request['source']}/{idx}.png")
			# also save a ground truth image
			gt_images = request["images"]
			for idx, img in enumerate(gt_images):
				img.save(f"{save_dir}/pre_edit_source_{request['source']}/{idx}_gt.png")

	
	new_pipe, _ = apply_emcid_to_text_encoder(
						pipe, 
						requests=demo_requests, 
						hparams=hparams, 
						device=pipe.device, 
						cache_name=cache_name)
	
	post_edit_source_imgs = []
	post_edit_dest_imgs = []
	with torch.no_grad():
		# generate post edit source images
		for request in demo_requests:
			generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
			for idx in range(eval_sample_per_prompt):
				img = new_pipe(
					[request["prompts"][0].format(request["source"])], 
					generator=generator, 
					guidance_scale=7.5).images[0]
				post_edit_source_imgs.append((img, request["source"], idx))

		# generate post edit dest images
		# for request in demo_requests:
		# 	generator = torch.Generator(pipe.device).manual_seed(int(request["seed_train"])) if request["seed_train"] is not None else None
		# 	for idx in range(eval_sample_per_prompt):
		# 		img = new_pipe(
		# 			[request["prompts"][0].format(request["dest"])], 
		# 			generator=generator, 
		# 			guidance_scale=7.5).images[0]
		# 		post_edit_dest_imgs.append((img, request["dest"], idx))
	
	for idx, item in enumerate(post_edit_source_imgs):
		if not os.path.exists(f"{save_dir}/post_edit_source_{item[1]}"):
			os.makedirs(f"{save_dir}/post_edit_source_{item[1]}")
		for idx in range(eval_sample_per_prompt):
			item[0].save(f"{save_dir}/post_edit_source_{item[1]}/{item[2]}.png")

	# for idx, item in enumerate(post_edit_dest_imgs):
	# 	if not os.path.exists(f"{save_dir}/post_edit_dest_{item[1]}"):
	# 		os.makedirs(f"{save_dir}/post_edit_dest_{item[1]}")
	# 	for idx in range(eval_sample_per_prompt):
	# 		item[0].save(f"{save_dir}/post_edit_dest_{item[1]}/{item[2]}.png")




if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--device", type=str, default="cuda:0")

	args = parser.parse_args()


	aliases_to_mend = [
		# "alligator lizard",
		# "boa constrictor",
		"Appenzeller",
		"kelpie",
		# "sorrel",
		"rock beauty",
		# "beaker",
		"shoji",
		# "marmoset",
		# "Capuchin",
		# "crutch",
		# "Brabancon griffon",
		"poke bonnet",
		"mortar",
		"bolete"
		]

	execute_imgnet_mend_noise_only(
		aliases_to_mend,
		hparam_name="dest_s-1000_c-1.5_ly-11_lr-0.2_wd-5e-04_true_noise_loss",
		eval_sample_per_prompt=6*3,
		device=args.device)