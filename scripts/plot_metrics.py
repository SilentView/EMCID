import os
import json
from typing import List, Literal
from argparse import ArgumentParser


import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np


def extract_stat_emcid(content, hparam_name):
	"""
	Here the invariant is that the keys in content are of the form 
	edit{edit_num}_weight{weight}_ew{edit_weight}
	return a dict, whose keys are weight{weight}_ew{edit_weight}, and values are
	list of metrics of different edit numbers
	"""
	all_keys = list(content.keys())

	def _divide_keys_by_mom2_and_edit_weight(keys):
		weight_strs = set([extract_weight_str(key) for key in keys])

		keys_by_weight = {}
		for weight_str in weight_strs:
			keys_by_weight[weight_str] = []
		for key in keys:
			keys_by_weight[extract_weight_str(key)].append(key)
		return keys_by_weight

	divided_keys = _divide_keys_by_mom2_and_edit_weight(all_keys)
	to_ret = {}
	for weight_str, keys in divided_keys.items():
		keys.sort(key=lambda x: extract_edit_num_and_mom2_weight(x)[0])

		efficacy_source_forget = []
		pre_source_score = []
		efficacy_source2dest = []
		pre_dest_score = []
		pre_source_dest_cls_score_edit = []
		efficacy_dest_forget = []

		generalization_source_forget = []
		pre_source_score_general = []
		generalization_source2dest = []
		pre_dest_score_general = []
		pre_source_dest_cls_score_general = []
		generalization_dest_forget = []

		generalization_alias_forget = []
		generalization_alias2dest = []
		pre_alias_score = []
		pre_source_dest_cls_score_alias = []

		specificity_delta= []
		ng_pre_specificity = []
		for key in keys:
			efficacy_source_forget.append(
				[content[key]["pre_source_cls_score_edit"] - content[key]["post_source_cls_score_edit"],
				extract_edit_num(key)]
				)
			pre_source_score.append([content[key]["pre_source_cls_score_edit"], extract_edit_num(key)])

			pre_source_dest_cls_score_edit.append([content[key]["pre_source_dest_cls_score_edit"], extract_edit_num(key)])
			pre_source_dest_cls_score_alias.append([content[key]["pre_source_dest_cls_score_alias"], extract_edit_num(key)])
			pre_source_dest_cls_score_general.append([content[key]["pre_source_dest_cls_score_general"], extract_edit_num(key)])

			efficacy_source2dest.append(
				[content[key]["post_source_dest_cls_score_edit"] - content[key]["pre_source_dest_cls_score_edit"],
				extract_edit_num(key)]
				)
			pre_dest_score.append([content[key]["pre_dest_cls_score_edit"], extract_edit_num(key)])

			efficacy_dest_forget.append(
				[content[key]["pre_dest_cls_score_edit"] - content[key]["post_dest_cls_score_edit"],
				extract_edit_num(key)]
				)
			
			generalization_source_forget.append(
				[content[key]["pre_source_cls_score_general"] - content[key]["post_source_cls_score_general"],
				extract_edit_num(key)]
				)
			pre_source_score_general.append([content[key]["pre_source_cls_score_general"], extract_edit_num(key)])

			generalization_source2dest.append(
				[content[key]["post_source_dest_cls_score_general"] - content[key]["pre_source_dest_cls_score_general"],
				extract_edit_num(key)]
				)
			pre_dest_score_general.append([content[key]["pre_dest_cls_score_general"], extract_edit_num(key)])

			generalization_dest_forget.append(
				[content[key]["pre_dest_cls_score_general"] - content[key]["post_dest_cls_score_general"],
				extract_edit_num(key)]
				)

			generalization_alias2dest.append(
				[content[key]["post_source_dest_cls_score_alias"] - content[key]["pre_source_dest_cls_score_alias"],
				extract_edit_num(key)]
				)
			
			specificity_delta.append(
				[content[key]["post_cls_score_specificity"] - content[key]["pre_cls_score_specificity"],
				extract_edit_num(key)]
				)
			ng_pre_specificity.append([- content[key]["pre_cls_score_specificity"], extract_edit_num(key)])
		
		# deal with alias alone, TODO: temporary solution, remove this loop into last loop before release
		try:
			for key in keys:
				generalization_alias_forget.append(
						[content[key]["pre_source_cls_score_alias"] - content[key]["post_source_cls_score_alias"],
						extract_edit_num(key)]
						)
				pre_alias_score.append([content[key]["pre_source_cls_score_alias"], extract_edit_num(key)])
		except:
			pre_alias_score = []
			generalization_alias_forget = []

		to_ret[weight_str] = {
		"efficacy_source_forget": efficacy_source_forget,
		"pre_source_score": pre_source_score,
		"efficacy_source2dest": efficacy_source2dest,
		"pre_dest_score": pre_dest_score,
		"efficacy_dest_forget": efficacy_dest_forget,
		"pre_source_dest_cls_score_edit": pre_source_dest_cls_score_edit,
		"pre_source_dest_cls_score_general": pre_source_dest_cls_score_general,
		"pre_source_dest_cls_score_alias": pre_source_dest_cls_score_alias,


		"generalization_source_forget": generalization_source_forget,
		"pre_source_score_general": pre_source_score_general,
		"generalization_source2dest": generalization_source2dest,
		"pre_dest_score_general": pre_dest_score_general,
		"generalization_dest_forget": generalization_dest_forget,
		"generalization_alias_forget": generalization_alias_forget,
		"generalization_alias2dest": generalization_alias2dest,
		"pre_alias_score": pre_alias_score,

		"specificity_delta": specificity_delta,
		"ng_pre_specificity": ng_pre_specificity,
		"hparam_name": hparam_name}

	return to_ret



def extract_edit_num_and_mom2_weight(key):
	"""input belike edit{edit_num}_weight{mom2_weight}_ew{edit_weight}"""
	return int(key.replace("edit", "").split("_")[0]), key.split("_")[1]

def extract_edit_num(key):
	"""input belike edit{edit_num}_weight{mom2_weight}_ew{edit_weight}"""
	return int(key.replace("edit", "").split("_")[0])

def extract_mom2_weight_str(key):
	"""
	input belike edit{edit_num}_weight{mom2_weight}_ew{edit_weight}
	return weight{mom2_weight}
	"""
	return key.split("_")[1]

def extract_weight_str(key):
	"""
	input belike edit{edit_num}_weight{mom2_weight}_ew{edit_weight}
	return weight{mom2_weight}_ew{edit_weight}
	"""
	if extract_edit_weight(key) is None:
		return extract_mom2_weight_str(key)
	else:
		return f"{extract_mom2_weight_str(key)}_ew{extract_edit_weight(key)}"


def extract_edit_weight(key):
	"""
	input belike edit{edit_num}_weight{mom2_weight}_ew{edit_weight}
	or edit{edit_num}_weight{mom2_weight}
	"""
	if "ew" in key:
		return float(key.replace("ew", "").split("_")[2])
	else:
		return None


def extract_stat_baseline(content, name):
	"""
	Here the invariant is that the keys in content are of the form edit{edit_num},
	used for baselines
	"""
	efficacy_source_forget = []
	pre_source_score = []
	efficacy_source2dest = []
	pre_dest_score = []
	efficacy_dest_forget = []

	generalization_source_forget = []
	pre_source_score_general = []
	generalization_source2dest = []
	pre_dest_score_general = []
	generalization_dest_forget = []

	specificity_delta= []
	ng_pre_specificity = []

	generalization_alias_forget = []
	generalization_alias2dest = []
	pre_alias_score = []

	keys = list(content.keys())
	
	# sore keys
	keys.sort(key=lambda x: int(x.replace("edit", "")))
	for key in keys:
		efficacy_source_forget.append(
			[content[key]["pre_source_cls_score_edit"] - content[key]["post_source_cls_score_edit"],
			int(key.replace("edit", ""))]
			)
		pre_source_score.append([content[key]["pre_source_cls_score_edit"], int(key.replace("edit", ""))])

		efficacy_source2dest.append(
			[content[key]["post_source_dest_cls_score_edit"] - content[key]["pre_source_dest_cls_score_edit"],
			int(key.replace("edit", ""))]
			)
		pre_dest_score.append([content[key]["pre_dest_cls_score_edit"], int(key.replace("edit", ""))])

		efficacy_dest_forget.append(
			[content[key]["pre_dest_cls_score_edit"] - content[key]["post_dest_cls_score_edit"],
			int(key.replace("edit", ""))]
			)
		
		generalization_source_forget.append(
			[content[key]["pre_source_cls_score_general"] - content[key]["post_source_cls_score_general"],
			int(key.replace("edit", ""))]
			)
		pre_source_score_general.append([content[key]["pre_source_cls_score_general"], int(key.replace("edit", ""))])

		generalization_source2dest.append(
			[content[key]["post_source_dest_cls_score_general"] - content[key]["pre_source_dest_cls_score_general"],
			int(key.replace("edit", ""))]
			)
		pre_dest_score_general.append([content[key]["pre_dest_cls_score_general"], int(key.replace("edit", ""))])

		generalization_dest_forget.append(
			[content[key]["pre_dest_cls_score_general"] - content[key]["post_dest_cls_score_general"],
			int(key.replace("edit", ""))]
			)
		
		generalization_alias_forget.append(
			[content[key]["pre_source_cls_score_alias"] - content[key]["post_source_cls_score_alias"],
			int(key.replace("edit", ""))]
			)
		generalization_alias2dest.append(
			[content[key]["post_source_dest_cls_score_alias"] - content[key]["pre_source_dest_cls_score_alias"],
			int(key.replace("edit", ""))]
			)

		pre_alias_score.append([content[key]["pre_source_cls_score_alias"], int(key.replace("edit", ""))])
		
		specificity_delta.append(
			[content[key]["post_cls_score_specificity"] - content[key]["pre_cls_score_specificity"],
			int(key.replace("edit", ""))]
			)
		ng_pre_specificity.append([- content[key]["pre_cls_score_specificity"], int(key.replace("edit", ""))])

	return {
		"efficacy_source_forget": efficacy_source_forget,
		"pre_source_score": pre_source_score,
		"efficacy_source2dest": efficacy_source2dest,
		"pre_dest_score": pre_dest_score,
		"efficacy_dest_forget": efficacy_dest_forget,
		"generalization_source_forget": generalization_source_forget,
		"pre_source_score_general": pre_source_score_general,
		"generalization_source2dest": generalization_source2dest,
		"pre_dest_score_general": pre_dest_score_general,
		"generalization_dest_forget": generalization_dest_forget,
		"generalization_alias_forget": generalization_alias_forget,
		"generalization_alias2dest": generalization_alias2dest,
		"specificity_delta": specificity_delta,
		"ng_pre_specificity": ng_pre_specificity,
		"name": name
	}

def plot_edit_results_simple(
		results_paths=["results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary.json"], 
		save_path="results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary/summary.png"):
	
	contents = []
	stats = []
	hparam_names = [path.split("/")[-2] for path in results_paths]

	for results_path in results_paths:
		with open(results_path, 'r') as f:
			contents.append(json.load(f))
	for content, hparam_name in zip(contents, hparam_names):
		stats.append(extract_stat_baseline(content, hparam_name))

	# plot on 3 subplots
	fig, axs = plt.subplots(3, 1, figsize=(10, 10), tight_layout=True)
	def _get_x_y(xys):
		x = [item[1] for item in xys]
		y = [item[0] for item in xys]
		return x, y
	
	def _plot_stat(axs, stat, color="orange"):
		axs[0].plot(*_get_x_y(stat["efficacy_source_forget"]), label="source forget", marker="o", linestyle="-", color=color)
		axs[0].plot(*_get_x_y(stat["efficacy_source2dest"]), label="source2dest", marker="o", linestyle=":", color=color)
		# axs[0].plot(*_get_x_y(stat["efficacy_dest_forget"]), label="dest forget", marker="o", linestyle="--", color=color)

		axs[1].plot(*_get_x_y(stat["generalization_source_forget"]), label="source forget", marker="o", linestyle="-", color=color)
		axs[1].plot(*_get_x_y(stat["generalization_source2dest"]), label="source2dest", marker="o", linestyle=":", color=color)
		# axs[1].plot(*_get_x_y(stat["generalization_dest_forget"]), label="dest forget", marker="o", linestyle="--", color=color)

		axs[2].plot(*_get_x_y(stat["specificity_delta"]), label="specificity delta", marker="o", color=color)
	
	def _set_legend(axs, loc, frameon=False):
		print(f"setting legend to {loc}")
		for ax in axs:
			ax.legend(loc=loc, frameon=frameon)

	colors = ["orange", "blue", "green", "red", "purple", "brown"]
	for stat, color in zip(stats, colors):
		_plot_stat(axs, stat, color)
	
	_set_legend(axs, "upper right", frameon=False)


	axs[0].set_title("Efficacy")
	axs[1].set_title("Generalization")
	axs[2].set_title("Specificity")

	# set xlabels
	axs[2].set_xlabel("Edit Number")

	# set xticks
	for ax in axs:
		xs = _get_x_y(stats[0]["generalization_source_forget"])[0]
		ax.set_xticks(xs)
		ax.set_xticklabels([str(x) for x in xs])
	
	# white facecolor
	# for ax in axs:
	#     ax.set_facecolor("white")

	# set ylabels
	axs[0].set_ylabel("Cls Score")
	axs[1].set_ylabel("Cls Score")
	axs[2].set_ylabel("Cls Score")

	if not os.path.exists(os.path.dirname(save_path)):
		os.makedirs(os.path.dirname(save_path))

	plt.savefig(save_path)
	# plt.show()


def plot_edit_results_full_two_row(
		emcid_results_paths=["results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary.json"], 
		baseline_results_paths=None,
		save_path="results/emcid/summary.png",
		exclude=None,
		xs=[1, 5, 10, 20, 30, 40, 50],
		no_alias=False,
		plot_bound=False):
	
	emcid_contents = []
	emcid_stats = []
	hparam_names = [path.split("/")[-2] for path in emcid_results_paths]

	baseline_contents = []
	baseline_stats = []
	baseline_names = [path.split("/")[-2] for path in baseline_results_paths]\
		if baseline_results_paths is not None else None

	for results_path in emcid_results_paths:
		with open(results_path, 'r') as f:
			try:
				emcid_contents.append(json.load(f))
			except:
				print(f"failed to load {results_path}")
				raise ValueError("failed to load")
	for content, hparam_name in zip(emcid_contents, hparam_names):
		emcid_stats.append(extract_stat_emcid(content, hparam_name))
	
	if baseline_results_paths is not None:
		for results_path in baseline_results_paths:
			with open(results_path, 'r') as f:
				baseline_contents.append(json.load(f))
		for content, name in zip(baseline_contents, baseline_names):
			baseline_stats.append(extract_stat_baseline(content, name))

	fig = plt.figure(figsize=(16, 7))
	# plt.rcParams.update({'font.size': 22})
	plt.subplots_adjust(wspace=0.26, hspace=0.3)

	legend_font = 16
	x_label_font = 10
	y_label_font = 10
	title_font_size = 14
	marker_size = 7



	gs = GridSpec(2, 6)

	def _get_x_y(xys, max_x=None, min_x=None, xs=None):
		new_xys = xys[:]
		if max_x is not None:
			new_xys = [item for item in new_xys if item[1] <= max_x]
		if min_x is not None:
			new_xys = [item for item in new_xys if item[1] >= min_x]

		if xs is not None:
		# only keep the xys whose x is in xs
			new_xys = [item for item in new_xys if item[1] in xs]
		x = [item[1] for item in new_xys]
		y = [item[0] for item in new_xys]
		return x, y

	slices = [
		gs[0, 0:2], gs[0, 2:4], gs[0, 4:6],
		gs[1, 0:2], gs[1, 2:4], gs[1, 4:6]]
	
	# metrics = [
	# 	"efficacy_source_forget", "efficacy_source2dest", "specificity_delta",
	# 	"generalization_source_forget", "generalization_source2dest", "generalization_alias_forget"]
	# metrics = [
	# 	"efficacy_source_forget", "efficacy_source2dest", "specificity_delta",
	# 	"generalization_source_forget", "generalization_source2dest", "generalization_alias2dest"]
	
	metrics = [
		"specificity_delta", "efficacy_source_forget", "efficacy_source2dest", 
		"generalization_source_forget", "generalization_source2dest", "generalization_alias2dest"]
	
	# metrci2bounds = {
	# 	"efficacy_source_forget": "pre_source_score",
	# 	"efficacy_source2dest": "pre_dest_score",
	# 	"generalization_source_forget": "pre_source_score_general",
	# 	"generalization_source2dest": "pre_dest_score_general",
	# 	"specificity_delta": "ng_pre_specificity",
	# 	"generalization_alias_forget": "pre_alias_score"
	# }

	metrci2bounds = {
		"efficacy_source_forget": "pre_source_score",
		"efficacy_source2dest": "pre_source_dest_cls_score_edit",
		"generalization_source_forget": "pre_source_score_general",
		"generalization_source2dest": "pre_source_dest_cls_score_general",
		"specificity_delta": "ng_pre_specificity",
		"generalization_alias2dest": "pre_source_dest_cls_score_alias"
	}

	metric2titles = {
		"efficacy_source_forget": r"Efficacy: Source Forget$\uparrow$",
		"efficacy_source2dest": r"Efficacy: Source2Dest$\uparrow$",
		"generalization_source_forget": r"Generalization: Source Forget$\uparrow$",
		"generalization_source2dest": r"Generalization: Source2Dest$\uparrow$",
		"generalization_alias2dest": r"Alias2Dest$\uparrow$",
		"specificity_delta": r"Holdout Delta$\uparrow$"
	}
	# titles = [
	# 	r"Efficacy: source Forget$\uparrow$", r"Efficacy: source2dest$\uparrow$", r"Specificity: Holdout Delta$\uparrow$",
	# 	r"Generality: source Forget$\uparrow$", r"Generality: source2dest$\uparrow$", r"Generality: Alias Forget$\uparrow$"]
	
	titles = [
		metric2titles[metric] for metric in metrics
	]
	expected_order = ["bound", "emcid(ours)", "refact", "time", "uce", "ablate", "sa", "fgmn", "esd", "sdd"]

	label2color = {
		"emcid(ours)": "orange",
		"refact": "blue",
		"time": "red",
		"uce": "purple",
		"ablate": "brown",
		"sa": "black",
		"fgmn": "cyan",
		"esd": "green",
		"sdd": "teal",
	}

	# titles = [
	# 	r"Efficacy: source Forget$\uparrow$", r"Efficacy: source2dest$\uparrow$", r"Specificity: Holdout Delta$\uparrow$",
	# 	r"Generalization: source Forget$\uparrow$", r"Generalization: source2dest$\uparrow$", r"Generalization: Alias2dest$\uparrow$"]
	
	
	if no_alias:
		titles = titles[:-1]
		metrics = metrics[:-1]
	
	colors = ["orange", "blue", "green", "red", "purple", "brown", "black", "pink", "gray",
			  "yellow", "cyan", "magenta", "teal", "lime", "lavender", "tan", "salmon", "gold", "lightcoral"]
	if len(emcid_stats) > len(colors):
		raise ValueError("Too many stats to plot")
	
	# first plot the bound of each key
	for i, (metric, title) in enumerate(zip(metrics, titles)):
		ax = plt.subplot(slices[i])
		# plot the bound of this key
		if metric in metrci2bounds and plot_bound:
			# find the longest bound
			def _find_longest_bound_stat_weight(stats):
				longest_bound_stat_weight = None
				longest_bound_len = 0
				for stat in stats:
					for weight, records in stat.items():
						if len(records[metrci2bounds[metric]]) > longest_bound_len:
							longest_bound_stat_weight = (stat, weight)
							longest_bound_len = len(records[metrci2bounds[metric]])
				return longest_bound_stat_weight

			longest_bound_stat, longest_bound_weight = _find_longest_bound_stat_weight(emcid_stats)
			xs, ys = _get_x_y(longest_bound_stat[longest_bound_weight][metrci2bounds[metric]], xs=xs)
			if "dest" in metric:
				ys = [1 - y for y in ys]
			ax.plot(xs,
		   			ys,
					label="bound", 
					color="black", 
					linestyle="--")
		ax.set_title(title, fontsize=title_font_size)
		ax.set_xlabel("Edit Number", fontsize=x_label_font)
		if i == 0 or i == 3:
			ax.set_ylabel("Cls Score", fontsize=y_label_font)
		# remove 20, 30, 40
		if max(xs) == 300 and min(xs) == 10:
			xs_ticks = [x for x in xs if x not in [20, 30, 40]]
		else:
			xs_ticks = xs

			# use log scale for x axis
			# import matplotlib.ticker as ticker
			# plt.rcParams['xtick.minor.size'] = 0
			# plt.rcParams['xtick.minor.width'] = 0

			# ax.set_xscale('log')
			# ax.set_xticks(xs_ticks)
			# ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

		ax.set_xticks(xs_ticks)

		if metric != "specificity_delta":
			# set y range from 0 to 0.8
			ax.set_ylim([-0.1, 1.05])
		else:
			# set y range from -0.65 to 0.05
			ax.set_ylim([-0.8, 0.05])


	itr = iter(colors)
	for i1, stat in enumerate(emcid_stats):
		for i2, (weight, records) in enumerate(stat.items()):
			if exclude is not None:
				exclude_flag = False
				for ex in exclude:
					if ex["hparam"] == records["hparam_name"] and f'weight{ex["mom2_weight"]}' in weight:
						if ex["edit_weight"] == 0.5:
							if "ew" not in weight:
								exclude_flag = True
								break
						elif f'ew{ex["edit_weight"]}' in weight:
							exclude_flag = True
							break
				if exclude_flag:
					continue
			try:
				color = next(itr)
			except StopIteration:
				print(f"len(colors) = {len(colors)}, current i1 = {i1}, i2 = {i2}, len(stats) = {len(emcid_stats)}")
			for i3, (metric, title) in enumerate(zip(metrics, titles)):
				ax = plt.subplot(slices[i3])
				label = "emcid(ours)"
				ax.plot(*_get_x_y(records[metric], xs=xs), 
						label=(records["hparam_name"] + weight.replace("weight", "_w")).replace(
							"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01_w4000_ew0.6",
							"emcid(ours)"
						), 
						marker="o",
						markersize=marker_size,
						color=label2color[label])
						
	
	for i1, stat in enumerate(baseline_stats):
		try:
			color = next(itr)
		except StopIteration:
			print(f"number of colors not enough, need {len(baseline_stats) - i1} more, but only {len(colors)}")
			raise ValueError("number of colors not enough")

		for i2, (metric, title) in enumerate(zip(metrics, titles)):
			ax = plt.subplot(slices[i2])
			if metric not in stat.keys():
				ax.plot([], [], label=stat["name"], marker="x",markersize=marker_size, color=label2color[label])
				continue

			label = stat["name"]
			label = label.replace("uce-no_prompts", "uce")

			if label in ["esd", "sdd", "fgmn"]:
				marker = "x"
				if metric in ["efficacy_source2dest", "generalization_source2dest", "generalization_alias2dest"]:
					ax.plot([], [], label=label, marker=marker, markersize=marker_size, color=label2color[label])
					continue
			else:
				marker = "o"
				
			ax.plot(*_get_x_y(stat[metric], xs=xs), 
					label=label,
					marker=marker,
					markersize=marker_size,
					color=label2color[label]
					)
			
	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	# Shrink current axis's height by 10% on the bottom
	# box = fig.get_position()
	# fig.set_position([box.x0, box.y0 + box.height * 0.1,
	#              box.width, box.height * 0.9])
	# sort the legend by the name	
	expected_order = ["bound", "emcid(ours)", "refact", "time", "uce", "ablate", "sa", "fgmn", "esd", "sdd"]

	items = sorted(by_label.items(), key=lambda x: expected_order.index(x[0]))
	keys = [item[0] for item in items]
	values = [item[1] for item in items]


	plt.legend(
		values,
		keys,
		bbox_to_dest=(-0.7, -0.15), 
		fontsize=legend_font,
		loc="upper center", 
		ncol=5, 
		frameon=False)
	
	
	# save
	if not os.path.exists(os.path.dirname(save_path)):
		os.makedirs(os.path.dirname(save_path))
	
	plt.savefig(save_path, bbox_inches='tight')

	# also save as pdf
	if not os.path.exists(os.path.dirname(save_path.replace(".png", ".pdf"))):
		os.makedirs(os.path.dirname(save_path.replace(".png", ".pdf")))
	plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')


def plot_edit_results_full_one_row(
		emcid_results_paths=["results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary.json"], 
		baseline_results_paths=None,
		save_path="results/emcid/summary.png",
		exclude=None,
		xs=[1, 5, 10, 20, 30, 40, 50],
		plot_bound=False,
		no_alias=False):
	
	emcid_contents = []
	emcid_stats = []
	hparam_names = [path.split("/")[-2] for path in emcid_results_paths]

	baseline_contents = []
	baseline_stats = []
	baseline_names = [path.split("/")[-2] for path in baseline_results_paths]\
		if baseline_results_paths is not None else None

	for results_path in emcid_results_paths:
		with open(results_path, 'r') as f:
			try:
				emcid_contents.append(json.load(f))
			except:
				print(f"failed to load {results_path}")
				raise ValueError("failed to load")
	for content, hparam_name in zip(emcid_contents, hparam_names):
		emcid_stats.append(extract_stat_emcid(content, hparam_name))
	
	if baseline_results_paths is not None:
		for results_path in baseline_results_paths:
			with open(results_path, 'r') as f:
				baseline_contents.append(json.load(f))
		for content, name in zip(baseline_contents, baseline_names):
			baseline_stats.append(extract_stat_baseline(content, name))

	if min(xs) == 1:
		fig = plt.figure(figsize=(18, 2))
		plt.subplots_adjust(wspace=0.2)

		legend_font = 12
		x_label_font = 8
		y_label_font = 8
		title_font_size = 10
		marker_size = 3
		x_tick_font_size = 6
	else:
		fig = plt.figure(figsize=(17, 2))
	# plt.rcParams.update({'font.size': 22})
		plt.subplots_adjust(wspace=0.26)

		legend_font = 12
		x_label_font = 8
		y_label_font = 8
		title_font_size = 10
		marker_size = 3
		x_tick_font_size = 6

	gs = GridSpec(1, 6)

	def _get_x_y(xys, max_x=None, min_x=None, xs=None):
		new_xys = xys[:]
		if max_x is not None:
			new_xys = [item for item in new_xys if item[1] <= max_x]
		if min_x is not None:
			new_xys = [item for item in new_xys if item[1] >= min_x]

		if xs is not None:
		# only keep the xys whose x is in xs
			new_xys = [item for item in new_xys if item[1] in xs]
		x = [item[1] for item in new_xys]
		y = [item[0] for item in new_xys]
		return x, y

	slices = [
		gs[0, 0:1], gs[0, 1:2], gs[0, 2:3], gs[0, 3:4], gs[0, 4:5], gs[0, 5:6]]
	
	metrics = [
		"specificity_delta",
		"efficacy_source_forget", "efficacy_source2dest",
		"generalization_source_forget", "generalization_source2dest", "generalization_alias2dest",
		]
	
	# metrci2bounds = {
	# 	"efficacy_source_forget": "pre_source_score",
	# 	"efficacy_source2dest": "pre_dest_score",
	# 	"generalization_source_forget": "pre_source_score_general",
	# 	"generalization_source2dest": "pre_dest_score_general",
	# 	"specificity_delta": "ng_pre_specificity",
	# 	"generalization_alias_forget": "pre_alias_score"
	# }

	metrci2bounds = {
		"efficacy_source_forget": "pre_source_score",
		"efficacy_source2dest": "pre_source_dest_cls_score_edit",
		"generalization_source_forget": "pre_source_score_general",
		"generalization_source2dest": "pre_source_dest_cls_score_general",
		"specificity_delta": "ng_pre_specificity",
		"generalization_alias2dest": "pre_source_dest_cls_score_alias"
	}
	
	# titles = [
	# 	r"Efficacy: source Forget$\uparrow$", r"Efficacy: source2dest$\uparrow$", r"Specificity: Holdout Delta$\uparrow$",
	# 	r"Generality: source Forget$\uparrow$", r"Generality: source2dest$\uparrow$", r"Generality: Alias Forget$\uparrow$"]
	
	titles = [
		r"Specificity: Holdout Delta$\uparrow$",
		r"Efficacy: source Forget$\uparrow$", r"Efficacy: source2dest$\uparrow$",
		r"Generalization: source Forget$\uparrow$", r"Generalization: source2dest$\uparrow$", r"Generalization: Alias2dest$\uparrow$",
		]
	
	
	if no_alias:
		titles = titles[:-1]
		metrics = metrics[:-1]
	
	colors = ["orange", "blue", "green", "red", "purple", "brown", "black", "pink", "gray",
			  "yellow", "cyan", "magenta", "teal", "lime", "lavender", "tan", "salmon", "gold", "lightcoral"]
	
	expected_order = ["bound", "emcid(ours)", "refact", "time", "uce", "ablate", "sa", "fgmn", "esd", "sdd"]

	label2color = {
		"emcid(ours)": "orange",
		"refact": "blue",
		"time": "red",
		"uce": "purple",
		"ablate": "brown",
		"sa": "black",
		"fgmn": "cyan",
		"esd": "green",
		"sdd": "teal",
	}

	if len(emcid_stats) > len(colors):
		raise ValueError("Too many stats to plot")
	
	# first plot the bound of each key
	for i, (metric, title) in enumerate(zip(metrics, titles)):
		ax = plt.subplot(slices[i])
		# plot the bound of this key

		if metric in metrci2bounds and plot_bound:
			# find the longest bound
			def _find_longest_bound_stat_weight(stats):
				longest_bound_stat_weight = None
				longest_bound_len = 0
				for stat in stats:
					for weight, records in stat.items():
						if len(records[metrci2bounds[metric]]) > longest_bound_len:
							longest_bound_stat_weight = (stat, weight)
							longest_bound_len = len(records[metrci2bounds[metric]])
				return longest_bound_stat_weight

			longest_bound_stat, longest_bound_weight = _find_longest_bound_stat_weight(emcid_stats)
			xs, ys = _get_x_y(longest_bound_stat[longest_bound_weight][metrci2bounds[metric]], xs=xs)
			if "dest" in metric:
				ys = [1 - y for y in ys]
			ax.plot(xs,
		   			ys,
					label="bound", 
					color="black", 
					linestyle="--")
		ax.set_title(title, fontsize=title_font_size)
		ax.set_xlabel("Edit Number", fontsize=x_label_font)
		# if i == 0:
			# ax.set_ylabel("Cls Score", fontsize=y_label_font)
		# remove 20, 30, 40
		if max(xs) == 300 and min(xs) == 10:
			xs_ticks = [x for x in xs if x not in [20, 30, 40]]
		else:
			xs_ticks = xs
		
		# if min(xs) < 10:
		# 	# use log scale for x axis
		# 	import matplotlib.ticker as ticker

		# 	ax.set_xscale('log', base=10)
		# 	ax.set_xticks(xs_ticks)
		# 	ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
		# 	ax.set_xticklabels(ax.get_xticks(), rotation=30)
		# 	ax.set_xlabel("Edit Number", fontsize=x_label_font)

		ax.set_xticks(xs_ticks)

		if metric == "specificity_delta": 
			# set y range from -0.65 to 0.05
			ax.set_ylim([-0.8, 0.05])
		elif "source_forget" in metric:
			if max(xs) == 100:
				ax.set_ylim([-0.1, 0.85])
			else:
				ax.set_ylim([0.2, 0.85])
		else:	
			# set y range from 0 to 0.8
			ax.set_ylim([-0.1, 1.05])
				

		# set the size of the ticks
		plt.xticks(fontsize=x_tick_font_size)
		plt.yticks(fontsize=8)



	itr = iter(colors)
	for i1, stat in enumerate(emcid_stats):
		for i2, (weight, records) in enumerate(stat.items()):
			if exclude is not None:
				exclude_flag = False
				for ex in exclude:
					if ex["hparam"] == records["hparam_name"] and f'weight{ex["mom2_weight"]}' in weight:
						if ex["edit_weight"] == 0.5:
							if "ew" not in weight:
								exclude_flag = True
								break
						elif f'ew{ex["edit_weight"]}' in weight:
							exclude_flag = True
							break
				if exclude_flag:
					continue
			# try:
			# 	color = next(itr)
			# except StopIteration:
			# 	print(f"len(colors) = {len(colors)}, current i1 = {i1}, i2 = {i2}, len(stats) = {len(emcid_stats)}")
			for i3, (metric, title) in enumerate(zip(metrics, titles)):
				ax = plt.subplot(slices[i3])
				label = (records["hparam_name"] + weight.replace("weight", "_w")).replace(
							"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01_w4000_ew0.6",
							"emcid(ours)"
						)
				ax.plot(*_get_x_y(records[metric], xs=xs), 
						label=label,
						marker="o",
						markersize=marker_size,
						color=label2color[label])
	
	for i1, stat in enumerate(baseline_stats):
		# try:
		# 	color = next(itr)
		# except StopIteration:
		# 	print(f"number of colors not enough, need {len(baseline_stats) - i1} more, but only {len(colors)}")
		# 	raise ValueError("number of colors not enough")

		for i2, (metric, title) in enumerate(zip(metrics, titles)):
			ax = plt.subplot(slices[i2])
			if metric not in stat.keys():
				ax.plot([], [], label=stat["name"], marker="x",markersize=marker_size, color=color)
				continue

			label = stat["name"]
			label = label.replace("uce-no_prompts", "uce")

			if label in ["esd", "sdd", "fgmn"]:
				marker = "x"
				if metric in ["efficacy_source2dest", "generalization_source2dest", "generalization_alias2dest"]:
					ax.plot([], [], 
			 				label=label, 
							marker=marker, 
							markersize=marker_size, 
							color=label2color[label])
					continue
			else:
				marker = "o"
				
			ax.plot(*_get_x_y(stat[metric], xs=xs), 
					label=label,
					marker=marker,
					markersize=marker_size,
					color=label2color[label],
					)
			
	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	# sort the legend by the name	
	expected_order = ["bound", "emcid(ours)", "refact", "time", "uce", "ablate", "sa", "fgmn", "esd", "sdd"]

	items = sorted(by_label.items(), key=lambda x: expected_order.index(x[0]))
	keys = [item[0] for item in items]
	values = [item[1] for item in items]

	plt.legend(
		values, 
		keys, 
		bbox_to_dest=(-3.0, -0.15), 
		fontsize=legend_font,
		loc="upper center", 
		ncol=9, 
		frameon=False)
	
	
	# save
	if not os.path.exists(os.path.dirname(save_path)):
		os.makedirs(os.path.dirname(save_path))
	
	plt.savefig(save_path, bbox_inches='tight')

	# also save as pdf
	if not os.path.exists(os.path.dirname(save_path.replace(".png", ".pdf"))):
		os.makedirs(os.path.dirname(save_path.replace(".png", ".pdf")))
	plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')


def plot_trade_off_curves_var_weight(
		result_paths: List[str],
		num_edits=[40,50],
		dataset_name="imgnet_small",
		xaxis="generalization_source_forget",
		yaxis="specificity_delta"):

	"""
	Plot trade off curve for the results in the result folders.
	Note that, in our plot, a curve's point has same edit number, hparam but different weight. 
	x-axis is generalization_source_forget, y-axis is specificity_delta
	"""
	for result_path in result_paths:
		assert dataset_name in result_path, f"dataset name {dataset_name} not in result path {result_path}"

	contents = []
	stats = []
	hparam_names = [path.split("/")[-2] for path in result_paths]

	for results_path in result_paths:
		with open(results_path, 'r') as f:
			contents.append(json.load(f))
	for content, hparam_name in zip(contents, hparam_names):
		stats.append(extract_stat_emcid(content, hparam_name))
	
	fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
	marker_styles = ["o", "v", "s", "p", "P", "*", "X", "D", "d", "h", "H", "+", "x", "|", "_"]
	colors = ["orange", "blue", "green", "red", "purple", "brown", "black", "pink", "gray",
			  "yellow", "cyan", "magenta", "teal", "lime", "lavender", "tan", "salmon", "gold", "lightcoral"]
	num_edits = num_edits[::-1]
	# extract xs and ys for the same hparam
	for i1, (stat, hparam_name) in enumerate(zip(stats, hparam_names)):
		for idx, num_edit in enumerate(num_edits):
			# extract xs and ys for the same weight
			xs, ys = [], []
			weights = []
			for weight, records in stat.items():
				try:
					xs.append(*[item[0] for item in records[xaxis] if item[1] == num_edit])
					ys.append([item[0] for item in records[yaxis] if item[1] == num_edit])
					weights.append(weight)
				except TypeError:
					# no such edit number
					continue
				# plot
			# axhplot(xs, ys, marker="o", label=hparam_name, alpha=int(weight.replace("weight", ""))/15000)
			sizes = [int(weight.replace("weight", ""))/15000 * 200 for weight in weights]
			ax.scatter(xs, ys, marker=marker_styles[idx], label=hparam_name, s=sizes, color=colors[i1])
	
	# set x ticks
	ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
	ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.7"] )

	# set y ticks
	ax.set_yticks([-0.6, -0.4, -0.2, 0.0])
	ax.set_yticklabels(["-0.6", "-0.4", "-0.2", "0.0"])

	# set y range from -0.6 to 0.1
	ax.set_ylim([-0.6, 0.1])
	ax.set_xlim([0.0, 0.7])
	

	ax.set_xlabel("Generalization source Forget")
	ax.set_ylabel("Specificity Delta")

	ax.set_title(f"Trade-off w/ different weights, edit number = {num_edit}")

	# use extra legend to show the relation between marker size and edit num
	for idx, num_edit in enumerate(num_edits):
		ax.scatter([], [], marker=marker_styles[idx], label=f"edit num = {num_edit}", s=200, color="black")

	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	# Shrink current axis's height by 10% on the bottom
	# box = fig.get_position()
	# fig.set_position([box.x0, box.y0 + box.height * 0.1,
	#              box.width, box.height * 0.9])
	plt.legend(
		by_label.values(), 
		by_label.keys(), 
		loc="lower left", 
		frameon=True)

	# save the trade off curve
	plt.savefig(f"results/emcid/{dataset_name}_trade_off_curve_var_weight_{xaxis}.png")
	
	



def plot_trade_off_curve_hparam(
		result_folders=[
			"results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary.json",
			"results/emcid/dest_s-200-c-0.75/summary.json",
			"results/emcid/dest_s-200_l-1.5e4/summary.json",
			"results/emcid/dest_s-200_la-1e4/summary.json",
		],
		label="",
		):
	

	fig, ax = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
	
	contents = []
	stats = []
	hparam_names = [path.split("/")[-2] for path in result_folders]

	for results_path in result_folders:
		with open(results_path, 'r') as f:
			contents.append(json.load(f))
	for content, hparam_name in zip(contents, hparam_names):
		stats.append(extract_stat_baseline(content, hparam_name))
	
	# extract xs and ys for the same hparam
	xys = []
	def _get_xys(stat, xkey, ykey):
		x = [item[0] for item in stat[xkey]]
		y = [item[0] for item in stat[ykey]]
		return x, y
	
	for stat in stats:
		xys.append(_get_xys(stat, "generalization_source_forget", "specificity_delta"))
	
	colors = ["orange", "blue", "green", "red", "purple", "brown", "black", "pink", "gray",
			  "yellow", "cyan", "magenta", "teal", "lime", "lavender", "tan", "salmon", "gold", "lightcoral"]
	# plot
	for idx, xy in enumerate(xys):
		ax.plot(*xy, marker="o", color=colors[idx], label=label + hparam_names[idx])
	
	set_trade_off_curve_axis(ax)
	# save the trade off curve
	plt.savefig("results/emcid/trade_off_curve_hparam.png")

def set_trade_off_curve_axis(ax):
	# set x ticks
	ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.7])
	ax.set_xticklabels(["0.0", "0.2", "0.4", "0.6", "0.7"] )

	# set y ticks
	ax.set_yticks([-0.6, -0.4, -0.2, 0.0])
	ax.set_yticklabels(["-0.6", "-0.4", "-0.2", "0.0"])

	# set y range from -0.6 to 0.1
	ax.set_ylim([-0.6, 0.1])
	ax.set_xlim([0.0, 0.7])
	ax.legend(loc="upper right", frameon=False)



def plot_trade_off_curve_edit_num(
		ax, 
		result_folders =[
			"results/emcid/dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04/summary.json",
			"results/emcid/dest_s-200-c-0.75/summary.json",
			"results/emcid/dest_s-200_l-1.5e4/summary.json",
			"results/emcid/dest_s-200_la-1e4/summary.json",
		],
		label="",
		):
	"""
	Plot trade off curve for the results in the result folders.
	"""

	contents = []
	stats = []
	hparam_names = [path.split("/")[-2] for path in result_folders]

	for results_path in result_folders:
		with open(results_path, 'r') as f:
			contents.append(json.load(f))
	for content, hparam_name in zip(contents, hparam_names):
		stats.append(extract_stat_baseline(content, hparam_name))
	
	num_edits = [5, 10, 20, 30, 40, 50]
	# extract xs and ys for the same edit number
	xys = []

	def _get_v(stat, key, edit_num):
		for item in stat[key]:
			if item[1] == edit_num:
				return item[0]
		raise ValueError(f"Edit number {edit_num} not found in {key}")

	for num_edit in num_edits:
		xys.append([])
		for stat in stats:
			x = _get_v(stat, "generalization_source_forget", num_edit)
			y = _get_v(stat, "specificity_delta", num_edit)
			xys[-1].append((x, y))
	colors = ["orange", "blue", "green", "red", "purple", "brown"] 
	# plot
	for idx, xy in enumerate(xys):
		xs = [item[0] for item in xy]
		ys = [item[1] for item in xy]
		ax.plot(xs, ys, marker="o", color=colors[idx], label=label + f"ed_{num_edits[idx]}")

	set_trade_off_curve_axis(ax)


def traverse_results(
		emcid_result_folder="results/emcid", 
		baseline_result_folder="results/baselines",
		dataset: Literal["imgnet_samll", "imgnet_aug", "artists"]="imgnet_small",
		trade_off_edit=50,
		min_x=1,
		max_x=300,
		plot_full=True,
		plot_trade_off_curv=False):
	"""
	Traverse the result folder and plot trade off curves for each subfolder.
	"""
	exclude = [
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 10000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 10000, "edit_weight": 0.1},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 8000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 7000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 6000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 5000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", "mom2_weight": 4000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", "mom2_weight": 8000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", "mom2_weight": 5000, "edit_weight": 0.5},
		{"hparam": "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", "mom2_weight": 7000, "edit_weight": 0.5},
	]
	# add exclude for different edit_weight
	for edit_weight in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
		edit_weight = float(edit_weight)
		if edit_weight == 0.6:
			continue
		else:
			exclude.append({
				"hparam": "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", 
				"mom2_weight": 4000, 
				"edit_weight": edit_weight})

	if min_x >= 10 :
		include = [
			"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", # current best
			# "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", # previous objective
			# "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01_spp-4",	# more training imgs
			# "source_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", # previous objective
			# "dest_s-200_c-1.5_ly-11_lr-0.2_ewc-1e7_txt-align-0.01", # with ewc
			# baseline results
			# "fgmn",
			"time",
			"refact",
			# "ablate",
			"uce",
			"esd",
			# "sa"
		]
	else:
		include = [
			"dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01", # current best
			# "dest_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", # previous objective
			# "dest_s-200_c-1.5_ly-11_lr-0.2_wd-5e-04_txt-align-0.01_spp-4",	# more training imgs
			# "source_s-200_c-1.5_ly-12_lr-0.2_wd-5e-04", # previous objective
			# "dest_s-200_c-1.5_ly-11_lr-0.2_ewc-1e7_txt-align-0.01", # with ewc
			# baseline results
			"fgmn",
			"refact",
			"ablate",
			"uce",
			"time",
			"sa",
			"sdd",
			"esd",
		]
	exclude_sub = [
	]
	# exclude_sub = []

	if not os.path.exists(emcid_result_folder):
		os.makedirs(emcid_result_folder)
	if not os.path.exists(baseline_result_folder):
		os.makedirs(baseline_result_folder)

	# get subfolders
	emcid_subfolders = [os.path.join(emcid_result_folder, subfolder) \
						for subfolder in os.listdir(emcid_result_folder)\
						if subfolder in include]

	baseline_subfolders = [os.path.join(baseline_result_folder, subfolder) \
						   for subfolder in os.listdir(baseline_result_folder)\
						   if subfolder in include]

	
	for ex_sub in exclude_sub:
		emcid_subfolders = [subfolder for subfolder in emcid_subfolders if ex_sub not in subfolder]
		baseline_subfolders = [subfolder for subfolder in baseline_subfolders if ex_sub not in subfolder]

	# get paths of each subfolder's summary.json
	emcid_summary_paths = [os.path.join(subfolder, f"{dataset}_summary.json") for subfolder in emcid_subfolders]
	baseline_summary_paths = [os.path.join(subfolder, f"{dataset}_summary.json") for subfolder in baseline_subfolders]
	# remove paths that are not files
	emcid_summary_paths = [path for path in emcid_summary_paths if os.path.isfile(path)]
	baseline_summary_paths = [path for path in baseline_summary_paths if os.path.isfile(path)]

	xs = [5, 10, 20, 30, 40, 50] if dataset == "imgnet_small" else \
		[1, 5, 10, 20, 30, 40, 50] + [i for i in range(100, 301, 50)]
	
	xs = [x for x in xs if x <= max_x and x >= min_x]
	if max_x == 300:
		xs = [x for x in xs if x not in [20, 30, 40]]

	if plot_full:
		# plot_edit_results_full_one_row(
		# 	emcid_results_paths=emcid_summary_paths, 
		# 	baseline_results_paths=baseline_summary_paths,
		# 	save_path=os.path.join(emcid_result_folder, f"aiced_summary_max-{max_x}.png"),
		# 	exclude=exclude,
		# 	xs=xs)

		plot_edit_results_full_two_row(
			emcid_results_paths=emcid_summary_paths, 
			baseline_results_paths=baseline_summary_paths,
			save_path=os.path.join(emcid_result_folder, f"aiced_summary_max-{max_x}_2r.png"),
			exclude=exclude,
			xs=xs
		)
		

	 # get subfolders, no exclude
	emcid_subfolders = [os.path.join(emcid_result_folder, subfolder) for subfolder in os.listdir(emcid_result_folder)]
	emcid_summary_paths = [os.path.join(subfolder, f"{dataset}_summary.json") for subfolder in emcid_subfolders\
						   if os.path.isfile(os.path.join(subfolder, f"{dataset}_summary.json"))]

	if plot_trade_off_curv:
		plot_trade_off_curves_var_weight(
			result_paths=emcid_summary_paths,
			num_edits=[trade_off_edit,],
			dataset_name=dataset,
			xaxis="generalization_source_forget")
		
		plot_trade_off_curves_var_weight(
			result_paths=emcid_summary_paths,
			num_edits=[trade_off_edit,],
			dataset_name=dataset,
			xaxis="generalization_source2dest")
		
		plot_trade_off_curves_var_weight(
			result_paths=emcid_summary_paths,
			num_edits=[trade_off_edit,],
			dataset_name=dataset,
			xaxis="efficacy_source_forget")
		
		plot_trade_off_curves_var_weight(
			result_paths=emcid_summary_paths,
			num_edits=[trade_off_edit,],
			dataset_name=dataset,
			xaxis="generalization_alias_forget")


def plot_clip_and_fid_coco(
	emcid_result_folder="results/emcid",
	baseline_result_folder="results/baselines",
	plot_lpips=False,
	max_x=300,
	direction:Literal["horizontal", "vertical"]="vertical"
):
	include = [
		"dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01",
		"uce-no_prompts",
	]
	# get subfolders
	emcid_subfolders = [os.path.join(emcid_result_folder, subfolder) \
						for subfolder in os.listdir(emcid_result_folder)
						if subfolder in include]

	baseline_subfolders = [os.path.join(baseline_result_folder, subfolder) \
						   for subfolder in os.listdir(baseline_result_folder)
						   if subfolder in include]
	
	# find subfolders that have artists as their subfolder
	artists_subfolders = []
	for emcid_subfolder in emcid_subfolders:
		if not os.path.isdir(emcid_subfolder):
			continue
		for sub in os.listdir(emcid_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(emcid_subfolder, sub))
	
	for baseline_subfolder in baseline_subfolders:
		if not os.path.isdir(baseline_subfolder):
			continue
		for sub in os.listdir(baseline_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(baseline_subfolder, sub))
	
	# get paths of each subfolder's summary.json
	coco_summary_paths = [os.path.join(subfolder, f"coco_summary.json") for subfolder in artists_subfolders]
	baseline_summary_paths = [os.path.join(subfolder, f"coco_summary.json") for subfolder in baseline_subfolders]

	if plot_lpips and direction == "horizontal":
		fig = plt.figure(figsize=(8, 1.2))
		gs = GridSpec(1, 3)
		axes = []
		for i in range(3):
			axes.append(plt.subplot(gs[i:i+1]))
		plt.subplots_adjust(wspace=0.3)
	elif plot_lpips and direction == "vertical":
		fig = plt.figure(figsize=(2, 6))
		gs = GridSpec(3, 1)
		axes = []
		for i in range(3):
			axes.append(plt.subplot(gs[i:i+1]))
		plt.subplots_adjust(hspace=0.8)
	else:
		fig = plt.figure(figsize=(6, 2))
		gs = GridSpec(1, 2)
		axes = []
		for i in range(2):
			axes.append(plt.subplot(gs[i:i+1]))
	
	legend_font = 10
	x_label_font = 10
	title_font_size = 12
	marker_size = 3

	def _extract_regions(records, max_x):
		xs = []
		lpips_lst = []
		clip_lst = []
		fid_lst = []
		for key, value in records.items():
			edit_num = int(key.split("_")[1])
			if edit_num > max_x:
				continue
			xs.append(edit_num)
			try:
				lpips_lst.append(value["lpips"]["mean"])
			except:
				print(value.keys())
				raise ValueError("lpips not found")
			clip_lst.append(value["clip_vit_large"]["mean"])
			fid_lst.append(value["fid"])

		# sort by xs
		xs, lpips_lst, clip_lst, fid_lst = zip(*sorted(
												zip(xs, lpips_lst, clip_lst, fid_lst),
												key=lambda x: x[0]))

		ret_dict = {
			"xs": xs,
			"lpips": lpips_lst,
			"clip": clip_lst,
			"fid": fid_lst
		}
		return ret_dict
	
	markers = ["o", "o", "o"]
	emcid_color = "blue"
	uce_color = "red"

	# plot baselines of original stable diffusion
	orig_coco_summary_path = "results/sd_orig/artists/coco_summary.json"
	with open(orig_coco_summary_path, 'r') as f:
		records = json.load(f)
	
	# xs = [x for x in [1, 50, 100, 200, 500, 1000] if x <= max_x]
	# orig_clip = records["sd_orig"]["clip_vit_large"]["mean"]
	# orig_fid = records["sd_orig"]["fid"]

	# orig_color = "black"
	
	# axes[0].plot(
	# 	xs, 
	# 	[orig_clip] * len(xs),
	# 	color=orig_color, 
	# 	linestyle="--",
	# 	label=f"orig_sd")

	# axes[1].plot(
	# 	xs,
	# 	[orig_fid] * len(xs),
	# 	color=orig_color,
	# 	linestyle="--",
	# 	label=f"orig_sd")
	
	# if plot_lpips:
	# 	# add dummy line for legend
	# 	axes[2].plot(
	# 		[],
	# 		[],
	# 		color=orig_color,
	# 		linestyle="--",
	# 		label=f"orig_sd")


	marker_iter = iter(markers)
	for idx, coco_summary_path in enumerate(coco_summary_paths):
		if not os.path.exists(coco_summary_path):
			continue
		marker = next(marker_iter)
		with open(coco_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions(records, max_x)
		label = str(coco_summary_path).split('/')[-3]
		label = label.replace("dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01", "emcid(ours)")
		# plot 
		axes[0].plot(
			regions["xs"], 
			regions["clip"], 
			color=emcid_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

		axes[1].plot(
			regions["xs"], 
			regions["fid"], 
			color=emcid_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

		if plot_lpips:
			axes[2].plot(
				regions["xs"], 
				regions["lpips"], 
				color=emcid_color, 
				marker=marker,
				markersize=marker_size,
				label=label)

		
	for idx, coco_summary_path in enumerate(baseline_summary_paths):
		if not os.path.exists(coco_summary_path):
			continue
		marker = next(marker_iter)
		with open(coco_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions(records, max_x)
		label = str(coco_summary_path).split('/')[-3]
		label = "uce"
		# plot
		axes[0].plot(
			regions["xs"], 
			regions["clip"], 
			color=uce_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

		axes[1].plot(
			regions["xs"], 
			regions["fid"], 
			color=uce_color,
			marker=marker,
			markersize=marker_size,
			label=label)
		
		if plot_lpips:
			axes[2].plot(
				regions["xs"], 
				regions["lpips"], 
				marker=marker,
				markersize=marker_size,
				color=uce_color,
				label=label)
	

	# set x ticks
	x_ticks = [1, 5, 10, 50, 100, 500, 1000]

	# use log scale for x axis
	import matplotlib.ticker as ticker
	

	x_ticks = [x for x in x_ticks if x <= max_x]
	for ax in axes:
		# plt.rcParams['xtick.minor.size'] = 0
		# plt.rcParams['xtick.minor.width'] = 0

		ax.set_xscale('log')
		ax.set_xticks(x_ticks)
		ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
		ax.set_xticklabels(ax.get_xticks(), rotation=30)
		ax.set_xlabel("Edit Number", fontsize=x_label_font)


	axes[0].set_title(r"CLIP Score $\uparrow$", fontsize=title_font_size)
	axes[1].set_title(r"FID Score $\downarrow$", fontsize=title_font_size)

	if plot_lpips:
		axes[2].set_title(r"LPIPS Score $\downarrow$", fontsize=title_font_size)

	# for ax in axes[0:1]:
	# 	ax.legend(frameon=False, fontsize=16)

	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	# Shrink current axis's height by 10% on the bottom
	if plot_lpips and direction == "horizontal":
		bbox_to_dest=(-0.9, -0.5)
	elif plot_lpips and direction == "vertical":
		bbox_to_dest=(0.46, -0.45)
	else:
		bbox_to_dest=(-0.1, -0.3)

	plt.legend(
		by_label.values(), 
		by_label.keys(), 
		bbox_to_dest=bbox_to_dest,
		fontsize=legend_font,
		loc="upper center", 
		ncol=2, 
		frameon=False)
	# save 
	plt.savefig(os.path.join(emcid_result_folder, "artists_coco_eval.png"), bbox_inches='tight')

	# save as pdf
	plt.savefig(os.path.join(emcid_result_folder, "artists_coco_eval.pdf"), bbox_inches='tight')



def plot_lpips_and_clip_artists(
	emcid_result_folder="results/emcid",
	baseline_result_folder="results/baselines",
	plot_edit_line=False,
	plot_std=False,
	plot_clip=True,
	plot_bound=False,
	max_x=300
):
	include = [
		"dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01"
	]
	# get subfolders
	emcid_subfolders = [os.path.join(emcid_result_folder, subfolder) \
						for subfolder in os.listdir(emcid_result_folder)
						if subfolder in include]
	baseline_subfolders = [os.path.join(baseline_result_folder, subfolder) \
						   for subfolder in os.listdir(baseline_result_folder)]
	
	# find subfolders that have artists as their subfolder
	artists_subfolders = []
	for emcid_subfolder in emcid_subfolders:
		if not os.path.isdir(emcid_subfolder):
			continue
		for sub in os.listdir(emcid_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(emcid_subfolder, sub))
	
	for baseline_subfolder in baseline_subfolders:
		if not os.path.isdir(baseline_subfolder):
			continue
		for sub in os.listdir(baseline_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(baseline_subfolder, sub))
	
	# get paths of each subfolder's summary.json
	artists_summary_paths = [os.path.join(subfolder, f"artists_summary.json") for subfolder in artists_subfolders]
	baseline_summary_paths = [os.path.join(subfolder, f"artists_summary.json") for subfolder in baseline_subfolders]

	## plot lpips curves
	# for each summary file, plot 2 lines: to edit line, and holdout line
	if plot_clip:
		fig = plt.figure(figsize=(3, 5), tight_layout=False)
		gs = GridSpec(2, 1)

		plt.subplots_adjust(hspace=0)

		axes = []
		for i in range(2):
			axes.append(plt.subplot(gs[i:i+1]))
	else:
		fig, axes = plt.subplots(1, 1, figsize=(10, 10), tight_layout=True)
		axes = [axes,]
	

	legend_font = 10
	x_label_font = 10
	title_font_size = 12
	marker_size = 3

	def _extract_regions(records, max_x):
		xs = []
		lpips_edit_y_up = []
		lpips_edit_y_mean = []
		lpips_edit_y_low = []
		lpips_hold_out_y_up = []
		lpips_hold_out_y_mean = []
		lpips_hold_out_y_low = []

		clip_edit_y_mean = []
		clip_hold_out_y_mean = []
		for key, value in records.items():
			edit_num = int(key.split("_")[1])
			if edit_num > max_x:
				continue
			xs.append(edit_num)
			lpips_edit_y_mean.append(value["edit_lpips"]["mean"])
			lpips_edit_y_up.append(value["edit_lpips"]["mean"] + value["edit_lpips"]["std"])
			lpips_edit_y_low.append(value["edit_lpips"]["mean"] - value["edit_lpips"]["std"])

			lpips_hold_out_y_mean.append(value["hold_out_lpips"]["mean"])
			lpips_hold_out_y_up.append(value["hold_out_lpips"]["mean"] + value["hold_out_lpips"]["std"])
			lpips_hold_out_y_low.append(value["hold_out_lpips"]["mean"] - value["hold_out_lpips"]["std"])
			clip_edit_y_mean.append(value["edit_clip"]["mean"])
			clip_hold_out_y_mean.append(value["hold_out_clip"]["mean"])

		ret_dict = {
			"xs": xs,
			"lpips_edit_y_up": lpips_edit_y_up,
			"lpips_edit_y_mean": lpips_edit_y_mean,
			"lpips_edit_y_low": lpips_edit_y_low,
			"lpips_hold_out_y_up": lpips_hold_out_y_up,
			"lpips_hold_out_y_mean": lpips_hold_out_y_mean,
			"lpips_hold_out_y_low": lpips_hold_out_y_low,
			"clip_edit_y_mean": clip_edit_y_mean,
			"clip_hold_out_y_mean": clip_hold_out_y_mean
		}
		return ret_dict
	
	# colors =[
	# 	("blue", "orange"),
	# 	("green", "red"),
	# 	("purple", "brown"),
	# 	("black", "pink"),
	# 	("gray", "yellow"),
	# 	("cyan", "magenta"),
	# 	("teal", "lime"),
	# 	("lavender", "tan"),
	# 	("salmon", "gold"),
	# 	("lightcoral", "blue"),
	# ]
	colors =[
		("blue", "orange"),
		("green", "blue"),
		("purple", "red"),
		("black", "pink"),
		("gray", "yellow"),
		("cyan", "magenta"),
		("teal", "lime"),
		("lavender", "tan"),
		("salmon", "gold"),
		("lightcoral", "blue"),
	]

	# plot original clip score
	orig_summary_path = "results/sd_orig/artists/artists_summary.json"
	with open(orig_summary_path, 'r') as f:
		records = json.load(f)
	
	if plot_clip and plot_bound:
		orig_xs = []
		orig_hold_out_clip = []
		orig_edit_clip = []
		for key, value in records.items():
			x = int(key.split("_")[-1])
			if x > max_x:
				continue
			orig_xs.append(x)
			orig_hold_out_clip.append(value["hold_out_clip"]["mean"])
			orig_edit_clip.append(value["edit_clip"]["mean"])

		if plot_edit_line:	
			axes[1].plot(
				orig_xs,
				orig_edit_clip,
				color=colors[0][0],
				linestyle="--",
				label="edit-orig_sd")
		
		axes[1].plot(
			orig_xs,
			orig_hold_out_clip,
			color=colors[0][1],
			linestyle="--",
			label="holdout-orig_sd")


	itr = iter(colors[1:])
	for idx, artists_summary_path in enumerate(artists_summary_paths):
		if not os.path.exists(artists_summary_path):
			continue
		color_pair = next(itr)
		with open(artists_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions(records, max_x)
		# plot
		if plot_edit_line:
			axes[0].plot(
				regions["xs"], 
				regions["lpips_edit_y_mean"], 
				color=color_pair[0], 
				marker="o",
				markersize=marker_size,
				label=f"edit-{str(artists_summary_path).split('/')[-3]}")
			
			try:
				axes[1].plot(
					regions["xs"], 
					regions["clip_edit_y_mean"], 
					color=color_pair[0], 
					marker="o",
					markersize=marker_size,
					label=f"edit-{str(artists_summary_path).split('/')[-3]}")
			except ValueError:
				print(f"ValueError for {str(artists_summary_path).split('/')[-3]}")
		
		if plot_std:
			axes[0].fill_between(regions["xs"], 
							regions["lpips_edit_y_up"], 
							regions["lpips_edit_y_low"], 
							color=color_pair[0], 
							alpha=0.2)
		
		label = f"holdout-{str(artists_summary_path).split('/')[-3]}"
		label = r"holdout-emcid"
		axes[0].plot(
			regions["xs"], 
			regions["lpips_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=color_pair[1], 
			label=label)
		
		if plot_clip:
			try:
				axes[1].plot(
					regions["xs"], 
					regions["clip_hold_out_y_mean"], 
					marker="o",
					markersize=marker_size,
					color=color_pair[1], 
					label=label.replace("down", "up"))
			except ValueError:
				print(f"ValueError for {str(artists_summary_path).split('/')[-3]}")

		if plot_std:
			axes[0].fill_between(regions["xs"],
							regions["lpips_hold_out_y_up"],
							regions["lpips_hold_out_y_low"],
							color=color_pair[1],
							alpha=0.2)

	for idx, artists_summary_path in enumerate(baseline_summary_paths):
		if not os.path.exists(artists_summary_path):
			continue
		color_pair = next(itr)
		with open(artists_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions(records, max_x)
		# plot
		if plot_edit_line:
			axes[0].plot(
				regions["xs"], 
				regions["lpips_edit_y_mean"], 
				marker="o",
				makersize=marker_size,
				color=color_pair[0], 
				label=f"edit-{str(artists_summary_path).split('/')[-3]}")
			
			try:
				axes[1].plot(
					regions["xs"], 
					regions["clip_edit_y_mean"], 
					marker="x",
					markersize=marker_size,
					color=color_pair[0], 
					label=f"edit-{str(artists_summary_path).split('/')[-3]}")
			except ValueError:
				print(f"ValueError for {str(artists_summary_path).split('/')[-3]}")

		if plot_std:
			axes[0].fill_between(regions["xs"], 
							regions["lpips_edit_y_up"], 
							regions["lpips_edit_y_low"], 
							color=color_pair[0], 
							alpha=0.2)
		label = f"holdout-{str(artists_summary_path).split('/')[-3]}"
		label = r"holdout-uce"
		axes[0].plot(
			regions["xs"], 
			regions["lpips_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=color_pair[1], 
			label=label)
		if plot_clip:
			try:
				axes[1].plot(
					regions["xs"], 
					regions["clip_hold_out_y_mean"], 
					marker="o",
					markersize=marker_size,
					color=color_pair[1], 
					label=label.replace("down", "up"))
			except ValueError:
				print(f"ValueError for {str(artists_summary_path).split('/')[-3]}")
		
		if plot_std:
			axes[0].fill_between(regions["xs"],
							regions["lpips_hold_out_y_up"],
							regions["lpips_hold_out_y_low"],
							color=color_pair[1],
							alpha=0.2)
	


	
	# set x ticks
	if max_x <= 100:
		x_ticks = [1, 5, 10, 50, 100]
	else:
		x_ticks = [1, 50, 100, 200, 500, 1000]
	x_ticks = [x for x in x_ticks if x <= max_x]

	# set x ticks
	x_ticks = [1, 5, 10, 50, 100, 500, 1000]

	# use log scale for x axis
	import matplotlib.ticker as ticker
	

	x_ticks = [x for x in x_ticks if x <= max_x]
	for ax in axes:
		# plt.rcParams['xtick.minor.size'] = 0
		# plt.rcParams['xtick.minor.width'] = 0

		ax.set_xscale('log')
		ax.set_xticks(x_ticks)
		ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
		ax.set_xticklabels(ax.get_xticks(), rotation=30)
		ax.set_xlabel("Edit Number", fontsize=x_label_font)

	# for ax in axes:
	# 	ax.set_xticks(x_ticks)
	# 	ax.set_xlabel("Edit Number", fontsize=16)

	axes[0].set_title(r"LPIPS$\downarrow$", fontsize=title_font_size)
	if plot_clip:
		axes[1].set_title(r"CLIP$\uparrow$", fontsize=title_font_size)

	# add arrows outside the plot to show which direction is better
	# axes[1].annotate('better', xy=(0, -0.1), xycoords='axes fraction', xytext=(1, -0.1), 
    #         arrowprops=dict(arrowstyle="<->", color='b'))
	
	
	
	# axes[1].annotate(
	# 	"better", xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0.5, 0.1), 
	# 	arrowprops=dict(arrowstyle="<-", color="black"))
	# for ax in axes[1:]:
	# 	ax.legend(fontsize=16, frameon=False)

	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	bbox_to_dest=(0.5, -0.4)

	plt.legend(
		by_label.values(), 
		by_label.keys(), 
		bbox_to_dest=bbox_to_dest,
		fontsize=legend_font,
		loc="upper center", 
		ncol=2, 
		frameon=False)

	# save 
	plt.savefig(os.path.join(emcid_result_folder, f"artists_max_x-{max_x}.png"))

	# save as pdf
	plt.savefig(os.path.join(emcid_result_folder, f"artists_max_x-{max_x}.pdf"))


def plot_coco_and_artists(
	emcid_result_folder="results/emcid",
	baseline_result_folder="results/baselines",
	max_x=300,
):
	include = [
		"dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01",
		"uce-no_prompts",
	]
	# get subfolders
	emcid_subfolders = [os.path.join(emcid_result_folder, subfolder) \
						for subfolder in os.listdir(emcid_result_folder)
						if subfolder in include]

	baseline_subfolders = [os.path.join(baseline_result_folder, subfolder) \
						   for subfolder in os.listdir(baseline_result_folder)
						   if subfolder in include]
	
	# find subfolders that have artists as their subfolder
	artists_subfolders = []
	for emcid_subfolder in emcid_subfolders:
		if not os.path.isdir(emcid_subfolder):
			continue
		for sub in os.listdir(emcid_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(emcid_subfolder, sub))
	
	for baseline_subfolder in baseline_subfolders:
		if not os.path.isdir(baseline_subfolder):
			continue
		for sub in os.listdir(baseline_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(baseline_subfolder, sub))
	
	# get paths of each subfolder's summary.json
	coco_summary_paths = [os.path.join(subfolder, f"coco_summary.json") for subfolder in artists_subfolders]
	baseline_summary_paths = [os.path.join(subfolder, f"coco_summary.json") for subfolder in baseline_subfolders]

	fig = plt.figure(figsize=(10, 6))
	gs = GridSpec(2, 2)

	plt.subplots_adjust(hspace=0.50, wspace=0.15)
	axes = []
	for i in range(4):
		axes.append(plt.subplot(gs[i:i+1]))
	
	legend_font = 18
	x_label_font = 15
	title_font_size = 16
	marker_size = 5

	def _extract_regions_artists(records, max_x):
		xs = []
		lpips_edit_y_up = []
		lpips_edit_y_mean = []
		lpips_edit_y_low = []
		lpips_hold_out_y_up = []
		lpips_hold_out_y_mean = []
		lpips_hold_out_y_low = []

		clip_edit_y_mean = []
		clip_hold_out_y_mean = []
		for key, value in records.items():
			edit_num = int(key.split("_")[1])
			if edit_num > max_x:
				continue
			xs.append(edit_num)
			lpips_edit_y_mean.append(value["edit_lpips"]["mean"])
			lpips_edit_y_up.append(value["edit_lpips"]["mean"] + value["edit_lpips"]["std"])
			lpips_edit_y_low.append(value["edit_lpips"]["mean"] - value["edit_lpips"]["std"])

			lpips_hold_out_y_mean.append(value["hold_out_lpips"]["mean"])
			lpips_hold_out_y_up.append(value["hold_out_lpips"]["mean"] + value["hold_out_lpips"]["std"])
			lpips_hold_out_y_low.append(value["hold_out_lpips"]["mean"] - value["hold_out_lpips"]["std"])
			clip_edit_y_mean.append(value["edit_clip"]["mean"])
			clip_hold_out_y_mean.append(value["hold_out_clip"]["mean"])

		ret_dict = {
			"xs": xs,
			"lpips_edit_y_up": lpips_edit_y_up,
			"lpips_edit_y_mean": lpips_edit_y_mean,
			"lpips_edit_y_low": lpips_edit_y_low,
			"lpips_hold_out_y_up": lpips_hold_out_y_up,
			"lpips_hold_out_y_mean": lpips_hold_out_y_mean,
			"lpips_hold_out_y_low": lpips_hold_out_y_low,
			"clip_edit_y_mean": clip_edit_y_mean,
			"clip_hold_out_y_mean": clip_hold_out_y_mean
		}
		return ret_dict

	def _extract_regions_coco(records, max_x):
		xs = []
		lpips_lst = []
		clip_lst = []
		fid_lst = []
		for key, value in records.items():
			edit_num = int(key.split("_")[1])
			if edit_num > max_x:
				continue
			xs.append(edit_num)
			try:
				lpips_lst.append(value["lpips"]["mean"])
			except:
				print(value.keys())
				raise ValueError("lpips not found")
			clip_lst.append(value["clip_vit_large"]["mean"])
			fid_lst.append(value["fid"])

		# sort by xs
		xs, lpips_lst, clip_lst, fid_lst = zip(*sorted(
												zip(xs, lpips_lst, clip_lst, fid_lst),
												key=lambda x: x[0]))

		ret_dict = {
			"xs": xs,
			"lpips": lpips_lst,
			"clip": clip_lst,
			"fid": fid_lst
		}
		return ret_dict
	
	markers = ["o", "o", "o"]
	emcid_color = "blue"
	uce_color = "red"

	# plot baselines of original stable diffusion
	orig_coco_summary_path = "results/sd_orig/artists/coco_summary.json"
	with open(orig_coco_summary_path, 'r') as f:
		records = json.load(f)
	
	marker_iter = iter(markers)
	for idx, coco_summary_path in enumerate(coco_summary_paths):
		if not os.path.exists(coco_summary_path):
			continue
		marker = next(marker_iter)
		with open(coco_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions_coco(records, max_x)
		label = str(coco_summary_path).split('/')[-3]
		label = label.replace("dest_s-200_c-1.5_ly-7-11_lr-0.2_wd-5e-04_txt-align-0.01", "emcid(ours)")
		# plot coco on the left
		axes[0].plot(
			regions["xs"], 
			regions["clip"], 
			color=emcid_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

		axes[1].plot(
			regions["xs"], 
			regions["fid"], 
			color=emcid_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

	for idx, coco_summary_path in enumerate(baseline_summary_paths):
		if not os.path.exists(coco_summary_path):
			continue
		marker = next(marker_iter)
		with open(coco_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions_coco(records, max_x)
		label = str(coco_summary_path).split('/')[-3]
		label = "uce"
		# plot
		axes[0].plot(
			regions["xs"], 
			regions["clip"], 
			color=uce_color, 
			marker=marker,
			markersize=marker_size,
			label=label)

		axes[1].plot(
			regions["xs"], 
			regions["fid"], 
			color=uce_color,
			marker=marker,
			markersize=marker_size,
			label=label)
		

	# plot artists on the right
	# find subfolders that have artists as their subfolder
	artists_subfolders = []
	for emcid_subfolder in emcid_subfolders:
		if not os.path.isdir(emcid_subfolder):
			continue
		for sub in os.listdir(emcid_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(emcid_subfolder, sub))
	
	for baseline_subfolder in baseline_subfolders:
		if not os.path.isdir(baseline_subfolder):
			continue
		for sub in os.listdir(baseline_subfolder):
			if "artists" in sub:
				artists_subfolders.append(os.path.join(baseline_subfolder, sub))
	
	# get paths of each subfolder's summary.json
	artists_summary_paths = [os.path.join(subfolder, f"artists_summary.json") for subfolder in artists_subfolders]
	baseline_summary_paths = [os.path.join(subfolder, f"artists_summary.json") for subfolder in baseline_subfolders]

	for idx, artists_summary_path in enumerate(artists_summary_paths):
		if not os.path.exists(artists_summary_path):
			continue
		with open(artists_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions_artists(records, max_x)
		# plot
		label = r"emcid"
		axes[3].plot(
			regions["xs"], 
			regions["lpips_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=emcid_color, 
			label=label)
		
		axes[2].plot(
			regions["xs"], 
			regions["clip_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=emcid_color, 
			label=label)


	for idx, artists_summary_path in enumerate(baseline_summary_paths):
		if not os.path.exists(artists_summary_path):
			continue
		with open(artists_summary_path, 'r') as f:
			records = json.load(f)
		regions = _extract_regions_artists(records, max_x)

		label = r"uce"
		axes[3].plot(
			regions["xs"], 
			regions["lpips_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=uce_color, 
			label=label)
		axes[2].plot(
			regions["xs"], 
			regions["clip_hold_out_y_mean"], 
			marker="o",
			markersize=marker_size,
			color=uce_color, 
			label=label)


	axes[0].set_title(r"COCO-30k: CLIP Score $\uparrow$", fontsize=title_font_size)
	axes[1].set_title(r"COCO-30k: FID Score $\downarrow$", fontsize=title_font_size)
	axes[3].set_title(r"Holdout Artists: LPIPS $\downarrow$", fontsize=title_font_size)
	axes[2].set_title(r"Holdout Artists: CLIP Score $\uparrow$", fontsize=title_font_size)

	

	# for ax in axes[0:1]:
	# 	ax.legend(frameon=False, fontsize=16)
	# set x ticks
	x_ticks = [1, 5, 10, 50, 100, 500, 1000]

	# use log scale for x axis
	import matplotlib.ticker as ticker
	

	x_ticks = [x for x in x_ticks if x <= max_x]
	for ax in axes:
		# plt.rcParams['xtick.minor.size'] = 0
		# plt.rcParams['xtick.minor.width'] = 0
		ax.set_xscale('log')
		ax.set_xticks(x_ticks)
		ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
		ax.set_xticklabels(ax.get_xticks(), rotation=30)
		ax.set_xlabel("Edit Number", fontsize=x_label_font)

	# only one legend at the bottom for the figure, we do this by deduplicating handles and labels
	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))

	# Shrink current axis's height by 10% on the bottom
	
	# bbox_to_dest=(-0.1, -0.3)
	bbox_to_dest=(-0.1, 2.95)

	plt.legend(
		by_label.values(), 
		by_label.keys(), 
		bbox_to_dest=bbox_to_dest,
		fontsize=legend_font,
		loc="upper center", 
		ncol=2, 
		frameon=False)
	# save 
	plt.savefig(os.path.join(emcid_result_folder, "artists_holdout_coco_eval.png"), bbox_inches='tight',
			 	dpi=600)

	# save as pdf
	plt.savefig(os.path.join(emcid_result_folder, "artists_holdout_coco_eval.pdf"), bbox_inches='tight')


if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, default="imgnet_aug")
	parser.add_argument("--trade_off_edit", type=int, default=300)
	parser.add_argument("--lpips", action="store_true", default=False)
	parser.add_argument("--max_x", type=int, default=300)
	parser.add_argument("--min_x", type=int, default=1)

	args = parser.parse_args()
	if args.dataset == "artists":
		plot_lpips_and_clip_artists(
			emcid_result_folder="results/emcid",
			baseline_result_folder="results/baselines",
			plot_edit_line=False,
			plot_clip=True,
			plot_bound=False,
			max_x=args.max_x
		)
		plot_clip_and_fid_coco(
			emcid_result_folder="results/emcid",
			baseline_result_folder="results/baselines",
			plot_lpips=True if args.lpips else False,
			max_x=args.max_x,
		)
		plot_coco_and_artists(
			emcid_result_folder="results/emcid",
			baseline_result_folder="results/baselines",
			max_x=args.max_x,
		)
	else:
		traverse_results(
			emcid_result_folder="results/emcid", 
			baseline_result_folder="results/baselines",
			dataset=args.dataset,
			trade_off_edit=args.trade_off_edit,
			max_x=args.max_x,
			min_x=args.min_x,
			plot_full=True)
	

	