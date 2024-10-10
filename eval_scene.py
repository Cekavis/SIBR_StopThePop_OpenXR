
from collections import defaultdict
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CATEGORIES = ["Low", "High", "Processing"]

MIP360_INDOOR = ("mip360_indoor", ["kitchen", "bonsai", "counter", "room"])
MIP360_OUTDOOR = ("mip360_outdoor", ["stump", "bicycle", "treehill", "flowers", "garden"])
ALL_DATASETS = [MIP360_INDOOR, MIP360_OUTDOOR]

# TANDT = ("TaT", ["train", "truck"])
# DB = ("DB", ["drjohnson", "playroom"])
# ALL_DATASETS = ALL_DATASETS + [TANDT, DB]

TANDTDB = ("tatdb", ["train", "truck", "drjohnson", "playroom"])
ALL_DATASETS = ALL_DATASETS + [TANDTDB]

SCENES_TO_DATASET = {scene: dataset for (dataset, scenes) in ALL_DATASETS for scene in scenes}
DATASETS_TO_SCENES = {dataset: scenes for (dataset, scenes) in ALL_DATASETS}

OUT_SUFFIX = ""

METHODS = {
    "3dgs": {
        # "": "\\gs (old)",
        "_new": "\\gs",
    },
    "ms_pretrained_dist_orig": {
        # "": "\\minigs (old)",
        "_new": "\\minigs"
    },
    "ms_finetuned_ewa_dist": {
        "_OptimalProjection_oldfoveated": "\\ours (2-pass fov.)",
        "_StopThePop_nofoveated_noopti_new": "\\minigs + \\stp",
        "_OptimalProjection_nofoveated_new": "\\ours (w/o fov.)",
        "_OptimalProjection_newfoveated": "\\ours",
        "_StopThePop_nofoveated_noopti_new_new": "\\minigs + \\stp (new)",
        "_StopThePop_nofoveated_noopti_new_new_nomask": "\\minigs + \\stp (new nomask)",
        "_OptimalProjection_nofoveated_new_new": "\\ours (w/o fov.) (new)",
    }
}

def get_files(a_dir):
    return sorted([name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and ".txt" in name])

results = []

for config_dir in [c for c in os.listdir(".") if os.path.isdir(c)]:
    
    if config_dir not in METHODS:
        continue
    
    for scene_folder_name in [c for c in os.listdir(config_dir) if os.path.isdir(os.path.join(config_dir, c))]:
        scene = str.split(scene_folder_name, "_")[0]
        scene_path = os.path.join(config_dir, scene_folder_name)
        

        tmp_dict = METHODS[str(config_dir)]

        for run_log_suffix in tmp_dict.keys():

            runs = defaultdict(list)
            run_log_filename = os.path.join(scene_path, f"perf_metrics{run_log_suffix}.txt")
            if not os.path.exists(run_log_filename):
                print(run_log_filename)
                break
            
            with open(run_log_filename) as f:
                full_text = f.read()
                
                for category in CATEGORIES:
                    timings = [float(prep.split(" ")[-1][:-2]) for prep in re.findall(fr'- {category}: [\d]*[.]?[\d]+ms', full_text)][1:]
                    runs[category].append(timings)
                            
            # if not os.path.exists(os.path.join(scene_path, "metrics.json")):
            #     continue
            # num_gaussians = json.load(open(os.path.join(scene_path, "metrics.json")))["num_gaussians"]
            num_gaussians = 0
            
            for category in CATEGORIES:
                category_run = np.array(runs[category])
                plt.plot(np.mean(category_run, axis=0), label=f"{category}")
                plt.fill_between(np.arange(category_run.shape[1]), np.min(category_run, axis=0), np.max(category_run, axis=0), color="lightgrey")
                
            category_mean_curves = [np.mean(runs[category], axis=0) for category in CATEGORIES]
            total_mean_curve = np.sum(category_mean_curves, axis=0)
            
            if config_dir not in METHODS:
                print(config_dir)
                continue
            
            method_name = METHODS[str(config_dir)][run_log_suffix]
            results.append([method_name, SCENES_TO_DATASET[scene], scene, num_gaussians, np.mean(total_mean_curve)] + [np.mean(c, axis=0) for c in category_mean_curves])
            
            print(config_dir, run_log_suffix, scene, np.mean(total_mean_curve))
            plt.plot(total_mean_curve, label="total")
            plt.ylabel("Timing in ms")
            plt.xlabel("Camera pose")
            plt.title("Timing over camera path by category")
                
            plt.legend()
            plt.tight_layout()
            # plt.savefig(os.path.join(scene_path, "timings.pdf"))
            # plt.show()
            plt.close()
        
results_df = pd.DataFrame(results, columns=["method", "dataset", "scene", "num_gaussians", "total"] + CATEGORIES)
results_df.to_csv("data_ps.csv")
full_results_df = pd.pivot_table(results_df, values=["total"] + CATEGORIES, index=["method"], aggfunc='mean').reindex(columns=CATEGORIES + ["total"])


def highlight_top_n(styler, df, styles, ascending, precision):
    def custom_style(v, column, ascending, styles):
        sorted_vals = df.sort_values(column, ascending=ascending)[column].values
        for i in range(len(styles)-1):
            if (not ascending and v >= sorted_vals[i]) or (ascending and v <= sorted_vals[i]):
                return styles[i]
        return styles[-1]
    
    for scene in df.columns:
        styler.map(custom_style, column=scene, ascending=ascending, styles=styles, subset=scene)
        
    styler.format(precision=precision)
    return styler
        
        
N_TOP_HIGHLIGHTED = 3
HIGHLIGHT_SHADE_FROM, HIGHLIGHT_SHADE_TO = 50, 10
HIGHLIGHT_SHADE_RANGE = HIGHLIGHT_SHADE_TO - HIGHLIGHT_SHADE_FROM
HIGHLIGHT_SHADE_STEPSIZE = 0 if N_TOP_HIGHLIGHTED <= 1 else (HIGHLIGHT_SHADE_RANGE // (N_TOP_HIGHLIGHTED - 1))

highlight_styles = [f"cellcolor:{{tab_color!{HIGHLIGHT_SHADE_FROM + i * HIGHLIGHT_SHADE_STEPSIZE}}}" for i in range(N_TOP_HIGHLIGHTED)]
highlight_styles = highlight_styles + [f"cellcolor:{{tab_color!0}}"]
s = full_results_df.style.pipe(highlight_top_n, df=full_results_df, styles=highlight_styles, ascending=True, precision=2)
s.to_latex(f"full_performance{OUT_SUFFIX}_ps.tex", hrules=True)

full_datasets_df = pd.pivot_table(results_df, values=["total"], columns=["dataset"], index=["method"], aggfunc='mean')["total"]
s = full_datasets_df.style.pipe(highlight_top_n, df=full_datasets_df, styles=highlight_styles, ascending=True, precision=2)
s.to_latex(f"full_performance_datasets{OUT_SUFFIX}_ps.tex", hrules=True)

for scene_key in SCENES_TO_DATASET.keys():
    dataset_result_df = results_df[results_df.scene == scene_key]
    
    dataset_categories_result_df = pd.pivot_table(dataset_result_df, values=["total"] + CATEGORIES, index=["method"], aggfunc='mean').reindex(columns=CATEGORIES + ["total"])
    s = dataset_categories_result_df.style.pipe(highlight_top_n, df=dataset_categories_result_df, styles=highlight_styles, ascending=True, precision=2)
    s.to_latex(f"per_scene/{scene_key}_cat_performance{OUT_SUFFIX}_ps.tex", hrules=True)
    
    dataset_scenes_total_result_df = pd.pivot_table(dataset_result_df, values=["total"], columns=["scene"], index=["method"], aggfunc='mean')["total"]
    s = dataset_scenes_total_result_df.style.pipe(highlight_top_n, df=dataset_scenes_total_result_df, styles=highlight_styles, ascending=True, precision=2)
    s.to_latex(f"per_scene/{scene_key}_scenes_total_performance{OUT_SUFFIX}_ps.tex", hrules=True)
    
    # print("#Gausians in Millions:" , pd.pivot_table(dataset_result_df[dataset_result_df.method == "(A) Ours"], values=["num_gaussians"], columns=["scene"], index=["method"], aggfunc='mean')["num_gaussians"] / 1000000)
    
for dataset_key, dataset_scenes in ALL_DATASETS:
    dataset_result_df = results_df[results_df.dataset == dataset_key]
    
    dataset_categories_result_df = pd.pivot_table(dataset_result_df, values=["total"] + CATEGORIES, index=["method"], aggfunc='mean').reindex(columns=CATEGORIES + ["total"])
    s = dataset_categories_result_df.style.pipe(highlight_top_n, df=dataset_categories_result_df, styles=highlight_styles, ascending=True, precision=2)
    s.to_latex(f"{dataset_key}_cat_performance{OUT_SUFFIX}_ps.tex", hrules=True)
    
    dataset_scenes_total_result_df = pd.pivot_table(dataset_result_df, values=["total"], columns=["scene"], index=["method"], aggfunc='mean')["total"]
    s = dataset_scenes_total_result_df.style.pipe(highlight_top_n, df=dataset_scenes_total_result_df, styles=highlight_styles, ascending=True, precision=2)
    s.to_latex(f"{dataset_key}_scenes_total_performance{OUT_SUFFIX}_ps.tex", hrules=True)
    
#     print("#Gausians in Millions:" , pd.pivot_table(dataset_result_df[dataset_result_df.method == "(A) Ours"], values=["num_gaussians"], columns=["scene"], index=["method"], aggfunc='mean')["num_gaussians"] / 1000000)
    