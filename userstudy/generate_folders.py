import numpy as np
import csv
import string, random
import sys, os

SCENES = ["bonsai", "bicycle", "drjohnson", "truck"]
METHOD_NAMES = [ "ms_pretrained", "ms_finetuned_ewa_dist"]
SORTING = ["z", "dist", "z", "dist"]

NUM_USERS = 25
NUM_SCENES_TO_TEST = 4
CL_ARGS = "--rendering-mode 2 --rendering-size 4128 2272 --vsync 0"

DATASET_ROOT = "../../VR_userstudy_models"
exit()

# testing NUM_SCENES_TO_TEST scenes

for user_id in range(NUM_USERS):
    used_scenes = np.random.permutation(SCENES)
    used_sorting = np.random.permutation(SORTING)

    ID = ''.join((random.choice(string.ascii_uppercase + string.digits) for i in range(6)))

    USER_DIR = f"user_{user_id}"
    os.makedirs(USER_DIR, exist_ok=True)
    with open(f"{USER_DIR}/selection.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        
        # Writing the header (optional)
        writer.writerow(['test_index', 'used_scene', 'method_1', 'method_2', 'sorting'])
        
        # Loop through the arrays and write data to CSV and generate .bat files
        for y in range(NUM_SCENES_TO_TEST):
            # randomly shuffle the order of methods
            used_methods = np.random.permutation([f"{METHOD_NAMES[0]}_{used_sorting[y]}", METHOD_NAMES[1]])
            
            writer.writerow([y, used_scenes[y], used_methods[0], used_methods[1], used_sorting[y]])

            with open(f'{USER_DIR}/run_{y+1}_method_1.bat', 'w') as bat_file:
                bat_file.write(f"start \"\" \"../install/bin/SIBR_gaussianViewer_app.exe\" -m {DATASET_ROOT}/{used_methods[0]}/{used_scenes[y]} {CL_ARGS}")
            
            with open(f'{USER_DIR}/run_{y+1}_method_2.bat', 'w') as bat_file:
                bat_file.write(f"start \"\" \"../install/bin/SIBR_gaussianViewer_app.exe\" -m {DATASET_ROOT}/{used_methods[1]}/{used_scenes[y]} {CL_ARGS}")
        
        writer.writerow([])
        writer.writerow(['ID', ID])

print("CSV and .bat files have been created.")