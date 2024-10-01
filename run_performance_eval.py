import os, shutil

root_dir = "../3DGS_VR_models"
EVAL_DIRS = [
    "ms_finetuned_ewa_dist",
    # "ms_finetuned_ewa_z",
]

CONFIGS = {
    # 'StopThePop': "config_stp.json",
    # 'OptimalProjection': "config_stp_optimal.json"
    # 'OptimalProjection_oldfoveated': "config_stp_optimal_foveated.json"
    'OptimalProjection_newfoveated': "config_stp_optimal_foveated.json"
}

PERF_OUT_FILENAME = 'perf_metrics'

for eval_dir in EVAL_DIRS:
    cam_paths_dir = os.path.join(root_dir, eval_dir)

    for scene in sorted(os.listdir(cam_paths_dir)):

        if not (os.path.isdir(os.path.join(cam_paths_dir, scene))):
            continue
        scene_dir = cam_paths_dir + '/' + scene
        
        print(f'Scene: {scene}')        
        model_dir = os.path.join(cam_paths_dir, f"{scene}")
        print(model_dir)

        # move the original config back one folder
        shutil.move(
            src=os.path.join(model_dir, 'config.json'),
            dst=os.path.join(cam_paths_dir, 'config.json'),
        )

        for cfg_name, cfg_path in CONFIGS.items():

            # delete the file if it exists
            FILENAME = f'{PERF_OUT_FILENAME}_{cfg_name}.txt'
            if (os.path.exists(f'{model_dir}/{FILENAME}')):
                os.remove(f'{model_dir}/{FILENAME}')

            # replace the config with the currently selected config
            shutil.copy(
                src=os.path.join(cam_paths_dir, cfg_path),
                dst=os.path.join(model_dir, 'config.json'),
            )

            command = f'{os.getcwd()}/install/bin/SIBR_gaussianViewer_app.exe -m {model_dir} --rendering-mode 0 --rendering-size 2064 2272 --vsync 0 --force-aspect-ratio'
            # print(command)
            os.system(f'{command} >> {model_dir}/{FILENAME}' )

            # remove the config again
            os.remove(os.path.join(model_dir, 'config.json'))
        
        # move the original config back one folder
        shutil.move(
            src=os.path.join(cam_paths_dir, 'config.json'),
            dst=os.path.join(model_dir, 'config.json'),
        )
