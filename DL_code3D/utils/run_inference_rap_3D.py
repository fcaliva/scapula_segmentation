import os
import subprocess
import pandas
import pickle
os.chdir("/data/knee_mri8/Francesco/scapula_project_RAP/DL_code3D")

path_with_cfg = "/data/knee_mri8/Francesco/scapula_project_RAP/DL_code3D/cfgs_new/inference"
for root, dirs, files in os.walk(path_with_cfg):
    if "/scapula_augmented_3D" in root or  "/scapula3D" in root:
        for name in files:
            if name.endswith((".yaml")):
                yaml_file =  os.path.join(root,name)
                curr_desc = "_".join(yaml_file.split("/")[8:]).split(".")[0]
                # print(curr_desc)
                subprocess.call(['python', '/data/knee_mri8/Francesco/scapula_project_RAP/DL_code3D/main_infer.py', '--cfg', yaml_file, '--desc', f'DL3D_{curr_desc}', '--gpu', '5'])
