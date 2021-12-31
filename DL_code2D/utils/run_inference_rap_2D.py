import os
import subprocess
import pandas
import pickle
os.chdir("/data/knee_mri8/Francesco/scapula_project_RAP/DL_code2D")

path_with_cfg = "/data/knee_mri8/Francesco/scapula_project_RAP/DL_code2D/cfgs_new/inference"
cnt = 0
for root, dirs, files in os.walk(path_with_cfg):
    if "/augmented/" in root or  "/axial/" in root or  "/coronal/" in root or "/sagittal/" in root:
        for name in files:
            if name.endswith((".yaml")):
                if cnt < 5:
                    cnt += 1
                    continue
                yaml_file =  os.path.join(root,name)
                curr_desc = "_".join(yaml_file.split("/")[8:]).split(".")[0]
                # print(curr_desc)
                subprocess.call(['python', '/data/knee_mri8/Francesco/scapula_project_RAP/DL_code2D/main_infer.py', '--cfg', yaml_file, '--desc', f'DL2D_{curr_desc}', '--gpu', '4'])
