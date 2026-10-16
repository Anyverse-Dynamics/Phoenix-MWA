import os
import shutil
import glob

folder = "/inspire/hdd/project/embodied-multimodality/public/syfei/libero_new/multi-env/LIBERO-pro/libero/libero/assets/textures"
for file in glob.glob(os.path.join(folder, '*')):  # 当前目录
    if '-' in os.path.basename(file):
        new_name = file.split("/textures")[0] + "/textures" + file.split("/textures")[-1].replace('-', '_')
        shutil.copy2(file, new_name)
        print(f"已复制: {file} -> {new_name}")