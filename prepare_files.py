import os
import pypdf
import shutil
import json
from pathlib import Path
from data_viz_plot import STUDY_IDS

ROOT = Path('/home/nbusleep/data') 

def sort_lfp(json_files, pdf_files, pt):
    study_id = STUDY_IDS[pt[:-3]]

    for file in json_files:
        try:
            with open(file, 'r') as f:
                lfp_data = json.load(f)
        except (PermissionError, json.JSONDecodeError, FileNotFoundError) as e:
            print(f'Could not read file: {file}, {e}')
            continue
            
        implant_loc = lfp_data['DeviceInformation']['Initial']['NeurostimulatorLocation']

        if 'CHEST_LEFT' in implant_loc:
            hemi = 'L'

        elif 'CHEST_RIGHT' in implant_loc:
            hemi = 'R'
        
        else:
            print(f'Invalid Neurostimulator loaction for file: {file}')
            continue

        save_dir = ROOT / 'LFP_DATA' / study_id / hemi / pt
        os.makedirs(save_dir, exist_ok=True)

        try:
            shutil.move(file, save_dir)
        except (FileNotFoundError, shutil.Error) as e:
            print(f'Could not move file: {file}, {e}')
            continue

    for file in pdf_files:
        try:
            reader = pypdf.PdfReader(file)
        except Exception as e:
            print(f'Could not read file: {file}, {e}')
            continue

        text = reader.pages[0].extract_text()

        if 'Left Chest' in text:
            hemi = 'L'

        elif 'Right Chest' in text:
            hemi = 'R'
        
        else:
            print(f'Invalid Neurostimulator loaction for file: {file}')
            continue
        
        save_dir = ROOT / 'LFP_DATA' / study_id / hemi / pt
        os.makedirs(save_dir, exist_ok=True)

        try:
            shutil.move(file, save_dir)
        except (FileNotFoundError, shutil.Error) as e:
            print(f'Could not move file: {file}, {e}')
            continue
        
    return

def sort_cgx(cgx_files, pt):
    study_id = STUDY_IDS[pt[:-3]]

    for file in cgx_files:
        file_seg = str(file).split('.')[0].split('_')
        date = f'{file_seg[-6]}-{file_seg[-5]}-{file_seg[-4]}'

        save_dir = ROOT / 'CGX_DATA' / study_id / pt / date
        os.makedirs(save_dir, exist_ok=True)

        try:
            shutil.move(file, save_dir)
        except (FileNotFoundError, shutil.Error) as e:
            print(f'Could not move file: {file}, {e}')
            continue
        
    return