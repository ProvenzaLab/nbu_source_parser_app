import paramiko
import os
import pypdf
import shutil
import json
from pathlib import Path
from config import SSH_INFO, ROOT, STUDY_IDS

def save_object_remote(target_folder, obj, obj_type='figure', ssh_info=SSH_INFO, filename=None):
    """
    Save an object to a remote folder using SSH.
    - target_folder: remote folder path (str)
    - obj: object to save (e.g., matplotlib figure)
    - obj_type: type of object ('figure', can expand for others)
    - ssh_info: dict with SSH connection info {host, port, username, password or key_filename}
    - filename: name for the saved file (optional)
    """
    if ssh_info is None:
        raise ValueError("ssh_info must be provided with host, port, username, and password or key_filename.")

    # Connect to SSH
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    if 'password' in ssh_info:
        ssh.connect(ssh_info['host'], username=ssh_info['username'], password=ssh_info['password'])
    else:
        ssh.connect(ssh_info['host'], username=ssh_info['username'], key_filename=ssh_info['key_filename'])

    sftp = ssh.open_sftp()
    try:
        # Create remote folder if it doesn't exist
        try:
            sftp.stat(target_folder)
        except FileNotFoundError:
            cmd = f'mkdir -p {target_folder}'
            ssh.exec_command(cmd)

        # Save object
        if obj_type == 'figure':
            if filename is None:
                filename = 'figure.pdf'
            local_tmp = os.path.join(os.getcwd(), filename)
            obj.savefig(local_tmp, bbox_inches='tight')
            remote_path = os.path.join(target_folder, filename)
            sftp.put(local_tmp, remote_path)
            os.remove(local_tmp)
        else:
            raise NotImplementedError(f"Saving object type '{obj_type}' not implemented.")
    finally:
        sftp.close()
        ssh.close()
    return

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