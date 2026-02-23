import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import random
import os
import glob
from datetime import datetime, timedelta
from pathlib import Path
import json
from percept_parser.percept import PerceptParser
from config import STUDY_IDS, DATALAKE, TZ, LOGGER_COLORS, ROOT 

def get_folders_in_range(parent_dir, start_folder_name, end_folder_name):
    """
    Gets all subfolders in a parent directory that fall alphabetically 
    between the start and end folder names.

    Args:
        parent_dir (str/Path): The path to the main directory to search in.
        start_folder_name (str): The starting boundary folder name (inclusive).
        end_folder_name (str): The ending boundary folder name (inclusive).

    Returns:
        list: A list of full paths to the folders within the range.
    """
    p = Path(parent_dir)
    # List all items in the directory and filter for actual directories
    all_entries = [entry for entry in p.iterdir() if entry.is_dir()]
    
    # Sort the list alphabetically by name
    all_entries.sort(key=lambda x: x.name)
    
    # Filter for entries within the specified range (inclusive)
    folders_in_range = []
    for entry in all_entries:
        name = entry.name
        if start_folder_name <= name <= end_folder_name:
            folders_in_range.append(str(entry.resolve())) # Use resolve() for absolute path
            
    return folders_in_range

def get_files_from_folder(pt, visit_start, visit_end, modality, loc=None):
    study_id = STUDY_IDS[pt[:-3]]
    arr = []

    if modality == 'logger':
        paths = get_folders_in_range(DATALAKE / study_id  / pt / 'NBU', visit_start.strftime('%Y-%m-%d'), visit_end.strftime('%Y-%m-%d'))
        def func(path):   return glob.glob(os.path.join(Path(path) / 'events', "*.csv"))
    
    elif modality == 'oura':
        paths = get_folders_in_range(DATALAKE / study_id  / pt / 'oura', visit_start.strftime('%Y-%m-%d'), visit_end.strftime('%Y-%m-%d'))
        def func(path):   return glob.glob(os.path.join(Path(path), f"{loc}.json"))

    elif modality == 'apple':
        paths = get_folders_in_range(DATALAKE / study_id  / pt / 'rune', visit_start.strftime('%Y-%m-%d'), visit_end.strftime('%Y-%m-%d'))
        def func(path): return glob.glob(os.path.join(Path(path) / "heart_rate" ,  "apple_watch*.csv"))

    elif modality == 'cgx':
        paths = get_folders_in_range(DATALAKE / study_id  / pt / 'NBU', visit_start.strftime('%Y-%m-%d'), visit_end.strftime('%Y-%m-%d'))
        def func(path):   return glob.glob(os.path.join(Path(path) / 'CGX', "*.cgx"))

    elif modality == 'video':
        paths = get_folders_in_range(DATALAKE / study_id / pt / 'NBU', visit_start.strftime('%Y-%m-%d'), visit_end.strftime('%Y-%m-%d'))
        def func(path):   return glob.glob(os.path.join(Path(path) / 'video' / loc, "*.json"))
    else:
        return ValueError
    
    for path in paths:
        arr.extend(func(path))

    return arr

def get_cgx_times(file_name):

    file_seg = file_name.split(os.sep)[-1].split('.')[0].split('_')
    end_time = datetime.strptime(f'{file_seg[-6]}-{file_seg[-5]}-{file_seg[-4]}T{file_seg[-3]}:{file_seg[-2]}:{file_seg[-1]}', '%d-%m-%yT%H:%M:%S')
    file_seg = file_seg[:-7]
    start_time = datetime.strptime(f'{file_seg[-6]}-{file_seg[-5]}-{file_seg[-4]}T{file_seg[-3]}:{file_seg[-2]}:{file_seg[-1]}', '%d-%m-%yT%H:%M:%S')

    start_time = pd.Timestamp(start_time).tz_localize('America/Chicago')
    end_time = pd.Timestamp(end_time).tz_localize('America/Chicago')

    return start_time, end_time

def find_continuous_segments(times, max_gap=1.0):
    """
    Find continuous segments in datetime array.
    
    Parameters:
    - times: array of datetime objects
    - max_gap: maximum gap to consider continuous (seconds)
    
    Returns:
    - List of (start_idx, end_idx) tuples for each segment
    """
    if len(times) == 0:
        return []
    
    # Convert to numpy array if not already
    times = np.array(times)
    
    # Calculate time differences in seconds
    gaps = np.diff(times) / np.timedelta64(1, 's')
    
    # Find where gaps exceed threshold
    break_points = np.where(gaps >= max_gap)[0]
    
    # Build segments
    segments = []
    start_idx = 0
    
    for break_idx in break_points:
        segments.append((start_idx, break_idx + 1))
        start_idx = break_idx + 1
    
    # Add final segment
    segments.append((start_idx, len(times)))
    
    return segments

def read_lfp_data(pt, visit_start, visit_end):
    study_id = STUDY_IDS[pt[:-3]]
    # Lab worlds folder
    #target_folder = Path('/mnt/projectworlds') / study_id / pt / 'NBU' / 'compiled_LFP'
    #os.makedirs(target_folder, exist_ok=True)

    # Gather all LFP files that were 'uploaded' (modified) after the visit start
    lfp_path = DATALAKE / study_id / pt / 'LFP'
    paths = glob.glob(os.path.join(f'{lfp_path}/**/*.json'), recursive=True)

    lfp_files = []
    start_timestamp = visit_start.replace(tzinfo=None)
    for fp in paths:
        mod_time = datetime.fromtimestamp(os.path.getmtime(fp))
        if start_timestamp < mod_time:
            lfp_files.append(fp)

    if len(lfp_files) == 0:
        return None
    
    # Run percept parser on the LFP files to create a single dataframe
    chunks = []
    for filename in lfp_files:
        try:
            parser = PerceptParser(filename)
            # skip if lead location is not VC/VS, for now
            if parser.lead_location != 'OTHER' and parser.lead_location != 'AIC':
                print(f'Skipping {filename}, lead location: {parser.lead_location}')
                continue
            td_data = parser.read_timedomain_data(indefinite_streaming=False)
            id_data = parser.read_timedomain_data(indefinite_streaming=True)
        except Exception as e:
            print(f'Error processing {filename}: {e}')
            continue

        if len(td_data) != 0:
            td_df = pd.concat(td_data)
            td_df['filename'] = filename
            chunks.append(td_df)

        if len(id_data) != 0:
            id_df = pd.concat(id_data)
            id_df['filename'] = filename
            chunks.append(id_df)

        else:
            print(f'No time domain data for {filename}')

    if len(chunks) == 0:
        print(f'No Time Domain data for {pt}')
        return
    
    pt_df = pd.concat(chunks)
    pt_df.sort_index()

    # Convert timestamps to Central + create times column
    pt_df['times'] = pt_df.index.to_series().dt.tz_localize('UTC').dt.tz_convert('America/Chicago')

    #pt_df.to_pickle(target_folder / f'{visit_start.strftime("%Y-%m-%d_%H-%M-%S")}-{visit_end.strftime("%Y-%m-%d_%H-%M-%S")}_visit_timedomain.pkl')
    return pt_df


def main(pt, visit_start, visit_end, ax):
    visit_start = datetime.strptime(visit_start, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=TZ)
    visit_end = datetime.strptime(visit_end, '%Y-%m-%dT%H:%M:%S').replace(tzinfo=TZ)
    
    try:
        logger_files = get_files_from_folder(pt, visit_start, visit_end, 'logger')
        plot_logger = True
    except FileNotFoundError:
        print(f'No logger data uploaded to Elias for {pt}, {visit_start} visit')
        plot_logger = False
    try:
        oura_sleep_files = get_files_from_folder(pt, visit_start, visit_end, 'oura', 'sleep')
        oura_met_files = get_files_from_folder(pt, visit_start, visit_end, 'oura', 'daily_activity')
        plot_oura = True
    except FileNotFoundError:
        print(f'No Oura data on Elias for {pt} between {visit_start} and {visit_end}')
        plot_oura = False
    try:
        cgx_files = get_files_from_folder(pt, visit_start, visit_end, 'cgx')
        plot_cgx = True
    except FileNotFoundError:
        print(f'No CGX data uploaded to Elias for {pt} on {visit_start} visit')
        plot_cgx = False
    try:
        sleep_video_files = get_files_from_folder(pt, visit_start, visit_end, 'video', 'sleep')
        lounge_video_files = get_files_from_folder(pt, visit_start, visit_end, 'video', 'lounge')
        plot_video = True
    except FileNotFoundError:
        print(f'No video data uploaded to Elias for {pt} on {visit_start} visit')
        plot_video = False
    try:
        lfp_df = read_lfp_data(pt, visit_start, visit_end)
        plot_lfp = True
    except Exception as e:
        print(f'Error retrieving LFP data for {pt} on {visit_start} visit: {e}')
        plot_lfp = False
    if lfp_df is None:
        plot_lfp = False
    
    # LFP Plotting
    if plot_lfp:
        lfp_df.dropna(subset=lfp_df.columns.difference(['filename', 'times']), how='all', inplace=True)
        lfp_df.sort_values(by='times', inplace=True)

        # Plot each continuous segment of TimeDomain data
        segments = find_continuous_segments(lfp_df.times.values)

        for start_idx, end_idx in segments:
            start = lfp_df.times.values[start_idx]
            end = lfp_df.times.values[end_idx - 1]
            color = [random.random() for _ in range(3)]

            ax.axvspan(start, end, ymin=0.1, ymax=0.3, alpha=0.3, color=color)
        
        # Add filenames to plot
        fn_handles = []
        fn_patches = []
        for filename, group in lfp_df.groupby('filename'):
            start = group.times.values[0]
            end = group.times.values[-1]
            label = f'{filename.split(os.sep)[-2]}/{filename.split(os.sep)[-1]}'
            color = [random.random() for _ in range(3)]

            y = 0.05

            ann = ax.annotate(
                '',
                xy=(end, y),          
                xytext=(start, y),    
                arrowprops=dict(
                    arrowstyle="|-|",      
                    color=color,
                    lw=2,                 
                )
            )

            fn_handles.append(label)
            fn_patches.append(mpatches.Patch(color=color))

    # Add and label logger events to the plot
    log_handles = []
    log_labels = []
    log_colors = {}
    col_idx = 0

    # Logger Plotting
    if plot_logger:
        for logger_fp in logger_files:
            log_df = pd.read_csv(logger_fp)
            for j, log_row in log_df.iterrows():
                if log_row['Notes'] == 'ABORTED' or log_row['Event'] == 'SESSION START' or log_row['Event'] == 'SESSION END':
                    continue

                try:
                    log_start = datetime.strptime(f"{log_row['Start Date']} {log_row['Start Time']}".strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
                    log_end = datetime.strptime(f"{log_row['End Date']} {log_row['End Time']}".strip(), '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ)
                    if log_end < log_start or log_start < visit_start or visit_end < log_start:
                        continue   

                    label = f"{log_start.strftime('%I:%M %p')} - {log_end.strftime('%I:%M %p')}: {log_row['Event']}"
                    if log_row['Event'] not in log_colors:
                        log_colors[log_row['Event']] = LOGGER_COLORS[col_idx]
                        color = LOGGER_COLORS[col_idx]
                        col_idx += 1
                    else:
                        color = log_colors[log_row['Event']]

                    ax.axvspan(log_start, log_end, ymin=0.7, ymax=0.8, color=color, label=label, zorder=5)
                    
                    log_handles.append(mpatches.Patch(color=color))
                    log_labels.append(label)
                except:
                    pass

    # Video Plotting
    if plot_video:
        for i, (video_files, color) in enumerate([(sleep_video_files, '#7fc97f'), (lounge_video_files, '#beaed4')]):
            for video_fp in video_files:
                with open(video_fp, 'r') as f:
                    try:
                        raw = json.load(f)
                    except Exception as e:
                        print(f"Error reading {video_fp} for {pt} on {visit_start.strftime('%Y-%m-%d')}: {e}")
                        continue
                times = pd.to_datetime(raw["real_times"]).tz_localize('UTC').tz_convert(TZ)
                times = times[(times >= visit_start) & (times <= visit_end)]
                if len(times) != 0:
                    ax.axvspan(times.values[0], times.values[-1], ymin=(0.34 + i * 0.1), ymax=(0.36 + i * 0.1), color=color)
            
    # Oura plotting
    if plot_oura:
        for oura_fp in oura_sleep_files:
            with open(oura_fp, 'r') as f:
                raw = json.load(f)
            
           
                for block in raw:
                    try:
                        sleep_start = datetime.strptime(block['bedtime_start'], '%Y-%m-%dT%H:%M:%S%z')
                        sleep_end = datetime.strptime(block['bedtime_end'], '%Y-%m-%dT%H:%M:%S%z')
                    except ValueError:
                        sleep_start = datetime.strptime(block['bedtime_start'], '%Y-%m-%dT%H:%M:%S.%f%z')
                        sleep_end = datetime.strptime(block['bedtime_end'], '%Y-%m-%dT%H:%M:%S.%f%z')
                    except Exception as e:
                        continue
                    type = block['type']

                    if visit_start < sleep_start and sleep_end < visit_end:
                        ax.axvspan(sleep_start, sleep_end, ymin=0.85, ymax=0.87, color='lightblue', zorder=5)
                        ax.text(sleep_start, 0.88, f'{type}', zorder=5, fontsize=8, va='bottom', ha='left')

        # Add oura met data
        met_inset = ax.inset_axes([0, 0.5, 1, 0.1])
        for oura_fp in oura_met_files:
            with open(oura_fp, 'r') as f:
                raw = json.load(f)

                for block in raw:

                    # MET metadata
                    met_info = block.get("met", {})
                    if not met_info:
                        continue

                    met_start = datetime.strptime(
                        met_info["timestamp"], "%Y-%m-%dT%H:%M:%S.%f%z"
                    )

                    interval_sec = int(met_info.get("interval", 60))
                    met_values = met_info.get("items", [])

                    # Build timestamps for each MET value
                    met_times = [
                        met_start + timedelta(seconds=i * interval_sec)
                        for i in range(len(met_values))
                    ]

                    # Filter to visit window
                    met_times_visit = []
                    met_vals_visit = []

                    for t, val in zip(met_times, met_values):
                        if visit_start <= t <= visit_end and val > 0.9:
                            met_times_visit.append(t)
                            met_vals_visit.append(val)

                    # Plot metabolic activity as points
                    if met_times_visit:
                        met_inset.plot(
                            met_times_visit,
                            met_vals_visit,
                            linewidth=0.4,
                            alpha=0.6,
                            color='blue',
                            zorder=5,
                            label="Oura MET activity"
                        )

        met_inset.spines['top'].set_visible(False)
        met_inset.spines['bottom'].set_visible(False)
        met_inset.set_xticks([])
        met_inset.set_yticks([])
    
    # Add Apple watch data availability (only TRBD pts)
    if pt[:-3] == 'TRBD':
        try:
            watch_files = get_files_from_folder(pt, visit_start, visit_end, 'apple')
        except FileNotFoundError:
            print(f'No Apple Watch data uploaded to Elias for {pt} on {visit_start} visit')
            watch_files = []

        aw_inset = ax.inset_axes([0, 0.6, 1, 0.1])
        for watch_fp in watch_files:
            watch_df = pd.read_csv(watch_fp)

            # Parse timestamps, convert to Central time
            watch_df['time'] = pd.to_datetime(watch_df['time'].values).tz_convert(TZ)

            # Keep only data inside visit window
            watch_df = watch_df[(watch_df['time'] >= visit_start) & (watch_df['time'] <= visit_end)]

            if watch_df.empty:
                continue
            
            watch_df.dropna(subset=['rpm'], inplace=True)
            segments = find_continuous_segments(watch_df.time.values, max_gap=600)

            for start_idx, end_idx in segments:
                start = watch_df.time.values[start_idx]
                end = watch_df.time.values[end_idx - 1]
                aw_inset.plot(watch_df.loc[start_idx:end_idx-1, 'time'], watch_df.loc[start_idx:end_idx-1, 'rpm'], linewidth=0.4, alpha=0.6, zorder=5, color='orange')

        aw_inset.spines['top'].set_visible(False)
        aw_inset.spines['bottom'].set_visible(False)
        aw_inset.set_xticks([])
        aw_inset.set_yticks([])

    # CGX Plotting
    if plot_cgx:
        for cgx_fp in cgx_files:
            start_date = cgx_fp.split(os.sep)[-3]
            cgx_start, cgx_end = get_cgx_times(cgx_fp)
        
            label = f'CGX sleep starting {start_date}'
            ax.axvspan(cgx_start, cgx_end, ymin=0.93, ymax=0.95, color='purple', zorder=5)

    ########################
    # Plot formatting      #
    ########################

    # Iterate through each day in the range
    # Add vertical dotted lines for each day
    current_day = visit_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
    # Add label for the starting day at the start of the plot
    ax.text(visit_start + pd.Timedelta(minutes=10), ax.get_ylim()[1] - 0.01, visit_start.strftime('%b %d'), 
            rotation=0, fontsize=9, ha='left', va='top', weight='bold')
    while current_day <= visit_end:
        next_day = current_day + pd.Timedelta(days=1)
        
        if next_day > visit_start and next_day <= visit_end:
            ax.axvline(next_day, color='black', linestyle=':', linewidth=1, alpha=0.7, zorder=10)
            ax.text(next_day + pd.Timedelta(minutes=10), ax.get_ylim()[1] - 0.01, next_day.strftime('%b %d'), 
                    rotation=0, fontsize=9, ha='left', va='top', weight='bold')
        
        current_day = next_day
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p', tz=TZ))
    ax.set_xlabel("Time (Hour:Minute)")
    ax.set_ylabel(" ")
    ax.set_yticks([0.05, 0.2, 0.35, 0.45, 0.55, 0.65, 0.775, 0.86, 0.94])
    ax.set_yticklabels(['Files', 'Neural Data', 'Sleep Room Video', 'Lounge Video', 'MET Data', 'Apple Watch HR', 'Logger Events', 'Oura Sleep', 'CGX Data'])
    ax.set_title(f"{pt} NBU visit {visit_start} to {visit_end}")
    ax.set_xlim(visit_start, visit_end)

    # Create separate legends for files and logger events
    if plot_lfp:
        file_legend = ax.legend(handles=fn_patches, labels=fn_handles, loc='center left', bbox_to_anchor=(1, 0.25), title='File Names', fontsize="small")
        ax.add_artist(file_legend)
    if plot_logger:
        ax.legend(handles=log_handles, labels=log_labels, loc='center left', bbox_to_anchor=(1, 0.75), title="Logger Events", fontsize="small")

    return ax

if __name__ == '__main__':
    # Manual usage

    pt = 'TRBD001'  # Change as needed
    visit_start = '2026-02-19T08:00:00'  # Change as needed
    visit_end = '2026-02-20T08:00:00'   # Change as needed

    fig, ax = plt.subplots(figsize=(15, 8), constrained_layout=True)
    ax = main(pt, visit_start, visit_end, ax)
    fig.tight_layout()
    fig.savefig(ROOT / 'SUMMARY_PLOTS' / f'{pt}_{visit_start[:10]}_to_{visit_end[:10]}_summary.png', dpi=150, bbox_inches='tight')
    plt.show()