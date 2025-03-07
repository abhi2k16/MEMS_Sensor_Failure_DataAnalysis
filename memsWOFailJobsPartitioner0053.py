#%%
import sys
sys.path.append('..')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
import glob
import os
from IPython.display import display, HTML
from ctd_partitioner.state_detection import compute_on_well_construction_activity_states
from ctd_partitioner.activity_partitioners import partition_data
#%% Get the .csv files and concat the data in that files in one .csv file 
directory_path = "D:/2025/DataAnalysisShockMEMSSensor/DataWOFail/O.1048592.133-5/TSdata"
csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
############# Get job Id ###########
match = re.search(r'DataWOFail/(O\.\d+\.\d+-\d+)/', directory_path)
if match:
    job_id = match.group(1)
    print(job_id)
data_frames = []
for file in csv_files:
    df = pd.read_csv(file, on_bad_lines='skip')
    df = df.drop(0)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    data_frames.append(df)
concatenated_data = pd.concat(data_frames, ignore_index=True)
concatenated_data = concatenated_data.sort_values(by = 'DateTime')
print(concatenated_data.head())
#%% ######  Write a .csv file for concatenated data and save it in job_id directry ##############
main_directory_path = directory_path.replace("/TSdata", "")
concatenated_data.to_csv(f"{main_directory_path}/{job_id}_concat_data.csv", index=True)
#%% ################  Data reading and index setting ###################################
data = concatenated_data
data['TIME'] = pd.to_datetime(data['DateTime'])
data.index = data['TIME']
columns_of_interest = ['TIME','BVEL','CT_WGT','DEPT', 'HDTH','FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']
data = data[['TIME','BVEL','CT_WGT','DEPT','HDTH', 'FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']]
print(data.head())

for column_name in columns_of_interest:
    data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
#%% Plot whole run data for jobs 
whole_run_data = data
whole_run_data_df = whole_run_data[['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','HDTH','BVEL']]
whole_run_data_df[['SHK_LAT']] = whole_run_data_df[['SHK_LAT']].mask((whole_run_data_df['SHK_LAT'] > 600.0)|
                                                                     (whole_run_data_df['SHK_LAT'] <0))
whole_run_data_df[['VIB_LAT']] = whole_run_data_df[['VIB_LAT']].mask((whole_run_data_df['VIB_LAT'] > 100.0)|
                                                                     (whole_run_data_df['VIB_LAT'] <0))
whole_run_data_df = whole_run_data_df.dropna()
plt.rcParams.update({'font.size': 15})
fig, ax1 = plt.subplots(figsize = (15,5))
ax1.plot(whole_run_data_df['VIB_LAT'],color = 'blue')
ax1.set_ylabel('Lat. Vib. [g]',color = 'blue')
ax1.tick_params(axis = 'y', colors = 'blue')
ax1.set_title(f"Job Id: [{job_id}] Whole Run")
ax1.set_xlabel('Time[MM-DD HH]')
ax1.grid()
ax2 = ax1.twinx()
ax2.plot(whole_run_data_df['SHK_LAT'],color = 'red', alpha = 0.7)
ax2.set_ylabel('Lat. Shock. [g]',color = 'red')
ax2.tick_params(axis = 'y', colors = 'red')
ax3 = ax1.twinx()
ax3.plot(whole_run_data_df['DEPT'], color = 'green' )
ax3.set_ylabel('Bit Depth(ft)',color = 'green')
ax3.tick_params(axis = 'y', colors = 'green')
ax3.spines['right'].set_position(('outward',60))
#ax3.grid(axis='y',linestyle='--', color='gray', alpha=0.7)
plt.show()
#%% ######################### box plot for whole run ###########################
vib_list = [x for x in whole_run_data_df['VIB_LAT'].tolist() if pd.notna(x)] # drop all NaN values
shk_list = [x for x in whole_run_data_df['SHK_LAT'].tolist() if pd.notna(x)]
fig, ax = plt.subplots(1,2, figsize = (10,5))
ax[0].boxplot([vib_list], patch_artist=True, labels =['Lat. Vib'],
           boxprops=dict(facecolor='blue'),
            flierprops=dict(marker='o', color='red'))
ax[0].set_ylabel('Value [g]')
ax[0].grid()
ax[1].boxplot([shk_list], patch_artist=True, labels =['Lat. shock'],
           boxprops=dict(facecolor='blue'),
            flierprops=dict(marker='o', color='red'))
ax[1].grid()
plt.suptitle(f"Job Id: [{job_id}] Whole Run" ,fontsize = 15)
plt.tight_layout()
plt.show()
#%% ############## partitioning of inputed data for a particular job ################
partition_df = partition_data(data)
pd.set_option('display.max_rows', data.shape[0])
partition_df.index = pd.to_datetime(partition_df['Start Time'])
partition_df['Start Time'] = pd.to_datetime(partition_df['Start Time'])
partition_df['End Time'] = pd.to_datetime(partition_df['End Time'])
partitions_sort = partition_df.sort_index()
activities = partition_df['Activity'].unique().tolist()
for activity in activities:
    activity_df = partitions_sort[partitions_sort['Activity'] == activity]
    html_output = activity_df.to_string()
    print(f"Activity: {activity}")
    print(html_output)
#%% 
data1 = data
mean_shk_lat  = []
mean_vib_lat = []
"""
activities = ['Trip In Run', 'On Bottom Time Drilling', 'Drill Run','On Bottom Drilling', 'Drill Off', 'N2 Rate Change Increasing',
             'N2 Rate Change Decreasing', 'N2 Rate On', 'Pull Test','Wiper Trip', 'Trip Out Run', 'N2 Rate Off']"""
activities = partition_df['Activity'].unique().tolist()
activities_to_remove = ['On Bottom Time Drilling','N2 Rate Change Increasing',
                        'N2 Rate Change Decreasing', 'N2 Rate On','N2 Rate Off','Drill Off']
activities = [activity for activity in activities if activity not in activities_to_remove]
for activity in activities:
    ctd_whole_run_data = []
    ctd_partitioner_data = []
    for p in range(1):
        all_activity_data = partitions_sort[['Start Time','End Time','Activity']]
        ctd_partitioner_data.append(all_activity_data)
        whole_run_data = data1[['TIME','VIB_LAT','FLWI','DEPT','BVEL','SHK_LAT']]
        ctd_whole_run_data.append(whole_run_data)
    ctd_partitioner_data_df = pd.concat(ctd_partitioner_data, ignore_index=True)
    ctd_whole_run_data_df = pd.concat(ctd_whole_run_data, ignore_index=True)
    ctd_whole_run_data_df = ctd_whole_run_data_df.reset_index()
    ctd_whole_run_data_df['TIME'] = pd.to_datetime(ctd_whole_run_data_df['TIME'])
    ctd_whole_run_data_df.index = ctd_whole_run_data_df['TIME']
    activity_data = ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
    activity_data_df = pd.DataFrame(columns=['VIB_LAT','FLWI','DEPT','BVEL','SHK_LAT'])
    activity_data_apppend = []
    for r,s in activity_data.iterrows():
        start_time = s['Start Time']
        end_time = s['End Time']
        data_activity = ctd_whole_run_data_df[start_time:end_time]
        activity_data_apppend.append(data_activity)
    activivtywise_data = pd.concat(activity_data_apppend, ignore_index=True)
    parameters_2 = ['VIB_LAT','FLWI','DEPT','BVEL','SHK_LAT']
    plt.rcParams.update({'font.size': 15})
    fig, ax1 = plt.subplots(figsize = (15,7))
    #y = np.arange(0,len(data['SHK_LAT']),1)/60
    ax1.plot(activivtywise_data.VIB_LAT, color = 'green',alpha=0.9)
    ax1.set_ylabel(parameters_2[0],color = 'green')
    #ax1.set_xlabel('Time(Minutes)')

    ax4 = ax1.twinx()
    ax4.plot(activivtywise_data.FLWI/42,color = 'dodgerblue')
    ax4.set_ylabel(parameters_2[1],color = 'dodgerblue')
    ax4.set_ylim([0,3])
    ax4.spines['right'].set_position(('outward',0))
    ax5 = ax1.twinx()
    ax5.plot(activivtywise_data.DEPT,color = 'deeppink',alpha=1,linewidth = 3)
    ax5.set_ylabel(parameters_2[2], color = 'deeppink')
    ax5.spines['right'].set_position(('outward',50))
    ax6 = ax1.twinx()
    ax6.plot(activivtywise_data.BVEL,color = 'brown',alpha=0.9)
    ax6.set_ylabel(parameters_2[3], color = 'brown')
    ax6.spines['right'].set_position(('outward',150))
    ax6.grid()
    ax1.tick_params(axis = 'y', colors = 'green')
    ax4.tick_params(axis = 'y', colors = 'dodgerblue')
    ax5.tick_params(axis = 'y', colors = 'deeppink')
    ax6.tick_params(axis = 'y', colors = 'brown')
    ax1.spines['right'].set_color('green')
    ax4.spines['right'].set_color('dodgerblue')
    ax5.spines['right'].set_color('deeppink')
    ax6.spines['right'].set_color('brown')
    fig.suptitle(f'Job_id: [{job_id}], {activity}',fontsize=25, color = 'blue',fontweight='bold')
    plt.tight_layout()
    plt.show()
    print('Mean Shock of '+ activity + ': '+ str(np.nanmean(activivtywise_data.SHK_LAT)))
    mean_shk_lat.append(round(np.nanmean(activivtywise_data.SHK_LAT, ),2))
    mean_vib_lat.append(round(np.nanmean(activivtywise_data.VIB_LAT, ),2))
#%% ####################  Bar plot for mean shock data ############################
plt.figure(figsize=(10, 6)) 
bars = plt.bar(activities, mean_shk_lat, color=['blue', 'orange', 'green', 'red', 'purple'], edgecolor='black')

plt.xlabel('Activities', fontsize=14)  
plt.ylabel('Mean shock[g]', fontsize=14)  
plt.title(f'Job_id: [{job_id}] Mean Lat. shock', fontsize=16)  
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=75)
plt.show()
#   ################# Bar plot for mean vib data
plt.figure(figsize=(10, 6))  
bars = plt.bar(activities, mean_vib_lat, color=['blue', 'orange', 'green', 'red', 'purple'], edgecolor='black')

plt.xlabel('Activity', fontsize=14)  
plt.ylabel('Mean Vib[g]', fontsize=14)  
plt.title(f'Job_id: [{job_id}] Mean Lat. Vib.', fontsize=16) 
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=75)
plt.show()
#%% ######## function for activity wise S&V plot in time domain and boxplot #########
def mems_sensor_shock_analysis(activities, partitioner_data, main_data):
    """
    ------------------------------------------input variables----------------------------------------------
    activities:  ['Wiper Trip']
    partitioner_data :  Dataset after partition e.g., "ctd_partitions_V_0054"
    main data :  non-partition data e.g., "dataset_for_plots"
    Job_id : e.g., 'O.1048592.29-5'
    -------------------------------------------------------------------------------------------------------
    """
    for activity in activities:
        vib_activity_list = []
        shk_activity_list = []
        ctd_partitioner_data = []
        ctd_main_dataset = []
        ctd_partitions = partitioner_data
        ctd_partitioner = ctd_partitions[['Start Time','End Time','Activity']]
        ctd_partitioner_data.append(ctd_partitioner)
        main_dataset = main_data
        data_m = main_dataset[['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL']]
        ctd_main_dataset.append(data_m)
        ctd_partitioner_data_df = pd.concat(ctd_partitioner_data,ignore_index = False)
        ctd_main_dataset_df = pd.concat(ctd_main_dataset, ignore_index = False)
        activity_data=ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
        activity_data_df = pd.DataFrame(columns = ['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL'])
        ctd_main_dataset_df['TIME'] = pd.to_datetime(ctd_main_dataset_df['TIME'])
        activity_data_append = []
        for r,s in activity_data.iterrows():
            start_time = s['Start Time']
            end_time = s['End Time']
            data_activity = ctd_main_dataset_df[start_time:end_time]
            activity_data_append.append(data_activity)
        activity_data_df = pd.concat(activity_data_append,ignore_index = False)
        activity_data_df[['SHK_LAT']] = activity_data_df[['SHK_LAT']].mask(
            (activity_data_df['SHK_LAT'] > 600.0)|
            (activity_data_df['SHK_LAT'] < 0))
        activity_data_df[['VIB_LAT']] = activity_data_df[['VIB_LAT']].mask(
                        (activity_data_df['VIB_LAT'] > 100.0)|
                        (activity_data_df['VIB_LAT'] < 0))
        activity_data_df = activity_data_df.dropna()
        vib_activity_list.append(activity_data_df['VIB_LAT'].values)
        shk_activity_list.append(activity_data_df['SHK_LAT'].values)
    
        fig, ax1 = plt.subplots(figsize = (15,7))
        ax1.plot(activity_data_df['VIB_LAT'],'-', color = 'blue',)
        ax1.set_ylabel('Lat. Vib. [g]',color = 'blue')
        ax1.set_xlabel('Time[MM-DD HH]')

        ax1.tick_params(axis = 'y', colors = 'blue')
        ax1.set_title(f"[{job_id}] activity: {activity}")
        ax1.grid()
        #ax1.set_xlim(xlim)
        #ax1.grid(True, which='major', color='gray', linewidth=0.7)
        ax2 = ax1.twinx()
        ax2.plot(activity_data_df['SHK_LAT'],'-',color = 'red')
        ax2.set_ylabel('Lat. Shock[g]',color = 'red')
        ax2.tick_params(axis = 'y', colors = 'red')
        ax3 = ax1.twinx()
        ax3.plot(activity_data_df['DEPT'],color = 'green')
        ax3.set_ylabel('Bit Depth(ft)',color = 'green')
        ax3.tick_params(axis = 'y', colors = 'green')
        ax3.spines['right'].set_position(('outward',60))
        #ax3.set_ylim([0,0.25])
        plt.show()
        fig, ax = plt.subplots(1,2, figsize = (10,5))
        ax[0].boxplot(vib_activity_list, patch_artist=True, labels =['Lat. Vib'],
                   boxprops=dict(facecolor='blue'),
                    flierprops=dict(marker='o', color='red'))
        ax[0].set_ylabel('Value [g]')
        ax[0].grid()
        ax[1].boxplot(shk_activity_list, patch_artist=True, labels =['Lat. shock'],
                   boxprops=dict(facecolor='blue'),
                    flierprops=dict(marker='o', color='red'))
        ax[1].grid()
        plt.suptitle(f"Job Id: [{job_id}] {activity}" ,fontsize = 15)
        plt.tight_layout()
        plt.show()
#%% ########################### Excute the function ##################################
activities = partition_df['Activity'].unique().tolist()
activities_to_remove = ['On Bottom Time Drilling','N2 Rate Change Increasing',
                        'N2 Rate Change Decreasing', 'N2 Rate On','N2 Rate Off','Drill Off']
activities = [activity for activity in activities if activity not in activities_to_remove]
vib_data = mems_sensor_shock_analysis(activities=activities, 
                                      partitioner_data=partitions_sort,
                                        main_data=data)
#%% ##########function for get activity wise S&V data for given jobs ###############
def mems_SnV_analysis(activities, partitioner_data, main_data):
    """---------------------------------------input variables----------------------------------------------
    activities:  ['Wiper Trip']
    partitioner_data :  Dataset after partition e.g., "ctd_partitions_V_0054"
    main data :  non-partition data e.g., "dataset_for_plots"
    Job_id : e.g., 'O.1048592.29-5'
    ----------------------------------------------------------------------------------------------------"""
    vib_list_all  = {}
    shock_list_all = {}
    vib_level_count = {}
    for activity in activities:
        vib_level_count[activity] = []
        vib_list_all[activity] = []
        shock_list_all[activity] = []
        ctd_partitioner_data = []
        ctd_main_dataset = []
        ctd_partitions = partitioner_data
        ctd_partitioner = ctd_partitions[['Start Time','End Time','Activity']]
        ctd_partitioner_data.append(ctd_partitioner)
        main_dataset = main_data
        data_m = main_dataset[['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL']]
        ctd_main_dataset.append(data_m)
        ctd_partitioner_data_df = pd.concat(ctd_partitioner_data,ignore_index = False)
        ctd_main_dataset_df = pd.concat(ctd_main_dataset, ignore_index = False)
        activity_data=ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
        activity_data_df = pd.DataFrame(columns = ['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL'])
        ctd_main_dataset_df['TIME'] = pd.to_datetime(ctd_main_dataset_df['TIME'])
        activity_data_append = []
        for r,s in activity_data.iterrows():
            start_time = s['Start Time']
            end_time = s['End Time']
            data_activity = ctd_main_dataset_df[start_time:end_time]
            activity_data_append.append(data_activity)
        activity_data_df = pd.concat(activity_data_append,ignore_index = False)
        
        activity_data_df[['SHK_LAT']] = activity_data_df[['SHK_LAT']].mask((activity_data_df['SHK_LAT'] > 600.0)|
                                                                           (activity_data_df['SHK_LAT'] < 0))
        activity_data_df[['VIB_LAT']] = activity_data_df[['VIB_LAT']].mask((activity_data_df['VIB_LAT'] > 100.0)|
                                                                           (activity_data_df['VIB_LAT'] < 0))
        activity_data_df = activity_data_df.dropna()
        vib_label = [10, 15, 20]
        for label in vib_label:
            count = activity_data_df[activity_data_df['VIB_LAT']> label]['VIB_LAT'].count()
            vib_level_count[activity].append(count)
        
        vib_level_count[activity].append(activity_data_df['VIB_LAT'].count())
        vib_list_all[activity].append(activity_data_df['VIB_LAT'].values)   
        shock_list_all[activity].append(activity_data_df['SHK_LAT'].values)
    return (vib_list_all, shock_list_all, vib_level_count)
#%% ###########Activity wise S&V box plot in a single plot###################
activities = partition_df['Activity'].unique().tolist()
activities_to_remove = ['On Bottom Time Drilling','N2 Rate Change Increasing',
                        'N2 Rate Change Decreasing', 'N2 Rate On','N2 Rate Off','Drill Off']
activities = [activity for activity in activities if activity not in activities_to_remove]
vib_dict, shk_dict,_ = mems_SnV_analysis(activities=activities, 
                                      partitioner_data=partitions_sort,
                                        main_data=data)
vib_list_data = []
shk_list_data = []
for activity in activities:
    vib_list_data.extend(vib_dict[activity])
    shk_list_data.extend(shk_dict[activity])
plt.figure(figsize=(10,5))
plt.boxplot(vib_list_data, patch_artist=True, labels =activities,
           boxprops=dict(facecolor='blue'),
            flierprops=dict(marker='o', color='red'))
plt.ylabel('Value [g]')
plt.grid()
plt.suptitle(f"Job Id: [{job_id}] Lat. Vib." ,fontsize = 15)
plt.tight_layout()
plt.xticks(rotation=75)
plt.xlabel('Activities',fontsize = 15)
plt.show()
plt.figure(figsize=(10,5))
plt.boxplot(shk_list_data, patch_artist=True, labels =activities,
           boxprops=dict(facecolor='blue'),
            flierprops=dict(marker='o', color='red'))
plt.ylabel('Value [g]')
plt.grid()
plt.suptitle(f"Job Id: [{job_id}] Lat. Shock" ,fontsize = 15)
plt.tight_layout()
plt.xticks(rotation=75)
plt.xlabel('Activity',fontsize = 15)
plt.show()
#%% ###########Cumulative time for different vibration labels in bar plot ##############
activities = partition_df['Activity'].unique().tolist()
activities_to_remove = ['On Bottom Time Drilling','N2 Rate Change Increasing',
                        'N2 Rate Change Decreasing', 'N2 Rate On','N2 Rate Off','Drill Off']
activities = [activity for activity in activities if activity not in activities_to_remove]
_, _,vib_count = mems_SnV_analysis(activities=activities, 
                                      partitioner_data=partitions_sort,
                                        main_data=data)
########################## Vibration Count ratio plot #############################
count_ratio_10g = []
count_ratio_15g = []
count_ratio_20g = []
for activity in activities:
    values = vib_count.get(activity)
    count_ratio_10g.append(values[0]/values[3])
    count_ratio_15g.append(values[1]/values[3])
    count_ratio_20g.append(values[2]/values[3])
barWidth = 0.25
fig = plt.subplots(figsize =(15, 10)) 
br1 = np.arange(len(count_ratio_10g)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
plt.bar(br1, count_ratio_10g, color ='r', width = barWidth, 
        edgecolor ='grey', label ='> 10g') 
plt.bar(br2, count_ratio_15g, color ='g', width = barWidth, 
        edgecolor ='grey', label ='> 15g') 
plt.bar(br3, count_ratio_20g, color ='b', width = barWidth, 
        edgecolor ='grey', label ='> 20g') 
plt.xlabel('Activity', fontweight ='bold', fontsize = 15) 
plt.ylabel('Ratio(Vib. label time/Total Time)', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(count_ratio_10g))], 
        activities)
plt.legend()
plt.title(f'Job Id:{job_id}')
plt.grid()
plt.show()
########################### Vibration Count plot ####################################
count_10g = []
count_15g = []
count_20g = []
for activity in activities:
    values = vib_count.get(activity)
    count_10g.append(values[0])
    count_15g.append(values[1])
    count_20g.append(values[2])
barWidth = 0.25
fig = plt.subplots(figsize =(15, 10)) 
br1 = np.arange(len(count_10g)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2] 
plt.bar(br1, count_10g, color ='r', width = barWidth, 
        edgecolor ='grey', label ='> 10g') 
plt.bar(br2, count_15g, color ='g', width = barWidth, 
        edgecolor ='grey', label ='> 15g') 
plt.bar(br3, count_20g, color ='b', width = barWidth, 
        edgecolor ='grey', label ='> 20g') 
plt.xlabel('Activity', fontweight ='bold', fontsize = 15) 
plt.ylabel('Time [sec]', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(count_10g))], 
        activities)
plt.legend()
plt.title(f'Job Id:{job_id}')
plt.grid()
plt.show()
#%% ################################# Activity-Wise Analysis For All Jobs Combined ####################################### 
directory_for_all_jobs = "D:/2025/DataAnalysisShockMEMSSensor/DataWOFail/AllJobData"
csv_files_allJobs = glob.glob(os.path.join(directory_for_all_jobs, "*.csv"))
#%%
job_ids = []
job_id_wise_csv_data = {}
for files in csv_files_allJobs:
    #print(files)
    match = re.search(r'AllJobData\\(O\.\d+\.\d+-\d+)_concat_data\.csv', files)
    if match:
        job_id = match.group(1)
        job_ids.append(job_id)
        job_id_wise_csv_data[job_id] = pd.read_csv(files)
    else:
        print("Job ID not found")
#%%  ################## Cleaning and indexing the dataset for all job_ids ########################### 
job_wise_cleaned_data = {}
for job_id in job_ids:
    print(job_id)
    data = job_id_wise_csv_data.get(job_id)
    data['TIME'] = pd.to_datetime(data['DateTime'])
    data.index = data['TIME']
    columns_of_interest = ['TIME','BVEL','CT_WGT','DEPT', 'HDTH','FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']
    data = data[['TIME','BVEL','CT_WGT','DEPT','HDTH', 'FLWI', 'APRS_RAW', 'IPRS_RAW', 'N2_RATE','VIB_LAT','SHK_LAT']]
    #print(data.head())
    for column_name in columns_of_interest:
        data[column_name] = pd.to_numeric(data[column_name], errors='coerce')
    job_wise_cleaned_data[job_id] = data
#%% ###################### Partitioning the dataset for all job_ids ############################# 
job_wise_partitioned_data = {}
non_partitioned_job_id = []
for job_id in job_ids:
    try:
        data_to_partition = job_wise_cleaned_data.get(job_id)
        partition_df = partition_data(data_to_partition)
        pd.set_option('display.max_rows', data.shape[0])
        partition_df.index = pd.to_datetime(partition_df['Start Time'])
        partition_df['Start Time'] = pd.to_datetime(partition_df['Start Time'])
        partition_df['End Time'] = pd.to_datetime(partition_df['End Time'])
        partitions_sort = partition_df.sort_index()
        job_wise_partitioned_data[job_id] = partitions_sort
    except:
        non_partitioned_job_id.append(job_id)
        print(f'Job Id: {job_id} is not partitioned')

updated_job_ids = [job_id for job_id in job_ids if job_id not in non_partitioned_job_id]
# %%
for job_id in updated_job_ids:
    partition_data_df_activity = job_wise_partitioned_data.get(job_id)['Activity'].unique().tolist()
    print(job_id)
    print(partition_data_df_activity)
#%% ##########function for get activity wise S&V data for given jobs ###############
def mems_SnV_analysis_alljobs(activities, partitioner_data, main_data):
    """---------------------------------------input variables----------------------------------------------
    activities:  ['Wiper Trip']
    partitioner_data :  Dataset after partition e.g., "ctd_partitions_V_0054"
    main data :  non-partition data e.g., "dataset_for_plots"
    Job_id : e.g., 'O.1048592.29-5'
    ----------------------------------------------------------------------------------------------------"""
    vib_list_all  = {}
    shock_list_all = {}
    vib_level_count = {}
    for activity in activities:
        vib_level_count[activity] = []
        vib_list_all[activity] = []
        shock_list_all[activity] = []
        ctd_partitioner_data = []
        ctd_main_dataset = []
        ctd_partitions = partitioner_data
        ctd_partitioner = ctd_partitions[['Start Time','End Time','Activity']]
        ctd_partitioner_data.append(ctd_partitioner)
        main_dataset = main_data
        data_m = main_dataset[['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL']]
        ctd_main_dataset.append(data_m)
        ctd_partitioner_data_df = pd.concat(ctd_partitioner_data,ignore_index = False)
        ctd_main_dataset_df = pd.concat(ctd_main_dataset, ignore_index = False)
        activity_data=ctd_partitioner_data_df[ctd_partitioner_data_df['Activity'] == activity]
        activity_data_df = pd.DataFrame(columns = ['TIME','VIB_LAT','SHK_LAT','FLWI','DEPT','BVEL'])
        ctd_main_dataset_df['TIME'] = pd.to_datetime(ctd_main_dataset_df['TIME'])
        activity_data_append = []
        for r,s in activity_data.iterrows():
            start_time = s['Start Time']
            end_time = s['End Time']
            data_activity = ctd_main_dataset_df[start_time:end_time]
            activity_data_append.append(data_activity)
        if activity_data_append:
            activity_data_df = pd.concat(activity_data_append,ignore_index = False)
        else:
            pass
        activity_data_df[['SHK_LAT']] = activity_data_df[['SHK_LAT']].mask((activity_data_df['SHK_LAT'] > 600.0)|
                                                                           (activity_data_df['SHK_LAT'] < 0))
        activity_data_df[['VIB_LAT']] = activity_data_df[['VIB_LAT']].mask((activity_data_df['VIB_LAT'] > 100.0)|
                                                                           (activity_data_df['VIB_LAT'] < 0))
        activity_data_df = activity_data_df.dropna()
        vib_label = [10, 15, 20]
        for label in vib_label:
            count = activity_data_df[activity_data_df['VIB_LAT']> label]['VIB_LAT'].count()
            vib_level_count[activity].append(count)
        
        vib_level_count[activity].append(activity_data_df['VIB_LAT'].count())
        vib_list_all[activity].append(activity_data_df['VIB_LAT'].values)   
        shock_list_all[activity].append(activity_data_df['SHK_LAT'].values)
    return (vib_list_all, shock_list_all, vib_level_count)
#%% ###########Activity wise S&V box plot for all jobs ###################
activities = ['Drill Run', 'On Bottom Drilling', 'Pull Test', 'Wiper Trip']
activity_dict_vib = {}
activity_dict_shk = {}
for activity in activities:
    activity_dict_vib[activity] = {}
    activity_dict_shk[activity] = {}
    for job_id in updated_job_ids:
        activity_dict_vib[activity][job_id] = []
        activity_dict_shk[activity][job_id] = []
        vib_dict, shk_dict,_ = mems_SnV_analysis_alljobs(activities=[activity], 
                                            partitioner_data=job_wise_partitioned_data.get(job_id),
                                                main_data=job_wise_cleaned_data.get(job_id))
        try:
            activity_dict_shk[activity][job_id].extend(shk_dict[activity])
            activity_dict_vib[activity][job_id].extend(vib_dict[activity])
        except:
            activity_dict_shk[activity][job_id].extend([np.nan])
            activity_dict_vib[activity][job_id].extend([np.nan])

#%%       
for activity in activities:
    print(activity)
    vib_list_data = []
    shk_list_data = []
    for job_id in updated_job_ids:
        try:
            vib_list_data.extend(activity_dict_vib[activity].get(job_id))
            shk_list_data.extend(activity_dict_shk[activity].get(job_id))
        except:
            vib_list_data.extend([np.nan])
            shk_list_data.extend([np.nan])
            print(f'For job_id: {job_id}, does not have activity: {activity}')
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(8,4))
    plt.boxplot(vib_list_data, patch_artist=True, labels =updated_job_ids,
            boxprops=dict(facecolor='blue'),
                flierprops=dict(marker='o', color='red'))
    plt.ylabel('Value [g]')
    plt.grid()
    plt.suptitle(f"Activity: [{activity}] Lat. Vib." ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job Id',fontsize = 15)
    if activity == 'Wiper Trip':
        plt.ylim([0,40])
    else:
        plt.ylim([0,70])
    plt.show()
    plt.figure(figsize=(10,5))
    plt.boxplot(shk_list_data, patch_artist=True, labels =updated_job_ids,
            boxprops=dict(facecolor='blue'),
                flierprops=dict(marker='o', color='red'))
    plt.ylabel('Value [g]')
    plt.grid()
    plt.suptitle(f"Activity: [{activity}] Lat. Shock" ,fontsize = 15)
    plt.tight_layout()
    plt.xticks(rotation=75)
    plt.xlabel('Job Id',fontsize = 15)
    plt.ylim([0,610])
    plt.show()
# %% ############### Activity wise mean SnV for all jobs in a plot ########################### 
from statistics import mean
for activity in activities:
    vib_list_mean = []
    shk_list_mean = []
    for job_id in updated_job_ids:
        if activity_dict_vib[activity][job_id]:
            vib_list = [item for sublist in activity_dict_vib[activity][job_id] for item in sublist.tolist()]
            if not vib_list:
                vib_list_mean.append(np.nan)
            else:
                vib_list_mean.append(mean(vib_list))
        if activity_dict_shk[activity][job_id]:
            shk_list = [item for sublist in activity_dict_shk[activity][job_id] for item in sublist.tolist()]
            if not shk_list:
                shk_list_mean.append(np.nan)
            else:
                shk_list_mean.append(mean(shk_list))
    vib_list_mean = list(round(x,2) for x in vib_list_mean)
    shk_list_mean = list(round(x,2) for x in shk_list_mean)
    plt.rcParams.update({'font.size': 15})
    plt.figure(figsize=(8, 4)) 
    bars = plt.bar(updated_job_ids, shk_list_mean, color=['blue', 'orange', 'green', 'red', 'purple'], edgecolor='black')

    plt.xlabel('Job Id', fontsize=14)  
    plt.ylabel('Mean shock[g]', fontsize=14)  
    plt.title(f'Activity: [{activity}] Mean Lat. shock', fontsize=16)  
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=75)
    plt.ylim([0,610])
    plt.show()
    plt.figure(figsize=(10, 6)) 
    bars = plt.bar(updated_job_ids, vib_list_mean, color=['blue', 'orange', 'green', 'red', 'purple'], edgecolor='black')

    plt.xlabel('Job Id', fontsize=14)  
    plt.ylabel('Mean Vib.[g]', fontsize=14)  
    plt.title(f'Activity: [{activity}] Mean Lat. Vib.', fontsize=16)  
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2, str(height), ha='center', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=75)
    plt.ylim([0,70])
    plt.show()
# %%
