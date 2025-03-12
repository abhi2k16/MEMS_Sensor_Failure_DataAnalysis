# MEMS_Sensor_Failure_DataAnalysis
This program performs data analysis on vibration and shock data from multiple jobs. The analysis includes data cleaning, partitioning, and visualization of the data. 

Reading and Concatenating CSV Files:

The program reads multiple CSV files from a specified directory and concatenates them into a single DataFrame.
It extracts the job ID from the directory path and saves the concatenated data as a new CSV file.
# Data Cleaning and Indexing:

The program reads the concatenated data and sets the TIME column as the index.
It selects columns of interest and converts them to numeric values, handling any errors by coercing them to NaN.
Plotting Whole Run Data:

The program plots the whole run data for vibration (VIB_LAT) and shock (SHK_LAT) values, along with other parameters such as flow rate (FLWI) and depth (DEPT).
Box Plot for Whole Run Data:
It creates box plots for the whole run data, showing the distribution of vibration and shock values.
Partitioning the Data:
The program partitions the data based on activities and sorts the partitions by start time.
It prints the partitioned data for each activity.
Activity-Wise Analysis:
The program performs activity-wise analysis for each job, plotting the vibration and shock values over time.
It creates box plots for each activity, showing the distribution of vibration and shock values.
Mean Shock and Vibration Analysis:

The program calculates the mean shock and vibration values for each activity and plots them in bar charts.
Activity-Wise Analysis for All Jobs Combined:

The program reads data for all jobs, cleans and indexes the data, and partitions it based on activities.
It performs activity-wise analysis for all jobs combined, creating box plots and bar charts for mean shock and vibration values.
Functions for Data Analysis:

The program defines several functions for data analysis, including mems_sensor_shock_analysis and mems_SnV_analysis_alljobs.
These functions perform specific tasks such as partitioning data, calculating vibration and shock counts, and creating visualizations.
Key Functions and Their Purpose
mems_sensor_shock_analysis:

Analyzes sensor shock data for specified activities.
Creates time-domain plots and box plots for vibration and shock values.
mems_SnV_analysis:

Analyzes vibration and shock data for specified activities.
Calculates vibration level counts and creates lists of vibration and shock values.
mems_SnV_analysis_alljobs:

Analyzes vibration and shock data for all jobs combined.
Calculates vibration level counts and creates lists of vibration and shock values for each activity.
