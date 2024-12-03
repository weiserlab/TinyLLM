
# This python code is used to merge the data from multiple files into a single csv file 

# The files are Split_Gesture_Data_person ID/Action_Distance_Light Intensity_person ID_trial ID.csv where 
# person ID is the ID of the person performing the gesture
# Action is the gesture performed which can be Double Tap, Hold and Single Tap
# Distance is the distance from the sensor which can be Low, Medium or High
# Light Intensity is the light intensity at the time of the gesture which can be Low or High
# trial ID is the ID of the trial

# Each csv file has a timeseries data which denotes the sensor data in a time interval. The columns are:
# Time (ms), Proximity, Red, Green, Blue

# The data should be merged into a single file with the following columns:
# Action, Distance, Proximity, Red, Green, Blue where Action and Distance are the labels and Proximity, Red, Green, Blue are arrays of the sensor data

# The merged data should be saved in a file named Merged_Gesture_Data.csv

import os
import pandas as pd

# The path to the directory where the files are stored
path = './data'

# The path to the directory where the merged file should be stored
output_path = './'

# The list of files in the directory starting with Split_Gesture_Data
folders = os.listdir(path)
# keep only the folders that start with Split_Gesture_Data
folders = [folder for folder in folders if folder.startswith('Split_Gesture_Data')]

# The list of columns in the merged file
columns = ['input', 'output']
# columns = ['instruction', 'input', 'output']
# columns = ['Action', 'Distance', 'Proximity', 'Red', 'Green', 'Blue']

# The dataframe to store the merged data
merged_data = pd.DataFrame(columns=columns)
instruction = "Sensors readings are used to detect the gestures from their readings. Categorize the given four input sensor readings: proximity, red, green and blue sensor data readings into one of the 3 gestures:\n\nTap\nDouble\nHold\n\n"

# Iterate through the files in the folders
for folder in folders:
    # Check if the file is a directory else continue
    if not os.path.isdir(os.path.join(path, folder)):
        continue
    # Get the list of files in the folder
    files = os.listdir(os.path.join(path, folder))
    for file in files:
        # Read the file
        data = pd.read_csv(os.path.join(path, folder, file))
        # print the file name
        print(file)
        # Get the action, distance from the file name which are the first and second elements of the file name. The file name is split by '_'
        action = file.split('_')[0]
        if action == 'Double Tap':
            action = 'Double'
        elif action == 'Single Tap':
            action = 'Tap'
            
        distance = file.split('_')[1]
        
        # input = str(instruction) + f'Proximity: {data["Proximity"].values.tolist()}\n' + f'Red: {data["Red"].values.tolist()}\n' + f'Green: {data["Green"].values.tolist()}\n' + f'Blue: {data["Blue"].values.tolist()}\n'
        input = f'Proximity: {data["Proximity"].values.tolist()}\n' + f'Red: {data["Red"].values.tolist()}\n' + f'Green: {data["Green"].values.tolist()}\n' + f'Blue: {data["Blue"].values.tolist()}\n'
        input_prompt = f'### Instruction:\n{instruction}\n\n### Input:\n{input}\n### Response:\n'
        # input_prompt = f'Below is an instruction that provides four sensor data values in an array format. Four such arrays are provided. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n### Response:\n'

        new_data = pd.DataFrame({
            # 'instruction': [instruction],
            'input': [input_prompt],
            # 'input': [input],
            'output': [action]
        })
        # Concatenate the new dataframe with the merged data
        merged_data = pd.concat([merged_data, new_data], ignore_index=True)

# split the merged data into training and testing data with 80% of the data as training data and 20% as testing data. The merged data has 3 classes. So the data should be split such that each class has 80% of its data as training data and 20% as testing data
training_data = pd.DataFrame(columns=columns)
validation_data = pd.DataFrame(columns=columns)
testing_data = pd.DataFrame(columns=columns)

# Get the unique classes in the merged data
unique_classes = merged_data['output'].unique()
for unique_class in unique_classes:
    # Get the data for the unique class
    class_data = merged_data[merged_data['output'] == unique_class]
    # Get the number of rows in the class data
    num_rows = class_data.shape[0]
    # Get the number of rows that should be in the training data
    num_training_rows = int(0.7 * num_rows)
    # Get the number of rows that should be in the validation data
    num_validation_rows = int(0.1 * num_rows)
    # Get the number of rows that should be in the testing data
    num_testing_rows = num_rows - num_training_rows - num_validation_rows
    # Get the training data for the class
    training_class_data = class_data.iloc[:num_training_rows, :]
    # Get the validation data for the class
    validation_class_data = class_data.iloc[num_training_rows:num_training_rows+num_validation_rows, :]
    # Get the testing data for the class
    testing_class_data = class_data.iloc[num_training_rows+num_validation_rows:, :]
    # Concatenate the training data with the training class data
    training_data = pd.concat([training_data, training_class_data], ignore_index=True)
    # Concatenate the validation data with the validation class data
    validation_data = pd.concat([validation_data, validation_class_data], ignore_index=True)
    # Concatenate the testing data with the testing class data
    testing_data = pd.concat([testing_data, testing_class_data], ignore_index=True)

# Shuffle the training and testing data
training_data = training_data.sample(frac=1).reset_index(drop=True)
validation_data = validation_data.sample(frac=1).reset_index(drop=True)
testing_data = testing_data.sample(frac=1).reset_index(drop=True)

# save the training and testing data to csv files
training_data.to_csv(os.path.join(output_path, 'training_data.csv'), index=False)
validation_data.to_csv(os.path.join(output_path, 'validation_data.csv'), index=False)
testing_data.to_csv(os.path.join(output_path, 'testing_data.csv'), index=False)



