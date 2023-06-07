import pandas as pd
import numpy as np
import random

# Read train data
train_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\train_data_v2.csv')

# Iterate over each row in train data
for index, row in train_data.iterrows():
    section_start_str = row['song.sections_start']
    section_start = np.array(eval(section_start_str))  # Parse string to array
    current_point = row['current_point']

    # Calculate time difference
    time_diff = current_point - section_start

    # Generate random values between section start points
    random_values = [random.uniform(section_start[i], section_start[i+1]) for i in range(len(section_start)-1)]

    # Check time difference and assign label
    if np.abs(time_diff - random_values) <= 1:
        train_data.at[index, 'section.scale_change'] = 1  # Scale change
    else:
        train_data.at[index, 'section.scale_change'] = 0  # No scale change

# Read test data
test_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\test_data_v2.csv')


# Iterate over each row in test data
for index, row in test_data.iterrows():
    section_start_str = row['song.sections_start']
    section_start = np.array(eval(section_start_str))  # Parse string to array
    current_point = row['current_point']

    # Calculate time difference
    time_diff = current_point - section_start

    # Generate random values between section start points
    random_values = [random.uniform(section_start[i], section_start[i+1]) for i in range(len(section_start)-1)]

    # Check time difference and assign label
    if np.abs(time_diff - random_values) <= 1:
        test_data.at[index, 'section.scale_change'] = 1  # Scale change
    else:
        test_data.at[index, 'section.scale_change'] = 0  # No scale change

# Save the modified train and test data
train_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_train_data.csv', index=False)
test_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_test_data.csv', index=False)

