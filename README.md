# Taxi-Track
# Sequence Classification with Deep Neural Networks

# Project Overview
This project implements a sequence classification task using deep learning to predict which taxi driver a given trajectory belongs to. The model is trained on a dataset containing daily driving trajectories of five taxi drivers over a 6-month period.

# Dataset
The dataset consists of taxi drivers' GPS records with the following features:

Plate (anonymized as 0-5)
Longitude
Latitude
Timestamp
Status (1 for occupied, 0 for vacant)

The original dataset can be found here.

# Project Structure

data/: https://drive.google.com/drive/folders/1ZnoPT7Y69_63hGFJJUWhQitqvnR8zTfg?usp=drive_link
evaluation.py: Script for processing data and running predictions
model.py: Implementation of the deep learning model
train.py: Script for training the model
test.pkl: Sample testing data

# Methodology
Data Loading: Each CSV file containing daily trajectories for 5 drivers was loaded and combined into a single dataset.
Temporal Sorting: GPS records were sorted based on timestamp to ensure chronological order within each driver's trajectory.
Trajectory Segmentation: Full-day trajectories were segmented into smaller sub-trajectories based on the 'status' field, separating occupied and vacant periods.
Time Feature Extraction: Timestamp information was converted into numerical features:

Hour of day (0-23)
Day of week (0-6)
Is_weekend (binary)

# Results
After training and evaluating our model, we achieved the following results:

Predicted Driver: 2
Model Accuracy: 83%

This high accuracy demonstrates that our model is capable of effectively distinguishing between different taxi drivers based on their GPS trajectories. The model successfully identified Driver 2 in the test dataset, showcasing its ability to capture and recognize individual driving patterns.
Our achieved accuracy of 83% significantly exceeds the project's baseline requirements:



# Clone the repository:
git clone https://github.com/vshaladhav97/Taxi-Track.git


# Evaluate the model:
python evaluation.py


# Dependencies
Python 3.x
Keras or PyTorch


# Acknowledgments
This project was completed as part of a course assignment. The original project description and dataset were provided by the course instructors.

