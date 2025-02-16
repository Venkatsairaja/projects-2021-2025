This repository contains a project focused on detecting car number plates using the YOLO (You Only Look Once) deep learning model. The project leverages YOLO's ability to perform real-time object detection for accurately identifying and localizing vehicle license plates in images or video streams.

Prerequisites

Before running the project, ensure you have the following installed:

Python 3.7 or higher

Required Python libraries:

numpy

pandas

opencv-python

tensorflow or pytorch (depending on the YOLO implementation used)

matplotlib


You can install the required libraries using:

pip install -r requirements.txt

How to Use

Clone the repository:

git clone https://github.com/venkatsairaja/car-number-plate-detection.git

Navigate to the project folder:

cd car-number-plate-detection

Download the pre-trained YOLO weights and configuration files (if not already included). Follow the instructions in the notebook or README to place them in the appropriate directory.

Run the Jupyter Notebook:

jupyter notebook "Car_Number_Plate_Detection.ipynb"

Follow the steps in the notebook to:

Load and preprocess data.

Configure and load the YOLO model.

Perform number plate detection on test images or videos.

Contents

deep learning pro.ipynb: Main notebook demonstrating the number plate detection process.

data/: Directory to store input images or videos.

weights/: Directory for YOLO model weights and configuration files.

output/: Directory to save detection results.

requirements.txt: List of Python dependencies.

Features

Real-time detection of car number plates using YOLO.

Supports image and video inputs.

Customizable detection thresholds and settings.

Easy-to-extend framework for training on custom datasets.

Example

Here is an example of number plate detection using this project:

Input Image:

Detected Output:

Contributing

Contributions are welcome! Please follow these steps:

Fork the repository.

Create a new branch for your feature or bug fix:

git checkout -b feature-name

Commit your changes:

git commit -m "Add feature/fix bug"

Push to your branch:

git push origin feature-name

Create a pull request.

s
Acknowledgments

This project is inspired by the YOLO object detection framework.

Special thanks to the open-source community for providing pre-trained YOLO models and datasets.