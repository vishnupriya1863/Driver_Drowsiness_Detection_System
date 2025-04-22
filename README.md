# Driver-Drowsiness-Detection
The Driver Drowsiness Detection system uses dlib for facial landmark detection and eye tracking to calculate the Eye Aspect Ratio (EAR), which monitors driver fatigue. imutils simplifies video frame processing, while scipy helps compute the Euclidean distance between eye landmarks to assess drowsiness. The system alerts the driver when signs of fatigue are detected, improving road safety by preventing accidents due to drowsy driving.

**Prerequisites**

1.Python Version: Ensure that you are using Python 3.x for compatibility.

2.Facial Landmarks File: You need a facial landmarks file, which can be downloaded here. Extract the file to your working directory for easy access.

3.Libraries: The following libraries are required for the project:

4.dlib for facial landmark detection.

5.imutils for video frame manipulation.

6.scipy for mathematical calculations like Euclidean distance.

7.opencv-python for image and video processing.

# Ensure these dependencies are installed via the requirements.txt file.

**dataset download link **

 https://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2/download


# Execution
To run the system, follow these steps:

#### Install the required libraries by running the command:

python -m pip install -r requirements.txt

python -m pip install opencv-python 

#### Run the main Python file using the command:


python Driver_Drowsiness_Detection.py
* pip install scipy
* pip install imutils
* pip install cmake / install it from 
* pip install dlib

if the camera seems to be not working then :pip install --upgrade opencv-python opencv-contrib-python
The system will begin capturing video from your webcam, detect facial landmarks, and alert you if signs of drowsiness are detected.

**Troubleshooting**

Webcam Issues: If you encounter issues with video capture, ensure that your webcam drivers are properly installed and recognized by OpenCV.
dlib Installation Issues: If dlib installation fails, make sure you have the necessary C++ build tools. You can find installation instructions here.
Notes
The **shape_predictor_68_face_landmarks.dat file** is crucial for accurate facial landmark detection. Ensure that it is extracted and available in the working directory or specify its path in the script.
