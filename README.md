## Lunar Crater Detection
A C++ project that detects craters in images of lunar surfaces using Hough Circle Transform form OpenCV. It uses labelled crater information from YoloV5 used for the same purposes. The performance of the processing pipline is measured and reported as True Positives, False Positives, and false Negatives.

## Requirements
- C++ 20 or later
- OpenCV

## How to Run
Open the solution file ```CraterDetection.slnx ``` in Visual Studio.

Set the working directory to the root on the project.

Select Build and then Build Solution to run the project.

Press any key to process and display the next image.

### Image Input
Input images are stored in ```/data``` in the root of the project.