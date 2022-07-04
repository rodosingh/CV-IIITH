[![Generic badge](https://img.shields.io/badge/CV-Assignment:1-BLUE.svg)](https://shields.io/)
[![Generic badge](https://img.shields.io/badge/DUE-23:00hrs,30/01/2022-RED.svg)](https://shields.io/)
# Assignment-01
The goal of the assignment is to familiarize you to the process of camera calibration
and the critical role it plays in using any measurements of the world from images.

Please raise doubts on the appropriate assignment thread on Moodle.

# Instructions
- Follow the directory structure as shown below: 
  ```
  ├── src           
        ├── Assignment01.ipynb
  ├── images //your images
  ├── data  //provided data
  └── README.md
  ```
- `src` will contain the Jupyter notebook(s) used for the assignment.
- `images` will contain images used for the questions.
- `data` contains images provided to you already, for solving the questions. 
- Follow this directory structure for all following assignments in this course.
- **Make sure you run your Jupyter notebook before committing, to save all outputs.**

## Dataset
- `black-dots.JPG` : for DLT and RANSAC based estimation.
- `measurement.JPG` : Measurements are shown according to scale(in mms). 
		  World co-ordinate of every point can be calculated using the unit measurements given from the origin. 

- `checkerboard-[01-15].JPG` : for Zhang algorithm based estimation
			    Size of each square on checkerboard 29mmX29mm.

## Helper code
Function to get Rotation matrix from Euler angles :
```
def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,0,0],[0,math.cos(theta[0]),-math.sin(theta[0])],[0,math.sin(theta[0]), math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]),0,math.sin(theta[1])],[0,1,0],[-math.sin(theta[1]),0,math.cos(theta[1])]])             
    R_z = np.array([[math.cos(theta[2]),-math.sin(theta[2]),0],[math.sin(theta[2]),math.cos(theta[2]),0],[0,0,1]])
    R = np.dot(R_z,np.dot(R_y,R_x))
    return R
```

