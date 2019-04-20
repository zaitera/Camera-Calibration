# Camera-Calibration
>├── README.md (This file)  
├── /Docs  
│ └── Abdullah_Zaiter__Ian_Moura.pdf report in portuguese  
│ └── Latex source code of the report  
├── /src  
| └── main.py (principal source code)  
| └── common.py (aux code with several relevant classes and methods)  
| └──calibrationUtils.py (a class for camera calibration optimizations)  

### OpenCV version : 3.4.1
### Python 3

### Python modules used:
     - cv2  
     - numpy  
     - exit from sys  
     - path from os  
     - deque from collections  
## Some results:
##### A measurement of a 142mm pen on raw image:

<p align="center">
  <img  src="https://github.com/abdullah-zaiter/Camera-Calibration/blob/master/docs/relatorio/Figs/Dado2.png">
</p>

##### A measurement of the  same pen on the undistorted image:

<p align="center">
  <img  src="https://github.com/abdullah-zaiter/Camera-Calibration/blob/master/docs/relatorio/Figs/Dado1.png">
</p>


