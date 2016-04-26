# HAR-Pipeline

### Prerequisites

* Python 2.7
  * [Numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
  * [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
  * [Pandas](https://pypi.python.org/pypi/pandas/0.18.0/#downloads)



File descrtiption: 
- training.py. Training both the Convolutional Neural Network and the Viterbi transition matrix. Use data located in DATA/TRAINING. 

- testing.py. Runs the data located in DATA/TESTING through the Convolutional Neural Network and the Viterbi model. Saves the predictions and statistics in the folder RESULTS/

- predicting.py. Predicting unlabelled data. Saves the prediction in RESULTS/RESULT_PREDICTING.py

- cnn.py. Covolutional Neural Network, used both for training and testing. Configuration is done via the TRAINING_VARIABLES.py file. 

- viterbi.py. Viterbi, used both for training and testing.

- data.py. Loads the different data sets. Splits the raw signal into windows used in the network. 

- TRAINING_VARIABLES. All variables used for training and testing. 

### Training and testing folders/files
The data from each subject must be placed in a spesific folder under DATA/TRAINING/ or DATA/TESTING.<br />
Example: DATA/TRAINING/A01

Inside that folder, the three files must contain the name of the sensor ("BACK", "THIGH" and "LAB"). The name must be separated by two underscores.<br />
Example: 
- DATA/TRAINING/A01/01A_Axivity_BACK_Back.csv
- DATA/TRAINING/A01/01A_Axivity_THIGH_Right.csv
- DATA/TRAINING/A01/01A_GoPro_LAB_All.csv

### Predicting folders/files
Same structure as training and testing (explained over), but without annotation file (01A_GoPro_LAB_All.csv).

### Data structure
The training and testing data need to be in a spesific format.  <br />
Sensor format: <br />
```
-1.0156,-0.079657,-0.0015319
-1.0303,-0.079044,-0.016544
-1.0312,-0.09375,-0.03125
-1.0156,-0.078125,-0.03125
-1.015,-0.078125,-0.03125
-1.0147,-0.078125,-0.032169
...
```
Label format:  <br />
```
17
17
6
6
6
6
...
```
### Training
In terminal:
```
python training.py
```
### Testing
In terminal:
```
python testing.py
```
