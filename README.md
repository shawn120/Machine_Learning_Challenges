# OSU AI 539 Machine Learning Challenges

Final Project

Shengxuan Wang

wangshe@oregonstate.edu

3/14/2022

## Challenges on Predicting People’s Annual Income

### Data set:

“Adult” from UCI Machine Learning Repository, source: https://archive-beta.ics.uci.edu/ml/datasets/adult Add column names on the original data set from the source:

***adult_data.csv***: The training set;

***adult_test.csv***: The test set. Notice that they are not used as training set or test set in this project, the data will be split again, details can be found in code or report.

### Program:

***Shawn_help.py***: contains some help functions, be depended by the program below, do not need to be run individually.

***baseline.py***: contains some help functions, be depended by the program below, run it can get the baseline.

***method.py***: the main program which is experimenting all the solutions for the challenges, run it can get the main results.

***profile.py***: an unreadable file for dynamically (which means often change codes to get different result) generate the data profile and some temporary test. The whole file is be commented, so do not need to be run or look at, also is not depended on any files. This file is just for archive.

**HOW to run the codes:**

Please keep all the directory structure as original to run these programs, because there are some dependencies with each other. The program is running under python 3, and tested under the specific version 3.9.7.

The project is using these packages, make sure you have installed them, and the latest version of them is recommended:

**a. pandas**

**b. numpy**

**c. matplotlib**

**d. sklearn (scikit-learn)**

Activate python virtual environment in the same directory where all the files are, then use the following command to run this program.

```shell
python3 _filename_.py
```

OR using any IDE/code-runable editor to run the codes is also doable.

### Foder "graphs":

Graphs reflecting the data profile, is used in the project report.