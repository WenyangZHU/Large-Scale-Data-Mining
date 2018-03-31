============================ EE219 Project 4 ==============================
804506544 Jui Chang 
904947071 Wenyang Zhu 
405033965 Xiaohan Wang 
505036001 Yang Tang 

============================ Project Description ===========================
Project Description:
In this project, we made regression analysis on Network Backup Dataset, which contains simulated traffic data on a backup system over a network. Our task is to make prediction of the backup size based on the attributes utilizing regression analysis.
Regression analysis is a statistical process in data mining for estimating the relationships among variables. In this project, we studied and used four prediction models, including Linear Regression Model, Random Forest, Neural Network Regression Model and Polynomial Regression Model, to predict the backup size and used cross-validation to evaluate the performances. The detailed explanation is described in Report.


======================= Environment & Dependencies ========================
Imported libraries:

import pandas as pd
import numpy as np
import re
import math
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.linear_model import ElasticNet
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor


============================= Folder Content ==============================
- codes
	- Q1_load_dataset.py
	- Q2_a_i_scalar_encoding.py
	- Q2_a_ii_data_preprocessing.py
	- Q2_a_iii_feature_selection.py
	- Q2_a_iv_feature_encoding.py
	- Q2_a_v_control_illconditioning_and_overfitting.py
	- Q2_b_random_forest.py
	- Q2_c_neural_network_regression.py
	- Q2_d_i_linear_regression_model.py
	- Q2_d_ii_polynomial_function_model.py
	- Q2_e_knn.py

- readme.txt
- report.pdf

