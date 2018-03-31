============================ EE219 Project 3 ==============================
804506544 Jui Chang 
904947071 Wenyang Zhu 
405033965 Xiaohan Wang 
505036001 Yang Tang 


============================ Project Description ===========================
The basic idea of recommendation system is to predict customers interest based on the dataset, which is the feedback of users like or dislike an item.
The basic models for recommender systems works with two kinds of data:
1. User-Item interactions such as ratings
2. Attribute information about the users and items such as textual profiles or relevant keywords
Models use type 1 are attributed as collaborative filtering methods. Models use type 2 are attributed to content based methods. In this Project, we build recommendation system using collaborative filtering methods. The detailed explanation is described in Report.


======================= Environment & Dependencies ========================
Imported libraries:

import pandas as pd
from collections import defaultdict
from surprise.prediction_algorithms.knns import KNNWithMeans
from surprise.prediction_algorithms.matrix_factorization import NMF
from surprise.model_selection.validation import cross_validate
from surprise.model_selection.split import train_test_split
from surprise.dataset import Dataset
from surprise.reader import Reader
from surprise import accuracy
from surprise.model_selection import KFold
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import csv
import os


============================= Folder Content ==============================
- codes
    - 3_MovieLens_dataset_Q1_9.py
    - 4_neighborhood_based_collaborative_filter_Q10_15.py
    - 5_attach_preprocessing.py
    - 5_model_based_collaborative_filter_Q16_29.py
    - 6_naive_collaborative_filter_Q30_33.py
    - 7_performance_comparison_Q34.py
    - 8_ranking_Q35_39.py

- readme.txt
- report.pdf

