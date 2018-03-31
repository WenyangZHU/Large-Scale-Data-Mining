============================ EE219 Project 2 ==============================
904947071 Wenyang Zhu 
804506544 Jui Chang 

============================ Project Description ===========================
Project Description:
Clustering algorithms are a method of unsupervised learning, which is for finding groups of data set with similar feature in a proper space. In this project, we use K- means clustering. We work with “20 Newsgroups” dataset we already used in project 1. The detailed explanation is described in Report.

======================== Environment & Dependencies ========================
Imported Libraries:
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
import string
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
import math
from sklearn.externals import joblib

============================= Folder Content ==============================
- Project2.py
- readme.txt
- report.pdf
