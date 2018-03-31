============================ EE219 Project 5 ==============================
804506544 Jui Chang 
904947071 Wenyang Zhu 
405033965 Xiaohan Wang 
505036001 Yang Tang 


============================ Project Description ===========================
Project Description:
The public discussion attribute of Twitter provides a good platform to perform popularity prediction analysis. In this project, we collected available Twitter data by querying popular hashtags related to the 2015 Super Bowl ranged from 2 weeks before the game to a week after the game. Then we trained a regression model and created a predictor for new samples. The test data consists of tweets containing a hashtag in a specified time window. Next, we used our regression model to predict number of tweets containing the hashtag posted within one hour immediately following the given time window. In the last part, we defined our problem with the knowing data, made analysis and tried to implement our idea. The detailed explanation is described in Report.


======================= Environment & Dependencies ========================
Imported libraries:

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime, time
import pytz
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
import math
from pytz import timezone
import nltk
import calendar
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


============================= Folder Content ==============================
- codes
	- Q1.1.py
	- Q1.2.py
	- Q1.3.py
	- Q1.4.py
	- Q1.5.py
	- Q2.py
	- Q3.py		

- readme.txt
- report.pdf

