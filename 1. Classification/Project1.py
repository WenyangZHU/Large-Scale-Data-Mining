from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer, ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import string
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.svm import SVC
from sklearn.metrics import auc, roc_curve, confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression


categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
graphics_train = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state = 42)
graphics_test = fetch_20newsgroups(subset='test', categories = categories, shuffle = True, random_state = 42)
print('********Loading Datasets with 8 Categories Successfully********')

'''Part a) Histogram Plot of number of training documents per class'''
plt.figure()
plt.hist(graphics_train.target, bins=[0,1,2,3,4,5,6,7,8], align='left', rwidth=0.6)
plt.xticks(np.arange(len(categories)), graphics_train.target_names, rotation = "vertical")
plt.subplots_adjust(bottom=0.2, right=0.8, top=0.9)
plt.title('Histogram')
print('********Histogram Plot of number of training documents per class has been shown.********')

'''Part b) tokenize and create TFxIDF vector representation'''
# a callable function for TfidfVectorizer, cutting off punctuations and prune out words with same stem
class Token(object):
    def __init__(self):
        self.doc = PorterStemmer()
    def __call__(self, doc):
        return [self.doc.stem(w) for w in word_tokenize(doc.translate(str.maketrans('', '', string.punctuation))) if not w.isdigit()]

def tfidf_transform(min_df):
    tfidf_transformer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, min_df=min_df, tokenizer=Token())
    tfidf_train = tfidf_transformer.fit_transform(graphics_train.data)  # TFxIDF Vector Representation for min_df = 2
    tfidf_test = tfidf_transformer.transform(graphics_test.data)
    print('TFxIDF vector representation has size: ' + str(tfidf_train.shape))
    print('Therefore, final number of terms is ' + str(tfidf_train.shape[1]))
    return tfidf_train, tfidf_test

#min_df = 2
print('********Results for min_df=2********')
tfidf1 = tfidf_transform(2)

#min_df = 5
print('********Results for min_df=5********')
tfidf2 = tfidf_transform(5)

'''Part c) '''
categories2 = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
graphics_train2 = fetch_20newsgroups(subset='train', shuffle = True, random_state = 42)
print('********Loading Datasets with 20 Categories Successfully********')

cv_transformer2 = CountVectorizer(stop_words=ENGLISH_STOP_WORDS, tokenizer=Token())
tfd2 = cv_transformer2.fit_transform(graphics_train2.data)
tfc2 = np.zeros(shape=(20, tfd2.shape[1]))
for i in range(tfd2.shape[0]):
    tfc2[graphics_train2.target[i]] += tfd2[i]
tficf_transformer2 = TfidfTransformer()
tficf2 = tficf_transformer2.fit_transform(tfc2)
print('TFxICF vector representation has size: ' + str(tficf2.shape))
print('Therefore, final number of terms is ' + str(tficf2.shape[1]))

print('********10 Most Significant Terms for 4 Classes********')

for i in range(tficf2.shape[0]):
    if graphics_train2.target_names[i] in categories2:
        print(graphics_train2.target_names[i], ':')
        c = tficf2[i]
        s = zip(c.indices, c.data)
        sorted_s = sorted(s, key=lambda x:x[1], reverse=True)
        for j in range(10):
            print(cv_transformer2.get_feature_names()[sorted_s[j][0]], '{:.3f}'.format(sorted_s[j][1]), end=' ')
        print('\n')

'''Part d) Reduce dimensionality with NMF and LSI'''
#Reduce dimentionality through NMF and get facorization matrix w1
print('********Reducing dimentionality with NMF********')
nmf = NMF(n_components=50, init='random', random_state=0)

#Reduce dimentinality through LSI and get truncated approximation singular vector matrix w2
print('********Reducing dimensionality with LSI********')
lsi = TruncatedSVD(n_components=50)

train_label = [_//4 for _ in graphics_train.target]
test_label = [_//4 for _ in graphics_test.target]
class_name = ['Computer Technology', 'Recreational Activity']

def roc(tfidf_name, test_label, predict_proba, model_names, title, roc_draw=True):
    fpr, tpr, thresholds = roc_curve(test_label, predict_proba)
    area = auc(fpr, tpr)
    lw = 2
    if roc_draw:
        plt.figure()
        plt.plot(fpr, tpr, color='red', lw=lw, label='ROC curve (area = %0.2f)' % area)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title(tfidf_name+', ROC curve for '+model_names[0]+' + '+model_names[1]+' '+title, fontsize=20)
        plt.legend(loc="lower right", fontsize=20)

def calc(tfidf_name, test_label, predict, model_names, title, cm_draw=True, type='', class_name=class_name):
    cm = confusion_matrix(test_label, predict)  # parameters for confusion matrix
    if cm_draw:
        l = np.arange(len(class_name))
        plt.figure()
        plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9)
        plt.imshow(cm, cmap=plt.cm.Blues_r)
        plt.xlabel('Predicted Class', fontsize=20)
        plt.ylabel('Actual Class', fontsize=20)
        plt.xticks(l, class_name, rotation=45, fontsize=20)
        plt.yticks(l, class_name, fontsize=20)
        xs, ys = np.meshgrid(l, l)
        for x, y in zip(xs.flatten(), ys.flatten()):
            plt.text(x, y, '{}'.format(cm[y][x]), color='red', fontsize=60, va='center', ha='center')
        plt.title(tfidf_name+', Confusion matrix for '+model_names[0]+' + '+model_names[1]+' '+title, fontsize=20)
        plt.colorbar()
    accuracy = accuracy_score(test_label, predict)
    if type == 'macro':
        recall = recall_score(test_label, predict, average='macro')
        precision = precision_score(test_label, predict, average='macro')
    elif type == 'weighted':
        recall = recall_score(test_label, predict, average='weighted')
        precision = precision_score(test_label, predict, average='weighted')
    else:
        recall = recall_score(test_label, predict)
        precision = precision_score(test_label, predict)
    print(model_names[1]+' '+title+' :\n', 'Confusion Matrix\n', cm, '\nAccuracy =', accuracy, 'Recall =',
          recall, 'Precision =', precision)

def train(tfidf_name, model, model_names, w, w_test, train_label, test_label, title, roc_draw=True, cm_draw=True):
    model.fit(w, train_label)
    predict_proba = model.predict_proba(w_test)[:, 1]
    predict = model.predict(w_test)
    roc(tfidf_name, test_label, predict_proba, model_names, title, roc_draw)
    calc(tfidf_name, test_label, predict, model_names, title, cm_draw)

def e2i(tfidf, tfidf_name, model, model_name, train_label, test_label):
    tfidf_train, tfidf_test = tfidf
    w = model.fit_transform(tfidf_train)
    w_test = model.transform(tfidf_test)

    '''Part e) SVC Classifier'''
    # set gamma = 1000 to separate documents into 2 groups
    gamma = 1000
    print('********SVC Classifier with Gamma = 1000, Data from '+model_name+'********')
    svc = SVC(kernel='linear', C = gamma, probability = True)
    train(tfidf_name, svc, [model_name, 'SVC'], w, w_test, train_label, test_label, '(Gamma=1000)')

    # set gamma = 0.001 to redo the work
    print('********SVC Classifier with Gamma = 0.001, Data from '+model_name+'********')
    gamma = 0.001
    svc = SVC(kernel='linear', C = gamma, probability = True)
    train(tfidf_name, svc, [model_name, 'SVC'], w, w_test, train_label, test_label, '(Gamma=0.001)')

    '''Part f) gamma'''
    print('********Choose Best Gamma for SVC Classifier, Data from '+model_name+'********')
    gammas = [10**k for k in range(-3, 4)]
    svcs, scores = [], []
    scoring = ['accuracy', 'recall', 'precision']
    for g in gammas:
        svcs.append(SVC(kernel='sigmoid', gamma = g))
        scores.append(cross_validate(svcs[-1], w, train_label, scoring = scoring, cv = 5, return_train_score = False))
        print('Gamma = {}, Accuracy = {:.3f}, Recall = {:.3f}, Precision = {:.3f}'.format(g, scores[-1]['test_accuracy'].mean(), scores[-1]['test_recall'].mean(), scores[-1]['test_precision'].mean()))
    #best gamma
    if model_name == 'LSI': gamma = 1
    else: gamma = 10
    svc = SVC(kernel='linear', C=gamma, probability=True)
    train(tfidf_name, svc, [model_name, 'SVC'], w, w_test, train_label, test_label, '(Gamma='+str(gamma)+')')

    '''Part g) Naive Bayes'''
    if model_name != 'LSI':
        print('********Naive Bayes Algorithm, Data from '+model_name+'********')
        nb = MultinomialNB()
        train(tfidf_name, nb, [model_name, 'Naive Bayes'], w, w_test, train_label, test_label, '')

    '''Part h) Logistic Regression'''
    print('********Logistic Regression, Data from '+model_name+'********')
    lr = LogisticRegression(penalty = 'l2', C = 10**40) #default 'l2'
    train(tfidf_name, lr, [model_name, 'Logistic Regression'], w, w_test, train_label, test_label, '')

    '''Part i) Regularization'''
    Cs = [10**k for k in range(-3, 4)]

    print('********Logistic Regression with L1 Regularization, Data from '+model_name+'********')
    # l1
    for C in Cs:
        lr2 = LogisticRegression(penalty = 'l1', C = C)
        train(tfidf_name, lr2, [model_name, 'Logistic Regression'], w, w_test, train_label, test_label, '(L1 Regularization)', roc_draw=False, cm_draw=False)
        score2 = lr2.score(w_test, test_label)
        distance2 = [abs(d) for d in lr2.decision_function(w_test)]
        print('L1 :', 'C =', C, 'Mean Accuracy = ', score2, 'Min Distance', min(distance2), 'Max Distance', max(distance2), 'Mean Distance', sum(distance2) / len(distance2))

    # l2
    print('********Logistic Regression with L2 Regularization, Data from '+model_name+'********')
    for C in Cs:
        lr3 = LogisticRegression(penalty = 'l2', C = C)
        train(tfidf_name, lr3, [model_name, 'Logistic Regression'], w, w_test, train_label, test_label, '(L2 Regularization)', roc_draw=False, cm_draw=False)
        score3 = lr3.score(w_test, test_label)
        distance3 = [abs(d) for d in lr3.decision_function(w_test)]
        print('L2 :', 'C =', C, 'Mean Accuracy = ', score3, 'Min Distance', min(distance3), 'Max Distance', max(distance3), 'Mean Distance', sum(distance3) / len(distance3))

'''Part e)-i)'''
print('mindf=2, NMF')
e2i(tfidf1, 'Min_df=2', nmf, 'NMF', train_label, test_label)
print('mindf=2, LSI')
e2i(tfidf1, 'Min_df=2', lsi, 'LSI', train_label, test_label)
print('mindf=5, LSI')
e2i(tfidf2, 'Min_df=5', lsi, 'LSI', train_label, test_label)

'''Part j) '''
graphics_train3 = fetch_20newsgroups(subset='train', categories = categories2, shuffle = True, random_state = 42)
graphics_test3 = fetch_20newsgroups(subset='test', categories = categories2, shuffle = True, random_state = 42)
print('********Loading Datasets with 4 Categories Successfully********')

tfidf_transformer3 = TfidfVectorizer(stop_words='english', min_df=2, tokenizer=Token())
tfidf3 = tfidf_transformer3.fit_transform(graphics_train3.data) #TFxIDF Vector Representation for min_df = 2

nmf3 = NMF(n_components=50, init='random', random_state=0)
w3 = nmf3.fit_transform(tfidf3)
tfidf_test3 = tfidf_transformer3.transform(graphics_test3.data)
w3_test = nmf3.transform(tfidf_test3)

lsi4 = TruncatedSVD(n_components=50)
w4 = lsi4.fit_transform(tfidf3)
w4_test = lsi4.transform(tfidf_test3)

train_label2 = graphics_train3.target
test_label2 = graphics_test3.target

def j(w, w_test, train_label, test_label, model_name, class_name, lsi=False):
    if not lsi:
        print('********Naive Bayes Algorithm for Multiclassifcation, Data from '+model_name+', Min_df=2********')
        nb2 = MultinomialNB()
        nb2.fit(w, train_label)
        predictnb = nb2.predict(w_test)
        calc('Min_df=2', test_label, predictnb, [model_name, 'Naive Bayes'], '', type='macro', class_name=class_name)

    # SVC One vs One
    print('********SVC, Data from '+model_name+', Min_df=2, One Vs. One********')
    svcovo = SVC(kernel='linear', C=100, decision_function_shape = 'ovo', probability = True)
    svcovo.fit(w, train_label)
    predictovo = svcovo.predict(w_test)
    calc('Min_df=2', test_label, predictovo, [model_name, 'SVC One Vs. One'], '', type='weighted', class_name=class_name)

    # SVC One vs All
    print('********SVC, Data from '+model_name+', Min_df=2, One Vs. All********')
    svcova = SVC(kernel='linear', C=100, decision_function_shape = 'ovr', class_weight = 'balanced')
    svcova.fit(w, train_label)
    predictova = svcova.predict(w_test)
    calc('Min_df=2', test_label, predictova, [model_name, 'SVC One Vs. All'], '', type='weighted', class_name=class_name)

j(w3, w3_test, train_label2, test_label2, 'NMF', categories2)
j(w4, w4_test, train_label2, test_label2, 'LSI', categories2, lsi=True)

plt.show()
