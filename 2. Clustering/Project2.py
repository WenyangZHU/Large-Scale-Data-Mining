from sklearn.datasets import fetch_20newsgroups
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from nltk import word_tokenize
import string
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score, adjusted_rand_score, adjusted_mutual_info_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer, normalize, StandardScaler
import math
from sklearn.externals import joblib


class Token(object):
    def __call__(self, doc):
        return [w for w in word_tokenize(doc.translate(str.maketrans('', '', string.punctuation))) if not w.isdigit()]

def tfidf_transform(min_df, graphics):
    tfidf_transformer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS, min_df=min_df, tokenizer=Token())
    tfidf = tfidf_transformer.fit_transform(graphics.data)  # TFxIDF Vector Representation for min_df = 3
    normalizer = Normalizer()
    tfidf = normalizer.fit_transform(tfidf)
    print('TFxIDF vector representation has size: ' + str(tfidf.shape))
    print('Therefore, final number of terms is ' + str(tfidf.shape[1]))
    return tfidf

def kmeans(n_clusters, tfidf, train_label, title, class_name):
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
    km.fit(tfidf)
    #(a)
    calc(train_label, km.labels_, title, class_name)
    #(b)
    hs = homogeneity_score(train_label, km.labels_)
    cs = completeness_score(train_label, km.labels_)
    vms = v_measure_score(train_label, km.labels_)
    ars = adjusted_rand_score(train_label, km.labels_)
    mis = adjusted_mutual_info_score(train_label, km.labels_)
    print("Homogeneity: %0.3f" % hs)
    print("Completeness: %0.3f" % cs)
    print("V-measure: %0.3f" % vms)
    print("Adjusted Rand-Index: %.3f"% ars)
    print("Mutual Info Score: %.3f" % mis)
    print()
    return km, hs, cs, vms, ars, mis

def calc(test_label, predict, title, class_name, cm_draw=True):
    cm = confusion_matrix(test_label, predict)  # parameters for confuion matrix
    if cm_draw:
        if class_name: l = np.arange(len(class_name))
        else: l = np.arange(20)
        plt.figure()
        plt.subplots_adjust(bottom=0.3, right=0.8, top=0.9)
        plt.imshow(cm, cmap=plt.cm.Blues_r)
        plt.xlabel('Predicted Class', fontsize=20)
        plt.ylabel('Actual Class', fontsize=20)
        if class_name:
            plt.xticks(l, ['Cluster '+str(i+1) for i in l], rotation=45, fontsize=20)
            plt.yticks(l, class_name, fontsize=20)
            xs, ys = np.meshgrid(l, l)
            for x, y in zip(xs.flatten(), ys.flatten()):
                plt.text(x, y, '{}'.format(cm[y][x]), color='red', fontsize=60, va='center', ha='center')
        else:
            plt.xticks(l, ['C'+str(i+1) for i in l])
            plt.yticks(l, ['C'+str(i+1) for i in l])
        plt.title('Contingency matrix of Kmeans ' + title, fontsize=20)
        plt.colorbar()
    print('Kmeans ' + title + ' :\n', 'Confusion Matrix\n', cm)

def draw(n_clusters, data, train_label, title, class_name):
    kmodel = kmeans(n_clusters, data, train_label, title, class_name)
    km = kmodel[0]
    data_reduced = km.transform(data)

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_reduced)
    x, y = data_2d.transpose()
    plt.figure()
    plt.scatter(x, y, s=2, c=km.labels_, alpha=0.5)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title('Visualization of ' + title, fontsize=20)
    plt.figure()
    plt.scatter(x, y, s=2, c=train_label, alpha=0.5)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('y', fontsize=20)
    plt.title('Visualization of data with ground truth ' + title, fontsize=20)

def log(data):
    new_data = np.zeros(data.shape)
    for i in range(len(data)):
        for j in range(len(data[0])):
            new_data[i][j] = math.log(data[i][j]+1)
    return new_data

def draw_metrics(metrics, rs):
    name = ['Homogeneity Score', 'Completeness Score', 'V-measure', 'Adjusted Rand Score', 'Adjusted Mutual Info Score']
    for index, metric in enumerate(metrics):
        plt.figure()
        plt.plot(rs, metric)
        plt.xlabel('r', fontsize=20)
        plt.ylabel(name[index], fontsize=20)
        plt.title(name[index]+' vs. r', fontsize=20)

#Loading dataset
class_name = ['Computer Technology', 'Recreational Activity']
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
dataset1 = fetch_20newsgroups(subset='train', categories = categories, shuffle = True, random_state = 42)
print('********Loading Datasets with 8 Categories Successfully********')


#Secton one to four
def o2f(dataset, n_clusters, class_name):
    #Building TFxIDF
    tfidf = tfidf_transform(3, dataset)
    if n_clusters == 2:
        train_label = [_ // 4 for _ in dataset.target]
    else:
        train_label = dataset.target
    print('********Building TFxIDF representation Successfully********')

    # K-means
    kmeans(n_clusters, tfidf, train_label, '', class_name)
    print('********Kmeans Successfully********')

    #Preprocessing the data
    #(a)
    r = 1000
    lsi = TruncatedSVD(n_components=r, random_state=42)
    lsi.fit(tfidf)
    ys = [0] * r
    xs = range(1, r + 1)
    ratios = lsi.explained_variance_ratio_
    ratio_sum = 0
    for i in range(r):
        ratio_sum += ratios[i]
        ys[i] = ratio_sum
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel('r', fontsize=20)
    plt.ylabel('Variance Ratio', fontsize=20)
    plt.title('The percent of variance the top r principle components can retain vs. r', fontsize=20)
    print('********Preprocessing dataset (a) Successfully********')

    # (b)
    rs = [1,2,3,5,10,20,50,100,300]
    metrics = [[] for _ in range(5)]
    for r in rs:
        lsi = TruncatedSVD(n_components=r, random_state=42)
        lsi.fit(tfidf)
        joblib.dump(lsi, 'models/20LSI+' + str(r) + '.pkl')
        lsi = joblib.load('models/20LSI+' + str(r) + '.pkl')
        lsi_data = lsi.transform(tfidf)
        kmodel = kmeans(n_clusters, lsi_data, train_label, 'LSI r='+str(r), class_name)
        for index, metric in enumerate(kmodel[1:]):
            metrics[index].append(metric)
    draw_metrics(metrics, rs)
    metrics = [[] for _ in range(5)]
    for r in rs:
        nmf = NMF(n_components=r, init='random', random_state=42)
        nmf.fit(tfidf)
        joblib.dump(nmf, 'models/20NMF+' + str(r) + '.pkl')
        nmf = joblib.load('models/20NMF+' + str(r) + '.pkl')
        nmf_data = nmf.transform(tfidf)
        kmodel = kmeans(n_clusters, nmf_data, train_label, 'NMF r='+str(r), class_name)
        for index, metric in enumerate(kmodel[1:]):
            metrics[index].append(metric)
    draw_metrics(metrics, rs)
    print('********Preprocessing dataset (b) Successfully********')

    # Visualization
    best_lsi_r, best_nmf_r = 20, 20 #3, 2 for 2 clusters
    lsi = joblib.load('models3/20LSI+' + str(best_lsi_r) + '.pkl')
    lsi_data = lsi.fit_transform(tfidf)
    nmf = joblib.load('models3/20NMF+' + str(best_nmf_r) + '.pkl')
    nmf_data = nmf.fit_transform(tfidf)

    # (a)
    draw(n_clusters, lsi_data, train_label, 'LSI, best r='+str(best_lsi_r), class_name)
    draw(n_clusters, nmf_data, train_label, 'NMF, best r='+str(best_nmf_r), class_name)
    print('********Visualizing dataset (a) Successfully********')

    #(b) Methods are used after LSI/NMF
    lsi_data1 = StandardScaler(with_mean=False).fit_transform(lsi_data)
    nmf_data1 = StandardScaler(with_mean=False).fit_transform(nmf_data)
    draw(n_clusters, lsi_data1, train_label, 'LSI, best r='+str(best_lsi_r)+' normalized', class_name)
    draw(n_clusters, nmf_data1, train_label, 'NMF, best r='+str(best_nmf_r)+' normalized', class_name)

    nmf_data2 = log(nmf_data)
    draw(n_clusters, nmf_data2, train_label, 'NMF, best r='+str(best_nmf_r)+' logarithm transformed', class_name)

    nmf_data3 = log(nmf_data1)
    nmf_data4 = StandardScaler(with_mean=False).fit_transform(nmf_data2)
    draw(n_clusters, nmf_data3, train_label, 'NMF, best r='+str(best_nmf_r)+' normalized -> logarithm transformed', class_name)
    draw(n_clusters, nmf_data4, train_label, 'NMF, best r='+str(best_nmf_r)+' logarithm transformed -> normalized', class_name)
    print('********Visualizing dataset (b) Successfully********')

o2f(dataset1, 2, class_name)

#Expanding dataset into 20 categories
dataset2 = fetch_20newsgroups(subset='train', shuffle = True, random_state = 42)
print('********Loading Datasets with 20 Categories Successfully********')

o2f(dataset2, 20, [])

plt.show()