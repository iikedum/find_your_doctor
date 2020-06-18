import flask
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

app = flask.Flask(__name__)
app.config["DEBUG"] = True




@app.route('/', methods=['GET'])
def home():
    dataset = pd.read_csv('clustering_test_16.csv')
    dataset.isnull().sum()
    dataset.drop_duplicates(inplace=True)
    X = dataset.iloc[:, [8]].values
    Y = dataset.iloc[:, [9]].values
    CHECK_VAL = []
    k_rng = range(1, 11)
    for kval in k_rng:
        km = KMeans(n_clusters=kval)
        km.fit(X)
        CHECK_VAL.append(km.inertia_)
    wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    #     kmeans.fit(X)
    #     # inertia method returns wcss for that model
    #     wcss.append(kmeans.inertia_)
    clusdict = {}
    princlus = {}
    for n_cluster in range(2, 11):
        kmeans = KMeans(n_clusters=n_cluster).fit(X)
        label = kmeans.labels_
        sil_coeff = silhouette_score(X, label, metric='euclidean')
        print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
        print(type(sil_coeff))
        clusdict[n_cluster] = sil_coeff
    print(clusdict)
    for key, value in clusdict.items():
        if value not in princlus.values():
            princlus[key] = value
    print(princlus)
    Keymax = max(princlus, key=princlus.get)
    print(Keymax)
    kmeans = KMeans(n_clusters=Keymax, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(X)
    print(pred_y)
    plt.scatter(X[pred_y == 0], Y[pred_y == 0], s=100, c='YELLOW', label='Cluster 1')
    plt.scatter(X[pred_y == 1], Y[pred_y == 1], s=100, c='blue', label='Cluster 2')
    plt.scatter(X[pred_y == 2], Y[pred_y == 2], s=100, c='green', label='Cluster 3')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s=300, c=['YELLOW', 'BLUE', 'GREEN'])
    plt.show()
    dataset['cluster'] = pred_y
    front_root_disease_val = "cardio"
    front_age_val = 30
    upper_range = front_age_val + 10
    dict1 = {"cardio": 1, "orthopaedics": 2, "Gynaecology": 3}
    RD_ID_VAL = dict1[front_root_disease_val]
    print("rd_id_val",RD_ID_VAL)
    rslt_df = dataset[(dataset['ROOT_DISEASE_CD'] == RD_ID_VAL) & (dataset['AGE'].between(front_age_val, upper_range))]
    print("rslt_df",rslt_df)
    success = {"STATUS":"SUCCESS"}
    return json.dumps(success)


app.run()
