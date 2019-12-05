'''
    Author: Adam Jilling
    Plug in a clustering algorithm, runs and returns 10 quality metrics
    that can be averaged to analyze which algorithms performed best, 
    returns average run-time (10 total) of each algorithm too
'''

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from time import time
import numpy as np
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import pandas

aa_results = []
ca_results = []
gmd_results = []
gmf_results = []
km_results = []
mk_results = []
wa_results = []

def run_experiments(directory):

    ro.r.assign('directory', directory)
    ro.r('fileNames = list.files(directory, pattern = ".csv")')
    fileNames = ro.r('fileNames')

    result_title = ["name", "avg runtime", "calinski", "silhouette", "dunn", "pearson", "tau", "davies", "xie", "sds", "sdd", "ray"]
    aa_results.append(result_title)
    ca_results.append(result_title)
    gmd_results.append(result_title)
    gmf_results.append(result_title)
    km_results.append(result_title)
    mk_results.append(result_title)
    wa_results.append(result_title)

    for name in fileNames:
        ro.r.assign('name', name)
        ro.r('mf = read.csv(file = paste0(directory, "/", name), header = FALSE, sep = ",")')
        array = np.array(ro.r('matrix = as.matrix(mf)'))
        print("****** DATASET: ", name, "******")

        # normalize dataset
        array_norm = np.interp(array, (array.min(), array.max()), (0, +1))
        num_classes = np.shape(array_norm)[1]

        # Average Agglomerative (AA)
        print("Algorithm: AA")
        # do 10 runs and take average
        aa_time_total = 0.0
        for x in range(10):
            t0_aa = time()
            aa = AgglomerativeClustering(n_clusters = num_classes, affinity = "euclidean", linkage = "average").fit(array_norm)
            aa_time = time() - t0_aa
            aa_time_total += aa_time
        aa_avg = aa_time_total / 10
        print("Average Runtime: ", aa_avg)
        labels = aa.labels_
        aa_entry = get_metrics(array_norm, labels)
        aa_entry.insert(0, aa_avg)
        aa_entry.insert(0, name)
        aa_results.append(aa_entry)

        # Complete Agglomerative (CA)
        print("Algorithm: CA")
        # do 10 runs and take average
        ca_time_total = 0.0
        for x in range(10):
            t0_ca = time()
            ca = AgglomerativeClustering(n_clusters = num_classes, affinity = "euclidean", linkage = "complete").fit(array_norm)
            ca_time = time() - t0_ca
            ca_time_total += ca_time
        ca_avg = ca_time_total / 10
        print("Average Runtime: ", ca_avg)
        labels = ca.labels_
        ca_entry = get_metrics(array_norm, labels)
        ca_entry.insert(0, ca_avg)
        ca_entry.insert(0, name)
        ca_results.append(ca_entry)

        # Gaussian Mixture Diagonal (GMD)
        print("Algorithm: GMD")
        # do 10 runs and take average
        gmd_time_total = 0.0
        for x in range(10):
            t0_gmd = time()
            gmd = GaussianMixture(n_components = num_classes, covariance_type = "diag").fit_predict(array_norm)
            gmd_time = time() - t0_gmd
            gmd_time_total += gmd_time
        gmd_avg = gmd_time_total / 10
        print("Average Runtime: ", gmd_avg)
        labels = gmd
        gmd_entry = get_metrics(array_norm, labels)
        gmd_entry.insert(0, gmd_avg)
        gmd_entry.insert(0, name)
        gmd_results.append(gmd_entry)

        # Gaussian Mixture Full (GMF)
        print("Algorithm: GMF")
        # do 10 runs and take average
        gmf_time_total = 0.0
        for x in range(10):
            t0_gmf = time()
            gmf = GaussianMixture(n_components = num_classes, covariance_type = "full").fit_predict(array_norm)
            gmf_time = time() - t0_gmf
            gmf_time_total += gmf_time
        gmf_avg = gmf_time_total / 10
        print("Average Runtime: ", gmf_avg)
        labels = gmf
        gmf_entry = get_metrics(array_norm, labels)
        gmf_entry.insert(0, gmf_avg)
        gmf_entry.insert(0, name)
        gmf_results.append(gmf_entry)

        # K-Means(KM)
        print("Algorithm: KM")
        # do 10 runs and take average
        km_time_total = 0.0
        for x in range(10):
            t0_km = time()
            km = KMeans(init='k-means++', n_clusters=num_classes, n_init=10).fit(array_norm)
            km_time = time() - t0_km
            km_time_total += km_time
        km_avg = km_time_total / 10
        print("Average Runtime: ", km_avg)
        labels = km.labels_
        km_entry = get_metrics(array_norm, labels)
        km_entry.insert(0, km_avg)
        km_entry.insert(0, name)
        km_results.append(km_entry)

        # Mini Batch K-Means (MK)
        print("Algorithm: MK")
        # do 10 runs and take average
        mk_time_total = 0.0
        for x in range(10):
            t0_mk = time()
            mk = MiniBatchKMeans(n_clusters=num_classes, init='k-means++').fit(array_norm)
            mk_time = time() - t0_mk
            mk_time_total += mk_time
        mk_avg = mk_time_total / 10
        print("Average Runtime: ", mk_avg)
        labels = mk.labels_
        mk_entry = get_metrics(array_norm, labels)
        mk_entry.insert(0, mk_avg)
        mk_entry.insert(0, name)
        mk_results.append(mk_entry)

        # Ward Agglomerative (WA)
        print("Algorithm: WA")
        # do 10 runs and take average
        wa_time_total = 0.0
        for x in range(10):
            t0_wa = time()
            wa = AgglomerativeClustering(n_clusters=num_classes, affinity="euclidean", linkage="ward").fit(array_norm)
            wa_time = time() - t0_wa
            wa_time_total += wa_time
        wa_avg = wa_time_total / 10
        print("Average Runtime: ", wa_avg)
        labels = wa.labels_
        wa_entry = get_metrics(array_norm, labels)
        wa_entry.insert(0, wa_avg)
        wa_entry.insert(0, name)
        wa_results.append(wa_entry)
        
        print("---------------------------------------------------")


    # write to files
    pandas.DataFrame(aa_results).to_csv("~/Desktop/Thesis/Results/aa_results.csv", index = False, header = False)
    pandas.DataFrame(ca_results).to_csv("~/Desktop/Thesis/Results/ca_results.csv", index = False, header = False)
    pandas.DataFrame(gmd_results).to_csv("~/Desktop/Thesis/Results/gmd_results.csv", index = False, header = False)
    pandas.DataFrame(gmf_results).to_csv("~/Desktop/Thesis/Results/gmf_results.csv", index = False, header = False)
    pandas.DataFrame(km_results).to_csv("~/Desktop/Thesis/Results/km_results.csv", index = False, header = False)
    pandas.DataFrame(mk_results).to_csv("~/Desktop/Thesis/Results/mk_results.csv", index = False, header = False)
    pandas.DataFrame(wa_results).to_csv("~/Desktop/Thesis/Results/wa_results.csv", index = False, header = False)
    

def get_metrics(array_norm, labels):

    metric_results = []

    # Calinski Harabasz
    ch = metrics.calinski_harabaz_score(array_norm, labels)
    print("Calinski Harabasz: ", ch)
    metric_results.append(ch)

    # Silhouette
    si = metrics.silhouette_score(array_norm, labels)
    print("Silhouette: ", si)
    metric_results.append(si)

    # Dunn
    ro.r('suppressPackageStartupMessages(library("fpc"))')
    r_data = numpy2ri.py2rpy(array_norm)
    r_labels = numpy2ri.py2rpy(labels)
    ro.r.assign('r_data', r_data)
    ro.r.assign('r_labels', r_labels)
    ro.r('stats <- cluster.stats(d = dist(r_data), r_labels)')
    dn = ro.r('stats$dunn')[0]
    print("Dunn: ", dn)
    metric_results.append(dn)

    # Pearson Gamma
    pg = ro.r('stats$pearsongamma')[0]
    print("Pearson Gamma: ", pg)
    metric_results.append(pg)

    # Tau
    ro.r('suppressPackageStartupMessages(library("clusterCrit"))')
    ro.r('stats <- intCriteria(r_data, as.integer(r_labels), "all")')
    tu = ro.r('stats$tau')[0]
    print("Tau: ", tu)
    metric_results.append(tu)

    # Davies-Bouldin
    db = ro.r('stats$davies_bouldin')[0]
    print("Davies-Bouldin: ", db)
    metric_results.append(db)

    # Xie-Beni
    xb = ro.r('stats$xie_beni')[0]
    print("Xie-Beni: ", xb)
    metric_results.append(xb)
    
    # SD-Scat
    ss = ro.r('stats$sd_scat')[0]
    print("SD-Scat: ", ss)
    metric_results.append(ss)
    
    # SD-Dis
    sd = ro.r('stats$sd_dis')[0]
    print("SD-Dis: ", sd)
    metric_results.append(sd)

    # Ray-Turi
    rt = ro.r('stats$ray_turi')[0]
    print("Ray-Turi: ", rt)
    metric_results.append(rt)

    print("\n")

    return metric_results



directory = "~/Desktop/Thesis/Data"
run_experiments(directory)

