'''
    Author: Adam Jilling
    Scans through a folder of datasets and calculates 6 statistical metafeatures for each:
    NE, NEA, PMV, MN, SK, PO
'''

import rpy2.robjects as ro
import numpy as np
import pandas

def calculate_statistical_metafeatures(directory):

    metafeatures = []
    metafeatures.append(["name", "ne", "nea", "pmv", "mn", "sk", "po"])
    ro.r.assign('directory', directory)
    ro.r('fileNames = list.files(directory, pattern = ".csv")')
    fileNames = ro.r('fileNames')

    for name in fileNames:
        ro.r.assign('name', name)
        ro.r('mf = read.csv(file = paste0(directory, "/", name), header = FALSE, sep = ",")')
        ro.r('matrix = as.matrix(mf)')
        matrix = ro.r('matrix = as.matrix(mf)')
        print("DATASET: ", name)

        # calculate NE
        ne = ro.r('nrow(matrix)')[0]
        ro.r.assign('ne', ne)
        print("ne = ", ne)

        # calculate NEA
        nea = ro.r('nrow(matrix)/ncol(matrix)')[0]
        ro.r.assign('nea', nea)
        print("nea = ", nea)

        # calculate PMV
        pmv = ro.r('which(is.na(matrix))/nrow(matrix)*ncol(matrix)')
        ro.r.assign('pmv', pmv)
        pmv = np.size(pmv)
        print("pmv = ", pmv)

        # calculate MN
        ro.r('suppressPackageStartupMessages(library("MVN"))')
        ro.r('mn <- mvn(matrix, mvnTest = "royston")')
        mn = ro.r('as.double(mn$multivariateNormality[3])')[0]
        ro.r.assign('as.double(mn$multivariateNormality[3])', mn)
        print("mn = ", mn)

        # calculate SK
        ro.r('sk <- mvn(matrix, mvnTest = "mardia")')
        sk = ro.r('sk$multivariateNormality[1, 3]').levels[0]
        ro.r.assign('sk$multivariateNormality', sk)
        print("sk = ", sk)

        # calculate PO
        matrix = np.array(matrix)
        num_elements = np.size(matrix)
        num_columns = np.shape(matrix)[1]
        num_outliers = 0
        subarrays = np.array_split(matrix, num_columns, axis = 1)

        for val in subarrays:
            standard_dev = np.std(val)
            mean = np.mean(val)
            lower_bound = mean - (2*standard_dev)
            upper_bound = mean + (2*standard_dev)
            for num in val:
                if (num < lower_bound or num > upper_bound):
                    num_outliers += 1

        po = num_outliers/num_elements
        print("po = ", po)

        print("\n")

        # add to result
        metafeatures.append([name, ne, nea, pmv, mn, sk, po])

    # write to file
    pd = pandas.DataFrame(metafeatures)
    pd.to_csv("~/Desktop/Thesis/Results/stat_features.csv", index = False, header = False)

directory = "~/Desktop/Thesis/Data"
calculate_statistical_metafeatures(directory)

