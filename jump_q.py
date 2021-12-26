#!/usr/bin/env python

import os, sys, re, numpy as np, pandas as pd
import numpy.ma as ma
from datetime import datetime
from pyteomics import ms2, mzxml
from scipy.stats import t
# import multiprocessing, tqdm
# from contextlib import contextmanager
# from functools import partial
# from itertools import islice


"""
@contextmanager
def poolcontext(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
"""

class progressBar:
    def __init__(self, total):
        self.total = total
        self.barLength = 20
        self.count = 0
        self.progress = 0
        self.block = 0
        self.status = ""

    def increment(self):
        self.count += 1
        self.progress = self.count / self.total
        self.block = int(round(self.barLength * self.progress))
        if self.progress == 1:
            self.status = "Done...\r\n"
        else:
            self.status = ""

        text = "\r    Progress: [{0}] {1}% {2}".format("#" * self.block + "-" * (self.barLength - self.block), int(self.progress * 100), self.status)
        sys.stdout.write(text)
        sys.stdout.flush()


def correctImpurity(df, params):
    if params['impurity_correction'] == "1":
        reporters = params["tmt_reporters_used"].split(";")
        dfImpurity = pd.read_table(params["impurity_matrix"], sep="\t", skiprows=1, header=None, index_col=0)
        dfImpurity = pd.DataFrame(np.linalg.pinv(dfImpurity.values), dfImpurity.columns, dfImpurity.index)
        dfCorrected = df[reporters].dot(dfImpurity.T)
        dfCorrected.columns = reporters
        df[reporters] = pd.concat([df[reporters]/2, dfCorrected]).groupby(level=0).max()

    return df


def ESDtest(x, alpha, maxOLs):
    xm = ma.array(x)
    n = len(xm)
    R, L, minds = [], [], []
    for i in range(maxOLs):
        # Compute mean and std of x
        xmean = xm.mean()
        xstd = xm.std(ddof=1)

        # Find maximum deviation
        rr = np.abs((xm - xmean) / xstd)
        minds.append(np.argmax(rr))
        R.append(rr[minds[-1]])
        p = 1.0 - alpha / (2.0 * (n - i))
        perPoint = t.ppf(p, n - i - 2)
        L.append((n - i - 1) * perPoint / np.sqrt((n - i - 2 + perPoint ** 2) * (n - i)))

        # Mask that value and proceed
        xm[minds[-1]] = ma.masked

    # Find the number of outliers
    ofound = False
    for i in range(maxOLs - 1, -1, -1):
        if R[i] > L[i]:
            ofound = True
            break

    # Prepare return value
    if ofound:
        return minds[0: i + 1]    # There are outliers
    else:
        return []    # No outliers could be detected


"""
def parExtractReporters(files, df, params, nCores, **kwargs):
    if "sig126" in kwargs:
        print("\n  Refined extraction of TMT reporter ion peaks")
    else:
        print("\n  Extraction of TMT reporter ion peaks")

    results = []
    with poolcontext(processes=nCores) as pool:
        for result in tqdm.tqdm(pool.imap(partial(singleExtractReporters, df=df, params=params, **kwargs), files),
                                total=len(files),
                                bar_format="    Progress: [{bar:20}] {percentage:3.0f}%"):
            results.append(result)

    # Processing the output
    res = pd.concat(results, axis=0)

    # Summary of quantified TMT reporter ions
    print()
    reporters = params["tmt_reporters_used"].split(";")
    reporterSummary = getReporterSummary(res, reporters)
    nTot = len(res)
    for reporter in reporters:
        n = reporterSummary[reporter]["nPSMs"]
        print("    %s\t%d (%.2f%%) matched" % (reporter, n, n / nTot * 100))

    return res, reporterSummary


def singleExtractReporters(file, df, params, **kwargs):
    # Input arguments
    # files: mzXML or ms2 files to be quantified
    # df: dataframe of ID.txt file
    # params: parameters

    dictQuan = {}
    ext = os.path.splitext(file)[-1]
    if ext == ".mzXML":
        reader = mzxml.MzXML(file)  # mzXML file reader
    elif ext == ".ms2":
        reader = ms2.IndexedMS2(file)  # MS2 file reader
    else:
        sys.exit(" Currently, either .mzXML or .ms2 file is supported")

    # Extraction of TMT reporter ions in each fraction
    scans = list(df['scan'][df['frac'] == file].unique())
    for scan in scans:
        spec = reader[str(scan)]
        res = getReporterIntensity(spec, params, **kwargs)  # Array of reporter m/z and intensity values
        key = file + "_" + str(scan)
        dictQuan[key] = res

    # Create a dataframe of quantification data
    reporters = params["tmt_reporters_used"].split(";")
    colNames = [re.sub("sig", "mz", i) for i in reporters] + reporters
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=colNames)
    return res
"""


def extractReporters(files, df, params, **kwargs):
    # Input arguments
    # files: mzXML or ms2 files to be quantified
    # df: dataframe of ID.txt file
    # params: parameters

    if "sig126" in kwargs:
        print("\n  Refined extraction of TMT reporter ion peaks")
    else:
        print("\n  Extraction of TMT reporter ion peaks")

    dictQuan = {}
    for file in files:
        print("    Working on {}".format(os.path.basename(file)))
        ext = os.path.splitext(file)[-1]
        if ext == ".mzXML":
            reader = mzxml.MzXML(file)  # mzXML file reader
        elif ext == ".ms2":
            reader = ms2.IndexedMS2(file)  # MS2 file reader
        else:
            sys.exit(" Currently, either .mzXML or .ms2 file is supported")

        # Extraction of TMT reporter ions in each fraction
        scans = list(df['scan'][df['frac'] == file].unique())
        progress = progressBar(len(scans))
        for scan in scans:
            progress.increment()
            spec = reader[str(scan)]
            res = getReporterIntensity(spec, params, **kwargs)  # Array of reporter m/z and intensity values
            key = file + "_" + str(scan)
            dictQuan[key] = res

    # Create a dataframe of quantification data
    reporters = params["tmt_reporters_used"].split(";")
    colNames = [re.sub("sig", "mz", i) for i in reporters] + reporters
    res = pd.DataFrame.from_dict(dictQuan, orient='index', columns=colNames)

    # Summary of quantified TMT reporter ions
    print()
    reporterSummary = getReporterSummary(res, reporters)
    nTot = len(res)
    for reporter in reporters:
        n = reporterSummary[reporter]["nPSMs"]
        print("    %s\t%d (%.2f%%) matched" % (reporter, n, n / nTot * 100))

    return res, reporterSummary


def filterByIntensity(df, methods, thresholds, reporters, verbose=1):
    methodsStr = ["none", "minimum intensity", "maximum intensity", "mean intensity", "median intensity"]
    res = []
    n = 0
    for i in range(len(methods)):
        if methods[i] == "0":
            pass
        else:
            idx = getFileteredIndexes(df, methods[i], float(thresholds[i]), reporters)    # "idx" is the index to be removed
            res.extend(idx.values)
        if verbose:
            res = list(set(res))
            print("    Removed {} PSMs based on the intensity-based filter ({})".format(len(res) - n, methodsStr[int(methods[i])]))
            n = len(res)

    return res


def filterPSMs(df, params):
    print("\n  Examining the extracted TMT reporter ions in PSMs")
    reporters = params["tmt_reporters_used"].split(";")

    # 0. Zero-intensity filter
    n = len(df)
    df = df[(df[reporters] > 0).all(axis=1)]
    print("    Removed {} PSMs due to zero intensity at least one channel".format(n - len(df)))

    # 1. Intensity-based filtering of all PSMs
    methods = params["min_intensity_method"].split(",")
    thresholds = params["min_intensity_value"].split(",")
    idx = filterByIntensity(df, methods, thresholds, reporters, 1)  # Indexes to be removed
    idx = list(set(idx))
    df = df[~df.index.isin(idx)]
    print("    Hereafter, {} PSMs will be used for the quantification".format(len(df)))

    # # 2. Further filtering when only 1 or 2 PSMs are mapped to a protein
    # print("    Further filtering of 1 or 2 PSMs mapped to a protein")
    # methods = params["min_intensity_method_1_2_psm"].split(",")
    # thresholds = params["min_intensity_value_1_2_psm"].split(",")
    # idx = []
    # progress = progressBar(len(prot2psm))
    # for prot, psms in prot2psm.items():
    #     progress.increment()
    #     psms = df.index.join(psms, how="inner")
    #
    #     if len(psms) == 0:
    #         continue
    #     elif len(psms) == 1:
    #         # Proteins mapped by only one PSM
    #         # If the PSM is filtered by the intensity-based filter, it will not be used for the quantification
    #         idxProt = filterByIntensity(df.loc[psms], methods, thresholds, reporters, 0)
    #         if len(idxProt) > 0:
    #             idx.extend(idxProt)
    #     elif len(psms) == 2:
    #         # Proteins mapped by two PSMs
    #         # Apply the intensity-based filter first
    #         # - If both PSMs are filtered out, they will not be used for the quantification
    #         # - If one of PSMs is filtered out, the PSM will not be used for the quantification
    #         # - If none of PSMs is filtered out, go to the next step (two PSMs can be used for the quantification)
    #         #   - For each PSM, check the variation (i.e., stdev) across the reporters (in log2-space)
    #         #   - One with smaller variation will be used for the quantification
    #         #   - If both PSMs have the same variation, the one with higher mean intensity will be used
    #         #         subDf = filterPSM1(subDf, methods, thresholds, reporters, 0)
    #         idxProt = filterByIntensity(df.loc[psms], methods, thresholds, reporters, 0)
    #         if len(idxProt) > 0:
    #             idx.extend(idxProt)
    #         else:
    #             psmStd = np.log2(df.loc[psms][reporters]).std(axis=1)
    #             psmMean = np.log2(df.loc[psms][reporters]).mean(axis=1)
    #             if psmStd[0] == psmStd[1]:
    #                 ii = np.argmin(psmMean)
    #             else:
    #                 ii = np.argmax(psmStd)
    #             idx.extend([psms[ii]])
    #
    # idx = list(set(idx))
    # print("    Removed {} PSMs due to the larger variation than the other PSM mapped to the same protein".format(len(idx)))
    # df = df[~df.index.isin(idx)]

    return df


def getFileteredIndexes(df, method, threshold, reporters):
    # This module produces indexes to be removed (i.e., filtered indexes)
    if method == '1':  # Minimum-based filter
        idx = df[(df[reporters] < threshold).any(axis=1)].index
    elif method == '2':  # Maximum-based filter
        idx = df[(df[reporters] > threshold).any(axis=1)].index
    elif method == '3':  # Mean-based filter
        idx = df[df[reporters].mean(axis=1) < threshold].index
    elif method == '4':  # Median-based filter
        idx = df[df[reporters].median(axis=1) < threshold].index
    else:
        sys.exit("  Please check 'min_intensity_method' parameter. It should be 0, 1, 2, 3, or 4")

    return idx


def getLoadingBias(df, params):
    ###########################
    # Loading-bias evaluation #
    ###########################
    subDf = getSubset(df, params)
    n = len(subDf)
    sm = 2 ** subDf.mean(axis=0)    # Sample-mean values
    msm = np.mean(sm)    # Mean of sample-mean values
    avg = sm / msm * 100
    sdVal = subDf.std(axis=0)
    sd = ((2 ** sdVal - 1) + (1 - 2 ** (-sdVal))) / 2 * 100
    sem = sd / np.sqrt(n)

    return avg, sd, sem, n


def getParams(paramFile):
    parameters = dict()
    with open(paramFile, 'r') as file:
        for line in file:
            if re.search(r'^#', line) or re.search(r'^\s', line):
                continue
            line = re.sub(r'#.*', '', line)  # Remove comments (start from '#')
            line = re.sub(r'\s*', '', line)  # Remove all whitespaces

            # Exception for "feature_files" parameter
            if "feature_files" in parameters and line.endswith("feature"):
                parameters["feature_files"].append(line)
            else:
                key = line.split('=')[0]
                val = line.split('=')[1]
                if key == "feature_files":
                    parameters[key] = [val]
                else:
                    parameters[key] = val

    return parameters


def getReporterIntensity(spec, params, **kwargs):
    tol = 10
    reporterNames = params["tmt_reporters_used"].split(";")
    mzArray = []
    intensityArray = []

    for reporter in reporterNames:
        if reporter in kwargs:
            mz = getReporterMz(reporter) * (1 + kwargs[reporter]["meanMzShift"] / 1e6)
            tol = kwargs[reporter]["sdMzShift"] * np.float(params['tmt_peak_extraction_second_sd'])
        else:
            mz = getReporterMz(reporter)

        lL = mz - mz * tol / 1e6
        uL = mz + mz * tol / 1e6
        ind = np.where((spec["m/z array"] >= lL) & (spec["m/z array"] <= uL))[0]
        if len(ind) == 0:
            mz = 0
        elif len(ind) == 1:
            ind = ind[0]
            mz = spec["m/z array"][ind]
        elif len(ind) > 1:
            if params['tmt_peak_extraction_method'] == '2':
                ind2 = np.argmin(abs(mz - spec["m/z array"][ind]))
                ind = ind[ind2]
                mz = spec["m/z array"][ind]
            else:
                ind2 = np.argmax(spec["intensity array"][ind])
                ind = ind[ind2]
                mz = spec["m/z array"][ind]
        if lL <= mz < uL:
            intensity = spec["intensity array"][ind]
        else:
            intensity = 0
        mzArray.append(mz)
        intensityArray.append(intensity)

    outArray = mzArray + intensityArray
    return outArray


def getReporterMz(name):
    if name == "sig126":
        return 126.127726
    elif name == "sig127" or name == "sig127N":
        return 127.124761
    elif name == "sig127C":
        return 127.131081
    elif name == "sig128N":
        return 128.128116
    elif name == "sig128" or name == "sig128C":
        return 128.134436
    elif name == "sig129" or name == "sig129N":
        return 129.131471
    elif name == "sig129C":
        return 129.137790
    elif name == "sig130N":
        return 130.134825
    elif name == "sig130" or name == "sig130C":
        return 130.141145
    elif name == "sig131" or name == "sig131N":
        return 131.138180
    elif name == "sig131C":
        return 131.144500
    elif name == "sig132N":
        return 132.141535
    elif name == "sig132C":
        return 132.147855
    elif name == "sig133N":
        return 133.144890
    elif name == "sig133C":
        return 133.151210
    elif name == "sig134N":
        return 134.148245


def getReporterSummary(df, reporters):
    print("  Summary of quantified TMT reporter ions")
    res = {}
    for reporter in reporters:
        res[reporter] = {}
        reporterMz = getReporterMz(reporter)
        measuredMz = df[reporter.replace("sig", "mz")]
        measuredMz = measuredMz[measuredMz > 0]
        n = len(measuredMz)
        meanMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).mean()
        sdMzShift = ((measuredMz - reporterMz) / reporterMz * 1e6).std(ddof=1)
        res[reporter]['nPSMs'] = n
        res[reporter]['meanMzShift'] = meanMzShift
        res[reporter]['sdMzShift'] = sdMzShift

    return res


def getSubset(df, params):
    # Get a subset of a dataframe to calculate loading-bias information
    # 1. Filter out PSMs based on the intensity level
    reporters = params["tmt_reporters_used"].split(";")
    noiseLevel = 1000
    snRatio = float(params["SNratio_for_correction"])
    subDf = df[reporters][(df[reporters] > noiseLevel * snRatio).prod(axis=1).astype(bool)]  # Zero-intensity PSMs are excluded

    # 2. Filter out highly variant PSMs in each column (reporter)
    psmMean = subDf.mean(axis=1)
    subDf = np.log2(subDf.divide(psmMean, axis=0))
    pctTrimmed = float(params["percentage_trimmed"])
    n = 0
    for reporter in reporters:
        if n == 0:
            ind = ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                   (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        else:
            ind = ind & ((subDf[reporter] > subDf[reporter].quantile(pctTrimmed / 200)) &
                         (subDf[reporter] < subDf[reporter].quantile(1 - pctTrimmed / 200)))
        n += 1

    subDf = subDf.loc[ind]
    return subDf


def normalization(df, params):
    ################################################
    # Normalization (i.e. loading-bias correction) #
    ################################################
    doNormalization = params["loading_bias_correction"]
    normalizationMethod = params["loading_bias_correction_method"]
    if doNormalization == "1":
        # First, get a subset for calculating normalization factors (same as loading-bias calculation)
        # Note that this subset is 1) divided by row-wise mean (i.e. PSM-wise mean) and then 2) log2-transformed
        subDf = getSubset(df, params)

        # Calculate normalization factors for samples (reporters)
        if normalizationMethod == "1":  # Trimmed-mean
            sm = subDf.mean(axis=0)
        elif normalizationMethod == "2":  # Trimmed-median
            sm = subDf.median(axis=0)
        target = np.mean(sm)
        normFactors = sm - target

        # Normalize the input dataframe, df (in log2-scale and then scale-back)
        res = df.copy()
        psmMeans = res[reporters].mean(axis=1)
        res[reporters] = np.log2(res[reporters].divide(psmMeans, axis=0).replace(0, np.nan))
        res[reporters] = res[reporters] - normFactors
        res[reporters] = 2 ** res[reporters]
        res[reporters] = res[reporters].multiply(psmMeans, axis=0)
        # After the normalization, no need to show loading-bias again (should be 100% for all samples)
        print("\n  Normalization is finished")
    else:
        print("\n  Normalization is skipped according to the parameter")

    return res


# Dictionary used for Dixon's Q-test
def outlierRemoval(df, alpha):
    n = len(df)
    nOutliers = int(np.round(n * 0.2))

    # e.g.,   n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], then
    # nOutliers = [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    # So, the rule becomes as follows,
    # 1. n < 3, no outlier removal
    # 2. 3 <= n <= 7, Dixon's Q-test for a single outlier removal
    # 3. n >= 8, ESD test for the removal of (potentially) multiple outliers

    indArray = []
    if nOutliers > n - 2:
        nOutliers = n - 2
    if nOutliers > 1:
        for i in range(df.shape[1]):
            ind = ESDtest(df.iloc[:, i], alpha, nOutliers)
            indArray.extend(ind)
    else:
        for i in range(df.shape[1]):
            ind = Qtest(df.iloc[:, i], alpha, nOutliers)
            indArray.extend(ind)

    # PSMs including one or more outliers will not be considered for the subsequent quantification
    indArray = list(set(indArray))    # Indices of outliers across all reporters
    df.drop(df.index[indArray], axis=0, inplace=True)
    return df


def Qtest(data, left=True, right=True, alpha=0.05):
    """
    From https://sebastianraschka.com/Articles/2014_dixon_test.html#implementing-a-dixon-q-test-function

    Keyword arguments:
        data = A ordered or unordered list of data points (int or float).
        left = Q-test of minimum value in the ordered list if True.
        right = Q-test of maximum value in the ordered list if True.
        q_dict = A dictionary of Q-values for a given confidence level,
            where the dict. keys are sample sizes N, and the associated values
            are the corresponding critical Q values. E.g.,
            {3: 0.97, 4: 0.829, 5: 0.71, 6: 0.625, ...}
    Returns a list of 2 values for the outliers, or None.
    E.g.,
       for [1,1,1] -> [None, None]
       for [5,1,1] -> [None, 5]
       for [5,1,5] -> [1, None]

    """

    q90 = [0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437,
           0.412, 0.392, 0.376, 0.361, 0.349, 0.338, 0.329,
           0.32, 0.313, 0.306, 0.3, 0.295, 0.29, 0.285, 0.281,
           0.277, 0.273, 0.269, 0.266, 0.263, 0.26
           ]
    Q90 = {n: q for n, q in zip(range(3, len(q90) + 1), q90)}
    q95 = [0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466,
           0.444, 0.426, 0.41, 0.396, 0.384, 0.374, 0.365, 0.356,
           0.349, 0.342, 0.337, 0.331, 0.326, 0.321, 0.317, 0.312,
           0.308, 0.305, 0.301, 0.29
           ]
    Q95 = {n: q for n, q in zip(range(3, len(q95) + 1), q95)}
    q99 = [0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568,
           0.542, 0.522, 0.503, 0.488, 0.475, 0.463, 0.452, 0.442,
           0.433, 0.425, 0.418, 0.411, 0.404, 0.399, 0.393, 0.388,
           0.384, 0.38, 0.376, 0.372
           ]
    Q99 = {n: q for n, q in zip(range(3, len(q99) + 1), q99)}

    if isinstance(data, list):
        pass
    else:
        x = list(data)

    if alpha == 0.1:
        q_dict = Q90
    elif alpha == 0.05:
        q_dict = Q95
    elif alpha == 0.01:
        q_dict = Q99

    assert(left or right), 'At least one of the variables, `left` or `right`, must be True.'
    assert(len(data) >= 3), 'At least 3 data points are required'
    assert(len(data) <= max(q_dict.keys())), 'Sample size too large'

    sdata = sorted(data)
    Q_mindiff, Q_maxdiff = (0,0), (0,0)

    if left:
        Q_min = (sdata[1] - sdata[0])
        try:
            Q_min /= (sdata[-1] - sdata[0])
        except ZeroDivisionError:
            pass
        Q_mindiff = (Q_min - q_dict[len(data)], sdata[0])

    if right:
        Q_max = abs((sdata[-2] - sdata[-1]))
        try:
            Q_max /= abs((sdata[0] - sdata[-1]))
        except ZeroDivisionError:
            pass
        Q_maxdiff = (Q_max - q_dict[len(data)], sdata[-1])

    if not Q_mindiff[0] > 0 and not Q_maxdiff[0] > 0:
        outliers = []
    elif Q_mindiff[0] == Q_maxdiff[0]:
        outliers = [Q_mindiff[1], Q_maxdiff[1]]
    elif Q_mindiff[0] > Q_maxdiff[0]:
        outliers = [Q_mindiff[1]]
    else:
        outliers = [Q_maxdiff[1]]

    outlierInd = [i for i, v in enumerate(data) if v in outliers]
    # survivedInd = np.setdiff1d(range(len(data)), outlierInd)

    return outlierInd


"""
# 2021/9/12
# While testing, "pool" function produces an error in a node of HPC
# It is still not clear why such an error occurs. I will discuss it with Karthik in RIS later
def chunks(data, nCPU):
    size = round(len(data) / nCPU)
    it = iter(data)
    res = []
    for i in range(0, len(data), size):
        res.append({k: data[k] for k in islice(it, size)})

    return res


def parSummarization(inputDict, df, params):
    nProc = round(multiprocessing.cpu_count() / 2)    # For safety, the half of available cores will be used
    listDict = chunks(inputDict, nProc)
    results = []
    with poolcontext(processes=nProc) as pool:
        for result in tqdm.tqdm(pool.imap(partial(summarization, df=df, params=params), listDict), total=len(listDict),
                                bar_format="    Progress: [{bar:20}] {percentage:3.0f}%"):
            results.append(result)
    # Processing the output
    res = pd.concat(results, axis=0)
    return res
"""


def summarization_1_2(df, params):
    # Summarization of proteins (and peptides?) mapped by 1 or 2 PSMs
    methods = params["min_intensity_method_1_2_psm"].split(",")
    thresholds = params["min_intensity_value_1_2_psm"].split(",")
    reporters = params["tmt_reporters_used"].split(";")
    idx = []
    if len(df) == 1:
        # Proteins mapped by only one PSM
        # If the PSM is filtered by the intensity-based filter, it will not be used for the quantification
        idx = filterByIntensity(df, methods, thresholds, reporters, 0)
        if len(idx) > 0:
            df.drop(idx, inplace=True)
    elif len(df) == 2:
        # Proteins mapped by two PSMs
        # Apply the intensity-based filter first
        # - If both PSMs are filtered out, they will not be used for the quantification
        # - If one of PSMs is filtered out, the PSM will not be used for the quantification
        # - If none of PSMs is filtered out, go to the next step (two PSMs can be used for the quantification)
        #   - For each PSM, check the variation (i.e., stdev) across the reporters (in log2-space)
        #   - One with smaller variation will be used for the quantification
        #   - If both PSMs have the same variation, the one with higher mean intensity will be used
        idx = filterByIntensity(df, methods, thresholds, reporters, 0)
        if len(idx) > 0:
            df.drop(idx, inplace=True)
        elif len(idx) == 0:
            psmStd = np.log2(df[reporters]).std(axis=1)
            psmMean = np.log2(df[reporters]).mean(axis=1)
            if psmStd[0] == psmStd[1]:
                ii = np.argmin(psmMean)
            else:
                ii = np.argmax(psmStd)
            df.drop(df.index[ii], inplace=True)

    return df


def summarization(inputDict, df, params, level):
    # Input arguments
    # df: a dataframe containing PSM-level quantification information
    # inputDict: a dictionary containing the relationship between protein (or peptide) and PSMs
    #            e.g., prot2Ppsm: key = each protein, value = list of PSMs corresponding to the protein
    # params: parameters from the .param file
    # level: protein or peptide (using a string)

    resDict = {}
    nRemoved = 0
    reporters = params["tmt_reporters_used"].split(";")
    progress = progressBar(len(inputDict))
    for entry, psms in inputDict.items():
        progress.increment()
        psms = df.index.join(psms, how="inner")
        if len(psms) == 0:
            continue
        else:
            subDf = df.loc[psms][reporters]

            # Limit the number of PSMs when there are TOO many PSMs(?)
            # Choose top-n PSMs according to the PSM-wise total intensity
            threshold = 100
            if len(subDf) > threshold:
                psmSum = subDf.sum(axis=1)
                topIndex = psmSum.sort_values(ascending=False).index[:threshold]
                subDf = subDf.loc[topIndex]

            # Summarization (both peptide and protein)
            if 0 < len(subDf) <= 2:
                subDf = summarization_1_2(subDf, params)
            elif len(subDf) >= 3:
                # Preprocessing for outlier removal
                # 1. Log2-transformation
                # 2. PSM-wise mean calculation
                # 2.1. Representative protein abundance by the mean of top3 PSM-wise means
                #      (equivalent to the grand mean of top3 PSMs)
                # 3. Mean-centering (using the PSM-wise mean obtained at step2)
                # 4. Outlier removal (using either Q-test or ESD test)
                # 5. Scale-back to the raw-scale
                subDf = np.log2(subDf)
                psmMeans = subDf.mean(axis=1)
                repAbundance = np.mean(sorted(psmMeans, reverse=True)[0:3])
                subDf = subDf.sub(psmMeans, axis=0)
                subDf = outlierRemoval(subDf, 0.05)  # Can I make it faster?
                subDf = 2 ** (subDf.mean(axis=0) + repAbundance)

            if len(subDf) > 0:
                resDict[entry] = subDf.to_dict()
            else:
                nRemoved += 1

    print("    {} (out of {}) {}s are not quantified due to the further filtering of PSMs".format(nRemoved, len(inputDict), level))
    res = pd.DataFrame.from_dict(resDict, orient="index")
    return res


if __name__ == "__main__":

    startTime = datetime.now()
    startTimeString = startTime.strftime("%Y/%m/%d %H:%M:%S")
    print("  " + startTimeString)
    print("  JUMPq for the quantification of TMT-dataset\n")

    ##################
    # Initialization #
    ##################
    # paramFile = sys.argv[1]
    paramFile = "jump_lib_q.params"
    params = getParams(paramFile)
    saveDir = os.path.join(os.getcwd(), "quan_" + params["save_dir"])
    os.makedirs(saveDir, exist_ok=True)

    ##################
    # Parsing ID.txt #
    ##################

    # Note that this part may need to be revised according to the Jump -f result format

    print("  Loading ID.txt file")
    dfId = pd.read_table(params["idtxt"], sep=";", skiprows=1, header=0)

    # Miscellaneous part for handling ID.txt
    dfId["frac"] = dfId["Outfile"].apply(lambda x: os.path.dirname(x).rsplit(".", 1)[0] + ".mzXML")
    dfId["scan"] = dfId["Outfile"].apply(lambda x: os.path.basename(x).split(".")[1])
    dfId["key"] = dfId["frac"] + "_" + dfId["scan"]
    fracs = dfId["frac"].unique()

    ##################################
    # Extract TMT reporter ion peaks #
    ##################################
    # 1st round of reporter ion extraction
    # dfQuan, reporterSummary = parExtractReporters(fracs, dfId, params, nCores)
    dfQuan, reporterSummary = extractReporters(fracs, dfId, params)

    # Before 2nd round of TMT reporter extraction, m/z-shifts of reporters are summarized
    print("\n  m/z-shift in each TMT reporter")
    reporters = params["tmt_reporters_used"].split(";")
    for reporter in reporters:
        m = reporterSummary[reporter]["meanMzShift"]
        s = reporterSummary[reporter]["sdMzShift"]
        print("    %s\tm/z-shift = %.4f [ppm]\tsd = %.4f" % (reporter, m, s))

    # 2nd round of reporter ion extraction
    # dfQuan, reporterSummary = parExtractReporters(fracs, dfId, params, nCores, **reporterSummary)
    dfQuan, reporterSummary = extractReporters(fracs, dfId, params, **reporterSummary)

    ###########################
    # TMT impurity correction #
    ###########################
    dfQuan = correctImpurity(dfQuan, params)

    #####################
    # Filtering of PSMs #
    #####################
    dfQuan = filterPSMs(dfQuan, params)

    #####################################
    # Show the loading-bias information #
    #####################################
    avgLb, sdLb, semLb, nn = getLoadingBias(dfQuan, params)
    print("\n  Loading bias (before correction)")
    print("    Reporter\tMean[%]\tSD[%]\tSEM[%]\t#PSMs")
    for i in range(len(reporters)):
        print("    %s\t%.2f\t%.2f\t%.2f\t%d" % (reporters[i], avgLb[i], sdLb[i], semLb[i], nn))

    #################
    # Normalization #
    #################
    dfNorm = normalization(dfQuan, params)
    dfNorm.to_csv(os.path.join(saveDir, "normalized_quan_psm_nonzero.txt"), sep="\t")

    ###############################
    # Peptide-level summarization #
    ###############################
    print("\n  Peptide-level summarization is being performed")
    pep2psm = dfId.groupby("Peptide")["key"].apply(lambda x: list(np.unique(x))).to_dict()
    # dfPep = parSummarization(pep2psm, dfNorm, params)
    dfPep = summarization(pep2psm, dfNorm, params, 'peptide')
    dfPep.to_csv(os.path.join(saveDir, "id_all_pep_quan_python.txt"), sep="\t")

    ###############################
    # Protein-level summarization #
    ###############################
    print("\n  Protein-level summarization is being performed")
    prot2psm = dfId.groupby("Protein")["key"].apply(lambda x: list(np.unique(x))).to_dict()
    # dfProt = parSummarization(prot2psm, dfNorm, params)
    dfProt = summarization(prot2psm, dfNorm, params, 'protein')
    dfProt.to_csv(os.path.join(saveDir, "id_all_prot_quan_python.txt"), sep="\t")

    endTime = datetime.now()
    endTimeString = endTime.strftime("%Y/%m/%d %H:%M:%S")
    print("\n  " + endTimeString)
    elapsed = (endTime - startTime).total_seconds()
    print("  Finished in {} seconds".format(int(elapsed)))
