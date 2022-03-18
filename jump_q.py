import os, sys
import numpy as np
import pandas as pd
from datetime import datetime
from utils import getParams, correctImpurity
from reporter import extractReporters
from filters import filterPSMs
from normalization import getLoadingBias, normalization
from summarization import summarization
from publication import generateTables

if __name__ == "__main__":

    startTime = datetime.now()
    startTimeString = startTime.strftime("%Y/%m/%d %H:%M:%S")
    print("  " + startTimeString)
    print("  JUMPq for the quantification of TMT-dataset\n")

    ##################
    # Initialization #
    ##################
    paramFile = sys.argv[1]
    # paramFile = "/Users/jcho/Research/JUMPq/Example/jump_q_HH_tmt10.params"
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

    #################
    # Summarization #
    #################
    # 1. Peptide-level summarization
    print("\n  Peptide-level summarization is being performed")
    pep2psm = dfId.groupby("Peptide")["key"].apply(lambda x: list(np.unique(x))).to_dict()
    # dfPep = parSummarization(pep2psm, dfNorm, params)
    dfPep = summarization(pep2psm, dfNorm, params, 'peptide')
    # dfPep.to_csv(os.path.join(saveDir, "id_all_pep_quan_python.txt"), sep="\t")

    # 2. Protein-level summarization
    print("\n  Protein-level summarization is being performed")
    prot2psm = dfId.groupby("Protein")["key"].apply(lambda x: list(np.unique(x))).to_dict()
    # dfProt = parSummarization(prot2psm, dfNorm, params)
    dfProt = summarization(prot2psm, dfNorm, params, 'protein')
    # dfProt.to_csv(os.path.join(saveDir, "id_all_prot_quan_python.txt"), sep="\t")

    ######################
    # Publication tables #
    ######################
    dfUniPep, dfAllPep, dfUniProt, dfAllProt = generateTables(dfPep, dfProt, params)
    dfUniPep.to_csv(os.path.join(saveDir, "id_uni_pep_quan.txt"), sep="\t", index=False)
    dfAllPep.to_csv(os.path.join(saveDir, "id_all_pep_quan.txt"), sep="\t", index=False)
    dfUniProt.to_csv(os.path.join(saveDir, "id_uni_prot_quan.txt"), sep="\t", index=False)
    dfAllProt.to_csv(os.path.join(saveDir, "id_all_prot_quan.txt"), sep="\t", index=False)

    endTime = datetime.now()
    endTimeString = endTime.strftime("%Y/%m/%d %H:%M:%S")
    print("\n  " + endTimeString)
    elapsed = (endTime - startTime).total_seconds()
    print("  Finished in {} seconds".format(int(elapsed)))
