import sys
import re
import pandas as pd
import numpy as np

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
