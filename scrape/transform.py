from collections import Counter

import pandas as pd
import pickle

from pprint import pprint


def run():
    with open(r'C:\repos\trade\data\etfs\list', 'rb') as f:
        etfs = pickle.load(f)
    pdfs = []
    for dct in etfs:
        pdfs.append(dct)
        pdfs.append(pd.DataFrame(dct['rows'], columns=dct['header']))
    pdf = pd.concat(pdfs)
    pdf = pdf[~pdf['Symbol'].duplicated()].reset_index()
    pprint(pdf['Symbol'])

    Counter()


if __name__ == '__main__':
    run()
