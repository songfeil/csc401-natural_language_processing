#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import _pickle as pickle
import os
import math

import a2.code.decode as decode
from a2.code.align_ibm1 import *
from a2.code.BLEU_score import *
from a2.code.lm_train import *

__author__ = 'Raeid Saqur'
__copyright__ = 'Copyright (c) 2018, Raeid Saqur'
__email__ = 'raeidsaqur@cs.toronto.edu'
__license__ = 'MIT'


discussion = """
Discussion :

{Enter your intriguing discussion (explaining observations/results) here}

"""

##### HELPER FUNCTIONS ########
def _getLM(data_dir, language, fn_LM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data from which
                    to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
    language    : (string) either 'e' (English) or 'f' (French)
    fn_LM       : (string) the location to save the language model once trained
    use_cached  : (boolean) optionally load cached LM using pickle.

    Returns
    -------
    A language model
    """
    if use_cached and os.path.isfile(fn_LM+'.pickle'):
        with open(fn_LM+'.pickle', 'rb') as f:
            LM = pickle.load(f)
    else:
        LM = lm_train(data_dir, language, fn_LM)

    return LM

def _getAM(data_dir, num_sent, max_iter, fn_AM, use_cached=True):
    """
    Parameters
    ----------
    data_dir    : (string) The top-level directory continaing the data
    num_sent    : (int) the maximum number of training sentences to consider
    max_iter    : (int) the maximum number of iterations of the EM algorithm
    fn_AM       : (string) the location to save the alignment model
    use_cached  : (boolean) optionally load cached AM using pickle.

    Returns
    -------
    An alignment model
    """
    if use_cached and os.path.isfile(fn_AM+'.pickle'):
        with open(fn_AM+'.pickle', 'rb') as f:
            AM = pickle.load(f)
    else:
        AM = align_ibm1(data_dir, num_sent, max_iter, fn_AM)

    return AM

def _get_BLEU_scores(eng_decoded, eng, google_refs, n):
    """
    Parameters
    ----------
    eng_decoded : an array of decoded sentences
    eng         : an array of reference handsard
    google_refs : an array of reference google translated sentences
    n           : the 'n' in the n-gram model being used

    Returns
    -------
    An array of evaluation (BLEU) scores for the sentences
    """
    result = []

    for de, en, go in zip(eng_decoded, eng, google_refs):
        candidate, references = de, [en, go]

        pi = 1
        bp = BLEU_score(candidate, references, 0, True)
        for i in range(1, n + 1):
            pi *= BLEU_score(candidate, references, i, False)

        result.append(bp * (pi ** (1 / n)))

    return result
    # return [BLEU_score(de, [en, go], n) for de, en, go in zip(eng_decoded, eng, google_refs)]


def main(args):
    """
    #TODO: Perform outlined tasks in assignment, like loading alignment
    models, computing BLEU scores etc.

    (You may use the helper functions)

    It's entirely upto you how you want to write Task5.txt. This is just
    an (sparse) example.
    """
    ## Write Results to Task5.txt (See e.g. Task5_eg.txt for ideation). ##

    f = open("Task5.txt", 'w+')
    f.write(discussion)
    f.write("\n\n")
    f.write("-" * 10 + "Evaluation START" + "-" * 10 + "\n")

    datadir = '../data/Hansard/Training/'
    test_dir = '../data/Hansard/Testing/Task5.f'
    hansard_ref_dir = '../data/Hansard/Testing/Task5.e'
    google_ref_dir = '../data/Hansard/Testing/Task5.google.e'

    test, href, gref = [], [], []
    with open(test_dir) as file:
        test.extend([preprocess(l.strip(), 'f') for l in file.readlines()])
    with open(hansard_ref_dir) as file:
        href.extend([preprocess(l.strip(), 'e') for l in file.readlines()])
    with open(google_ref_dir) as file:
        gref.extend([preprocess(l.strip(), 'e') for l in file.readlines()])

    use_cached = True
    max_iter = 50
    f.write("MAX ITERATION: " + str(max_iter) + "\n")

    AM_size = [1000, 10000, 15000, 30000]
    AM_names = ['1K', '10K', '15K', '30K']
    AMs = [_getAM(datadir, sn[0], max_iter, sn[1]+ str(max_iter) +'_AM', use_cached) for sn in zip(AM_size, AM_names)]

    eLM = _getLM(datadir, 'e', 'e_LM', use_cached)
    fLM = _getLM(datadir, 'f', 'f_LM', use_cached)


    for i, AM in enumerate(AMs):

        f.write(f"\n### Evaluating AM model: {AM_names[i]} ### \n")
        # Decode using AM #
        decoded_res = []
        for fr in test:
            decoded_res.append(decode.decode(fr, LM=eLM, AM=AM))

        # Eval using 3 N-gram models #
        all_evals = []
        for n in range(1, 4):
            f.write(f"\nBLEU scores with N-gram (n) = {n}: ")
            evals = _get_BLEU_scores(decoded_res, href, gref, n)
            for v in evals:
                f.write(f"\t{v:1.4f}")
            all_evals.append(evals)

        f.write("\n\n")

    f.write("-" * 10 + "Evaluation END" + "-" * 10 + "\n")
    f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use parser for debugging if needed")
    args = parser.parse_args()

    main(args)
