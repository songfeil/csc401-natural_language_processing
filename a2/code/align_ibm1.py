from a2.code.lm_train import *
from a2.code.log_prob import *
from a2.code.preprocess import *
from math import log
import os
import time

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm.
	We assume that we are implemented P(foreign|english)

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model

	OUTPUT:
	AM :			(dictionary) alignment model structure

	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word']
	is the computed expectation that the foreign_word is produced by english_word.

			LM['house']['maison'] = 0.5
	"""
    AM = {}

    # Read training data
    eng, fre = read_hansard(train_dir, num_sentences)

    # Initialize AM uniformly
    AM = initialize(eng, fre)

    # Iterate between E and M steps
    t = time.time()
    for i in range(max_iter):
        em_step(AM, eng, fre)
        print("Loop", i, "time:", time.time() - t)
        t = time.time()

    with open(fn_AM+'.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return AM

# ------------ Support functions --------------
def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.

	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider


	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.

	Make sure to read the files in an aligned manner.
	"""
    eng, fre = [], []

    for subdir, dirs, files in os.walk(train_dir):
        for file in files:
            fn, ext = os.path.splitext(file)
            if ext != '.' + 'e' or not os.path.isfile(os.path.join(subdir, fn + '.f')):
                continue  # If find french file, or corresponding french file does not exist
            fullEngFile = os.path.join(subdir, fn + '.e')
            fullFreFile = os.path.join(subdir, fn + '.f')

            with open(fullEngFile, 'r') as f:
                eng.extend([preprocess(x.strip(), 'e') for x in f.readlines()])
            with open(fullFreFile, 'r') as f:
                fre.extend([preprocess(x.strip(), 'f') for x in f.readlines()])

            assert len(eng) == len(fre)
            if len(eng) >= num_sentences:
                return eng[:num_sentences], fre[:num_sentences]


def initialize(eng, fre):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    AM = {}

    for idx, (engs, fres) in enumerate(zip(eng, fre)):
        eng_wordlist = engs.split()[1:-1]
        fre_wordlist = fres.split()[1:-1]

        for eword in eng_wordlist:
            edict = AM.setdefault(eword, {})
            for fword in fre_wordlist:
                edict[fword] = 1

    # Divide by length
    for k, v in AM.items():
        AM[k] = {k1: 1 / len(v) for k1, v1 in v.items()}

    AM.update({'SENTSTART': {'SENTSTART': 1}, 'SENTEND': {'SENTEND': 1}})

    return AM


def em_step(AM, eng, fre):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    # TODO
    engSet, freSet = set(), set()
    for idx, (engs, fres) in enumerate(zip(eng, fre)):
        engSet.update(engs.split()[1:-1])
        freSet.update(fres.split()[1:-1])

    # total = {k: 0 for k in engSet}
    tcount, total = {}, {}

    for E, F in zip(eng, fre):
        fwordlist, ewordlist = F.split()[1: -1], E.split()[1: -1]
        fset, eset = set(fwordlist), set(ewordlist)
        for f in fset:
            denom_c = 0
            for e in eset:
                denom_c += AM[e][f] * fwordlist.count(f)
            for e in eset:
                tcount_e = tcount.setdefault(e, {})
                tcount_e[f] = tcount_e.setdefault(f, 0) + AM[e][f] * fwordlist.count(f) * ewordlist.count(e) / denom_c
                total[e] = total.setdefault(e, 0) + AM[e][f] * fwordlist.count(f) * ewordlist.count(e) / denom_c

    for e, total_e in total.items():
        for f, tcount_ef in tcount[e].items():
            AM[e][f] = tcount_ef / total_e

