from a2.code.preprocess import *
from a2.code.lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0.0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""

    #TODO: Implement by student.

    lp = 0.0

    wordlist = sentence.split()
    for idx, word in enumerate(wordlist):
        if idx == len(wordlist) - 1 or word == 'SENTEND':
            continue

        next_word = wordlist[idx + 1]

        c_wd = LM['uni'].get(word, 0)
        c_wd_nwd = LM['bi'].get(word, {}).get(next_word, 0)

        if smoothing:
            c_wd += delta * vocabSize
            c_wd_nwd += delta

        if c_wd == 0 or c_wd_nwd == 0:
            return float('-inf')

        lp += log(c_wd_nwd / c_wd, 2)

    return lp
