from a2.code.log_prob import *
from a2.code.preprocess import *
import os

def preplexity(LM, test_dir, language, smoothing = False, delta = 0.0):
    """
	Computes the preplexity of language model given a test corpus

	INPUT:

	LM : 		(dictionary) the language model trained by lm_train
	test_dir : 	(string) The top-level directory name containing data
				e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	language : `(string) either 'e' (English) or 'f' (French)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	"""

    files = os.listdir(test_dir)
    pp = 0
    N = 0
    vocab_size = len(LM["uni"])

    for ffile in files:
        if ffile.split(".")[-1] != language:
            continue

        opened_file = open(test_dir+ffile, "r")
        for line in opened_file:
            processed_line = preprocess(line, language)
            tpp = log_prob(processed_line, LM, smoothing, delta, vocab_size)

            if tpp > float("-inf"):
                pp = pp + tpp
                N += len(processed_line.split())
        opened_file.close()
    if N > 0:
        pp = 2**(-pp/N)
    return pp

traindir = '../data/Hansard/Testing/'

#test
# test_LM = lm_train("lm_train_testdir/", "e", "e_temp")
# print(preplexity(test_LM, "lm_train_testdir/", "e"))
languages = ['e', 'f']

for lang in languages:
    print("Language:", lang)
    test_LM = lm_train(traindir, lang, lang+"_temp")
    print("No smoothing:", preplexity(test_LM, traindir, lang, smoothing=False, delta=0.1))

    deltas = [100, 10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    for d in deltas:
        print("Delta=", d, preplexity(test_LM, traindir, lang, smoothing=True, delta=d))

    print("--------------------")
