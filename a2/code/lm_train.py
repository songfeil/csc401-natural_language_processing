#TODO: UNDO MODIFICATION TO preprocess
from a2.code.preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM

	INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained

    OUTPUT

	LM			: (dictionary) a specialized language model

	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which
	incorporate unigram or bigram counts

	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """

    # TODO: Implement Function
    language_model = {'uni': {}, 'bi': {}}

    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if os.path.splitext(file)[1] != '.' + language:
                continue
            fullFile = os.path.join(subdir, file)

            with open(fullFile, 'r') as f:
                lines = [preprocess(x.strip(), language) for x in f.readlines()]
                for line in lines:
                    wordlist = line.split()
                    for idx, word in enumerate(wordlist):
                        language_model['uni'][word] = language_model['uni'].get(word, 0) + 1

                        if idx == len(wordlist) - 1 or word == "SENTEND":
                            continue  # Skip bigram

                        bidict = language_model['bi']
                        nword = wordlist[idx + 1]
                        worddict = bidict.setdefault(word, {})
                        worddict[nword] = worddict.get(nword, 0) + 1

    # Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(language_model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return language_model

