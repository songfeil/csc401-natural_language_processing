import math

def BLEU_score(candidate, references, n):
    """
    Calculate the BLEU score given a candidate sentence (string) and a list of reference sentences (list of strings). n specifies the level to calculate.
    n=1 unigram
    n=2 bigram
    ... and so on
    
    DO NOT concatenate the measurments. N=2 means only bigram. Do not average/incorporate the uni-gram scores.
    
    INPUTS:
	sentence :	(string) Candidate sentence.  "SENTSTART i am hungry SENTEND"
	references:	(list) List containing reference sentences. ["SENTSTART je suis faim SENTEND", "SENTSTART nous sommes faime SENTEND"]
	n :			(int) one of 1,2,3. N-Gram level.

	
	OUTPUT:
	bleu_score :	(float) The BLEU score
	"""
	
	#TODO: Implement by student.
            
    return bleu_score