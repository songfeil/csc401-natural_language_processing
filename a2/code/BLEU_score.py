import math

def BLEU_score(candidate, references, n, brevity=False):
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

    wordlist = candidate.split()

    # pn = 1
    # for i in range(n):
    #     count = 0
    #     for j in range(len(wordlist) - i):
    #         curr = " ".join(wordlist[j:j+i+1])
    #         count += 1 if sum([curr in x for x in references]) > 0 else 0
    #     pn *= count / (len(wordlist) - i + 1)
    # bleu_score = bp * pn

    count = 0
    for j in range(len(wordlist) - n):
        curr = " ".join(wordlist[j:j + n + 1])
        count += 1 if sum([curr in x for x in references]) > 0 else 0

    res = count / (len(wordlist) - n)

    if brevity:
        sent_len = len(wordlist)
        ref_len = [len(x.split()) for x in references]

        abs_len = [abs(x - sent_len) for x in ref_len]
        similiar_len = ref_len[abs_len.index(min(abs_len))]
        brevity = similiar_len / sent_len
        bp = math.exp(1 - brevity) if brevity >= 1 else 1

        if n == 0:
            return bp

        res *= bp

    return res
