import re

def preprocess(in_sentence, language):
    """
    This function preprocesses the input text according to language-specific rules.
    Specifically, we separate contractions according to the source language, convert
    all tokens to lower-case, and separate end-of-sentence punctuation

	INPUTS:
	in_sentence : (string) the original sentence to be processed
	language	: (string) either 'e' (English) or 'f' (French)
				   Language of in_sentence

	OUTPUT:
	out_sentence: (string) the modified sentence
    """

    out_sentence = in_sentence.strip()

    # 1. separate final punctuation
    out_sentence = out_sentence[:-1] + " " + out_sentence[-1]

    # 2. seperate other required chars
    out_sentence = re.sub('[.,:;()-+-<>=\"]', lambda x: " " + x.group() + " ", out_sentence)

    # 3. if french, split contractions:
    if language == "f":
        out_sentence = re.sub(r"\b(l'|c'|j'|t'|qu')", lambda x: " " + x.group() + " ", out_sentence)
        re.sub(r"\b(puisqu|lorsqu)'(on|il)\b", lambda x: x.group(1) + "' " + x.group(2), out_sentence)
        re.sub(r"\b(d') +(abord|accord|ailleurs|habitude)\b", lambda x: x.group(1) + x.group(2), out_sentence)

    # 4. remove extra space, add SENTSTART and SENTEND
    wordlist = out_sentence.split()
    wordlist = ['SENTSTART'] + wordlist + ['SENTEND']
    out_sentence = " ".join(wordlist)

    return out_sentence
