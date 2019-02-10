import numpy as np
import sys
import argparse
import os
import json
import re
import csv
import string
import itertools

# import warnings
#
# warnings.simplefilter('error')

def _read_dict(filename, keycoldict, wordpos=1, skipheader=True):
    """
    Read in the csv file and return a dict of {'word': {'aoa': int, 'img': int, 'fam':int}}.
    :param filename: File name for input file
    :type filename: str
    :param keycoldict: Dict for {key: column position}
    :type keycoldict:
    :param wordpos: Index for position of the word
    :type wordpos: int
    :param skipheader: If we want to skip the first line in csv file
    :type skipheader: bool
    :return: Dict containing {'word': {'key': item[col]}}
    :rtype: dict
    """
    d = {}

    with open(filename) as f:
        reader = csv.reader(f)
        for item in reader:
            if skipheader and reader.line_num == 1:
                continue
            d[item[wordpos]] = {key: item[col] for (key, col) in keycoldict.items()}

    return d


def _read_feats(idfile, featsfile):
    feats = np.load(featsfile)

    ids = []
    with open(idfile) as f:
        ids.extend([x.strip() for x in f.readlines()])

    if len(ids) != feats.shape[0]:
        return None

    d = dict()
    for i in range(len(ids)):
        d[ids[i]] = feats[i]

    return d


def _cat_to_int(string):
    return [{
        'Left': 0,
        'Center': 1,
        'Right': 2,
        'Alt': 3
    }[string]]


# globals
_songfeil_globals = {
    'prp': {1: ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'], 2: ['you', 'your', 'yours', 'u', 'ur', 'urs'], 3: [
        'he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']},
    'slang': ['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd', 'lylc', 'brb', 'atm', 'imao',
              'sml', 'btw', 'bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo', 'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys',
              'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic', 'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc',
              'tmi', 'ym', 'ur', 'u', 'sol', 'fml'],
    'future_tense': ['will', "'ll", "gonna"],
    'bgldict': _read_dict("/Users/florian_song/cs401data/wordlists/BristolNorms+GilhoolyLogie.csv", {'aoa': 3, 'img': 4, 'fam': 5}),
    'vaddict': _read_dict('/Users/florian_song/cs401data/wordlists/Ratings_Warriner_et_al.csv', {'v': 2, 'a': 5, 'd': 8}),
    # 'bgldict': _read_dict("/u/cs401/Wordlists/BristolNorms+GilhoolyLogie.csv", {'aoa': 3, 'img': 4, 'fam': 5}),
    # 'vaddict': _read_dict('/u/cs401/Wordlists/Ratings_Warriner_et_al.csv', {'v': 2, 'a': 5, 'd': 8}),
    'bglempty': {'aoa': 0.0, 'img': 0.0, 'fam': 0.0},
    'vadempty': {'v': 0.0, 'a': 0.0, 'd': 0.0}
}


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    feats = np.zeros(173)
    wordlist = list(map(lambda x: x.strip().rsplit('/', 1), comment.strip().split()))

    if len(wordlist) == 0:
        return feats

    # f1-3: first/second/third person pronouns
    for i in range(3):
        feats[i] = sum(map(lambda x: len(re.findall(r"\b(" + x + r"/)", comment)), _songfeil_globals['prp'][i + 1]))

    # f4: coordinating conjunctions
    feats[4 - 1] = sum(map(lambda x: 'CC' == x[1], wordlist))

    # f5: past tense verbs
    feats[5 - 1] = sum(map(lambda x: 'VBD' == x[1], wordlist))

    # f6: future tense verbs
    num_future_word = sum(map(lambda x: len(re.findall(r"\b(" + x + r"/)", comment)), _songfeil_globals['future_tense']))
    num_future_phrase = len(re.findall(r"\bgo/VBG to/TO [a-z]+/VB\b", comment))  # going to VB
    feats[6 - 1] = num_future_phrase + num_future_word

    # f7: commas
    feats[7 - 1] = sum(map(lambda x: ',' == x[1], wordlist))

    # f8: multi-character punctuation tokens
    multi_count = 0
    is_punc = [(x[0] in string.punctuation) for x in wordlist]
    for key, group in itertools.groupby(is_punc):
        if key and len(list(group)) > 1:
            multi_count += 1
    feats[8 - 1] = multi_count

    # f9: common nouns - NN and NNS
    feats[9 - 1] = sum(map(lambda x: 'NN' == x[1] or 'NNS' == x[1], wordlist))

    # f10: proper nouns - NNP and NNPS
    feats[10 - 1] = sum(map(lambda x: 'NNP' == x[1] or 'NNPS' == x[1], wordlist))

    # f11: adverbs - RB, RBR, RBS
    feats[11 - 1] = sum(map(lambda x: 'RB' == x[1] or 'RBR' == x[1] or 'RBS' == x[1], wordlist))

    # f12: wh- words - WDT, WP, WP$, WRB
    feats[12 - 1] = sum(map(lambda x: 'WDT' == x[1] or 'WP' == x[1] or 'WP$' == x[1] or 'WRB' == x[1], wordlist))

    # f13: slang acronyms
    feats[13 - 1] = sum([(x[0] in _songfeil_globals['slang']) for x in wordlist])

    # f14: Number of words in uppercase (≥ 3 letters long)
    feats[14 - 1] = sum(map(lambda x: x[0] == x[0].upper() and len(x[0]) >= 3, wordlist))

    # f15: Average length of sentences, in tokens
    feats[15 - 1] = np.average(list(map(lambda x: len(x.strip().split()), comment.split('\n'))))

    # f16: Average length of tokens, excluding punctuation-only tokens, in characters
    is_non_punc = [(x[0] not in string.punctuation) for x in wordlist]
    token_len = np.array([len(x[0]) for x in wordlist])[is_non_punc]
    feats[16 - 1] = np.mean(token_len) if len(token_len) > 0 else 0

    # f17: Number of sentences
    feats[17 - 1] = len(comment.split('\n'))

    # f18: Average of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    bgl_score = list(map(lambda a: a if a is not None else _songfeil_globals['bglempty'], [_songfeil_globals['bgldict'].get(x[0]) for x in wordlist]))
    aoa_score = list(map(lambda x: float(x['aoa']), bgl_score))
    feats[18 - 1] = np.mean(aoa_score)

    # f19: Average of IMG from Bristol, Gilhooly, and Logie norms
    img_score = list(map(lambda x: float(x['img']), bgl_score))
    feats[19 - 1] = np.mean(img_score)

    # f20: Average of FAM from Bristol, Gilhooly, and Logie norms
    fam_score = list(map(lambda x: float(x['fam']), bgl_score))
    feats[20 - 1] = np.mean(fam_score)

    # 21. Standard deviation of AoA (100-700) from Bristol, Gilhooly, and Logie norms
    feats[21 - 1] = np.std(aoa_score)

    # 22. Standard deviation of IMG from Bristol, Gilhooly, and Logie norms
    feats[22 - 1] = np.std(img_score)

    # 23. Standard deviation of FAM from Bristol, Gilhooly, and Logie norms
    feats[23 - 1] = np.std(fam_score)

    # 24. Average of V.Mean.Sum from Warringer norms
    war_score = list(map(lambda a: a if a is not None else _songfeil_globals['vadempty'], [_songfeil_globals['vaddict'].get(x[0]) for x in wordlist]))
    v_score = list(map(lambda x: float(x['v']), war_score))
    feats[24 - 1] = np.mean(v_score)

    # 25. Average of A.Mean.Sum from Warringer norms
    a_score = list(map(lambda x: float(x['a']), war_score))
    feats[25 - 1] = np.mean(a_score)

    # 26. Average of D.Mean.Sum from Warringer norms
    d_score = list(map(lambda x: float(x['d']), war_score))
    feats[26 - 1] = np.mean(d_score)

    # 27. Standard deviation of V.Mean.Sum from Warringer norms
    feats[27 - 1] = np.std(v_score)

    # 28. Standard deviation of A.Mean.Sum from Warringer norms
    feats[28 - 1] = np.std(a_score)

    # 29. Standard deviation of D.Mean.Sum from Warringer norms
    feats[29 - 1] = np.std(d_score)

    return feats


def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1))

    # TODO: your code here
    # Prepare for data read
    featsdir = "/Users/florian_song/cs401data/feats"
    # featsdir = "/u/cs401/A1/feats"

    liwc = {
        'Left': _read_feats(featsdir + '/Left_IDs.txt', featsdir + '/Left_feats.dat.npy'),
        'Right': _read_feats(featsdir + '/Right_IDs.txt', featsdir + '/Right_feats.dat.npy'),
        'Center': _read_feats(featsdir + '/Center_IDs.txt', featsdir + '/Center_feats.dat.npy'),
        'Alt': _read_feats(featsdir + '/Alt_IDs.txt', featsdir + '/Alt_feats.dat.npy')
    }

    for idx, item in enumerate(data):
        extract1_feat = extract1(item['body'])
        classid = _cat_to_int(item['cat'])
        feats[idx] = np.concatenate([extract1_feat, np.array(classid)])
        feats[idx][29:173] = liwc[item['cat']][item['id']]

    np.savez_compressed( args.output, feats)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()


    main(args)

