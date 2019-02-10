import sys
import argparse
import os
import json
import html
import re
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])
indir = '/Users/florian_song/cs401data/data'

# indir = '/u/cs401/A1/data/'
# abbrevfile = '/u/cs401/Wordlists/abbrev.english'
# cliticsfile = '/u/cs401/Wordlists/clitics'
# stopwordsfile = '/u/cs401/Wordlists/StopWords'

def read_file(sourcefile):
    lines = []
    with open(sourcefile, 'r') as f:
        lines.extend(f.readlines())
    return set([line.strip() for line in lines])

def puncshelper(x):
    word = x.group()
    if word not in _songfeil_globals['abbrevs']:
        word = _songfeil_globals['pattern']['puncs'].sub(lambda x: " " + x.group() + " ", word)
    return word

def lemmahelper(token):
    if token.lemma_[0] != '-':
        return token.lemma_
    else:
        return token.lemma_ if '-' in token.text else token.text

def processclitics(clitics):
    result = set(map(lambda a: (a + r"\b") if re.match(r"^'\w+", a) else a, clitics))
    result = set(map(lambda a: (r"\b" + a) if re.match(r"\w+'$", a) else a, result))
    return result

_songfeil_globals = {
    'dirs': {
        'indir': '/Users/florian_song/cs401data/data',
        'abbrev': '/Users/florian_song/cs401data/Wordlists/abbrev.english',
        'clitics': '/Users/florian_song/cs401data/Wordlists/clitics',
        'stopwords': '/Users/florian_song/cs401data/Wordlists/StopWords'
    }
}

_songfeil_globals['abbrevs'] = read_file(_songfeil_globals['dirs']['abbrev']).union({'e.g.', 'i.e.', 'U.S.'})
_songfeil_globals['clitics'] = processclitics(read_file(_songfeil_globals['dirs']['clitics'])).union({r"n't\b"})
_songfeil_globals['stopwords'] = read_file(_songfeil_globals['dirs']['stopwords'])

_songfeil_globals['pattern'] = {
    'puncs': re.compile("[\!\"\#\$\%\&\,\(\)\*\+\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\\\]+"),
    'clitics': re.compile("(" + "|".join(_songfeil_globals['clitics']) + ")", re.IGNORECASE)
}


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = comment
    if 1 in steps:
        modComm = modComm.replace('\n', ' ')
    if 2 in steps:
        modComm = html.unescape(modComm)
    if 3 in steps:
        # Here I define website to be have at least one char following 'www' or 'http'
        # Since these are also words that people might use
        modComm = re.sub(r'\b(www|http)\S+(\b|)', '', modComm)
    if 4 in steps:
        modComm = re.sub(r'(?<!\S)\S+(?!\S)', puncshelper, modComm)
    if 5 in steps:
        modComm = _songfeil_globals['pattern']['clitics'].sub(lambda x: " " + x.group() + " ", modComm)
    if 6 in steps:
        utt = nlp(modComm)
        wordlist = [token.text + "/" + token.tag_ for token in utt if len(token.text.strip()) != 0]
        modComm = " ".join(wordlist)
    if 7 in steps:
        wordlist = list(filter(lambda x: x.rsplit('/', 1)[0].lower() not in _songfeil_globals['stopwords'], modComm.split()))
        modComm = " ".join(wordlist)
    if 8 in steps:
        wordlist = list(map(lambda x: x.rsplit('/', 1), modComm.split()))
        newlist = []

        sentence = " ".join(map(lambda x: x[0], wordlist))
        proc_sentence = nlp(sentence)

        if len(proc_sentence) == len(wordlist):
            for idx, item in enumerate(wordlist):
                newlist.append(lemmahelper(proc_sentence[idx]) + '/' + item[1])
        else:
            for idx, item in enumerate(wordlist):
                tag = nlp(item[0])
                if len(tag) == 1:
                    newlist.append(lemmahelper(tag[0]) + '/' + item[1])
                else:
                    newlist.append("/".join(item))

        modComm = " ".join(newlist)
    if 9 in steps:
        modComm = re.sub("(\.)\"? +[A-Z]", lambda x: x.group().replace(' ', '\n'), modComm)
        # Since ! and ? won't possibly show in the abbreviation
        modComm = re.sub("[(\?/\.)(\!/\.)] [^\?\!\.]", lambda x: x.group().replace(' ', '\n'), modComm)
    if 10 in steps:
        modComm = re.sub(r'\b\w+/', lambda x: x.group().lower(), modComm)

    return modComm

def main( args ):

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            input_len = len(data)
            start_pos = args.ID[0] % len(data)
            end_pos = start_pos + args.max

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...)
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'

            for idx in range(start_pos, end_pos):
                # if (idx - start_pos) % 100 == 0:
                #     print((idx - start_pos) / 100, " ")


                line = data[idx % input_len]
                j = json.loads(line)
                cp_args = ["id", "body"]

                nj = {}
                for arg in cp_args:
                    nj[arg] = j[arg]

                # Process comment & add cat
                nj["body"] = preproc1(nj["body"], range(1, 11))

                # print(idx, nj["id"])
                # print(nj["body"])
                # print("---------------------------------------------------")

                nj["cat"] = file
                allOutput.append(nj)

    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000, type=int)
    args = parser.parse_args()

    if (args.max > 200272):
        print( "Error: If you want to read more than 200,272 comments per file, you have to read them all." )
        sys.exit(1)

    main(args)
