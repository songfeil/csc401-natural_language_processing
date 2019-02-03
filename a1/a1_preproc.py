import sys
import argparse
import os
import json
import html
import re
import spacy

nlp = spacy.load('en', disable=['parser', 'ner'])

indir = '/u/cs401/A1/data/'
abbrevfile = '/u/cs401/Wordlists/abbrev.english'
cliticsfile = '/u/cs401/Wordlists/clitics'
stopwordsfile = '/u/cs401/Wordlists/StopWords'

abbrevs = ['e.g.', 'i.e.', 'U.S.']
clitics = ["n't"]
stopwords = []

pattern = {}
pattern['puncs'] = re.compile("[\!\"\#\$\%\&\,\(\)\*\+\-\.\/\:\;\<\=\>\?\@\[\]\^\_\`\{\|\}\~\\\]+")
pattern['clitics'] = re.compile("(" + "|".join(clitics) + ")")

def read_file(sourcefile, target):
    with open(sourcefile, 'r') as f:
        lines = f.readlines()
        target.extend([line.strip() for line in lines])

def puncshelper(x):
    word = x.group()
    if word not in abbrevs:
        word = pattern['puncs'].sub(lambda x: " " + x.group() + " ", word)
    return word

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
        read_file(abbrevfile, abbrevs)
        modComm = re.sub(r'(?<!\S)\S+(?!\S)', puncshelper, modComm)
    if 5 in steps:
        read_file(cliticsfile, clitics)
        clitics.append(r"'")

        # Add '\b' for word boundry so that does not match irrelevant words
        for i in range(len(clitics)):
            if clitics[i][0] == "'" and len(clitics[i]) > 1:
                clitics[i] += r'\b'
            elif clitics[i][-1] == "'" and len(clitics[i]) > 1:
                clitics[i] = r'\b' + clitics[i]

        pattern['clitics'] = re.compile("(" + "|".join(clitics) + ")")
        modComm = pattern['clitics'].sub(lambda x: " " + x.group() + " ", modComm)
    if 6 in steps:
        utt = nlp(modComm)
        wordlist = [token.text + "/" + token.tag_ for token in utt if len(token.text.strip()) != 0]
        modComm = " ".join(wordlist)
    if 7 in steps:
        read_file(stopwordsfile, stopwords)
        wordlist = list(filter(lambda x: x.split('/')[0] not in stopwords, modComm.split()))
        modComm = " ".join(wordlist)
    if 8 in steps:
        wordlist = list(map(lambda x: x.split('/')[0], modComm.split()))
        utt = nlp(" ".join(wordlist))
        newlist = []
        for token in utt:
            if len(token.text.strip()) != 0:
                if token.lemma_[0] != '-':
                    newlist.append(token.lemma_ + "/" + token.tag_)
                else:
                    newlist.append(token.text + "/" + token.tag_)
        modComm = " ".join(newlist)
    if 9 in steps:
        modComm = re.sub("(\.) [A-Z]", lambda x : x.group().replace('. ', '.\n'), modComm)
    if 10 in steps:
        wordlist = []
        for word in modComm.split():
            w = word.split('/')
            w[0] = w[0].lower()
            wordlist.append("/".join(w))
        modComm = " ".join(wordlist)

    print(modComm)
    print("---------------------------------------------------")
        
    return modComm

def main( args ):
    # Read in abbrevs & clitics from file

    allOutput = []
    for subdir, dirs, files in os.walk(indir):
        for file in files:
            fullFile = os.path.join(subdir, file)
            print( "Processing " + fullFile)

            data = json.load(open(fullFile))

            # TODO: select appropriate args.max lines
            # TODO: read those lines with something like `j = json.loads(line)`
            # TODO: choose to retain fields from those lines that are relevant to you
            # TODO: add a field to each selected line called 'cat' with the value of 'file' (e.g., 'Alt', 'Right', ...) 
            # TODO: process the body field (j['body']) with preproc1(...) using default for `steps` argument
            # TODO: replace the 'body' field with the processed text
            # TODO: append the result to 'allOutput'

            for line in data:
                j = json.loads(line)
                cp_args = ["id", "body"]

                nj = {}
                for arg in cp_args:
                    nj[arg] = j[arg]

                # Process comment & add cat
                nj["body"] = preproc1(nj["body"], range(1, 11))
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
