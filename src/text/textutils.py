import fitz
import re


def load_vocab():
    with open('./src/es.txt') as file:
        lines = file.readlines()
    return [line.strip().lower() for line in lines]

VOCAB = load_vocab()

def post_process(text):
    spplited = re.sub(r'[^\w\s]|_', ' ', text).split()
    newtext = []
    skip = False
    length = len(spplited)
    i = 0
    while i < length - 1:
        word = spplited[i]
        next_word = spplited[i+1]
        comb = word + next_word
        if comb.lower() in VOCAB and len(word) >= 2 and len(comb) > 4:
            newtext.append(comb)
            skip = True
        else:
            newtext.append(word)
        i += 1
        if skip:
            i += 1
            skip = False

    if i == length - 1:  # Add the last word if not skipped
        newtext.append(spplited[-1])

    return " ".join(newtext)

def extract_text_with_position(fname, page_n, image, max_x, max_y, x, y, x2, y2):
    x, y, x2, y2 = x / image.shape[1], y/image.shape[0], x2/image.shape[1], y2/image.shape[0]
    for n, page in enumerate(fitz.open(fname)):

        if (n==int(page_n)): 
            text = page.get_textbox([max_x * x, max_y * y, max_x * x2, max_y * y2])
            text = post_process(text.replace('-', ''))          
            return text