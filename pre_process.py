import html2text

from stop_words import get_stop_words
stop_wordsFR = get_stop_words('french')

import spacy
nlp = spacy.load('fr_core_news_md')

# sind='!"#$%&\'()*+,-/:;.<=>?@[\\]^_`{|}~0123456789'
sind="!#$%&\(*+,-)/:;. <=>?@[\\]^_`{|}~0123456789'"


def process(CR):
    
    h = html2text.HTML2Text()
    CR=h.handle(CR).replace("\\n","\n") 
    CR=CR.replace("\n"," ")    
    CR_clean=nlp(CR)
    tok_end=[]
    if len(str(CR_clean))>150:
        for tok in CR_clean:
            tok_str=str(tok.lemma_.lower())
            if tok_str not in (stop_wordsFR) and tok.text not in '\n' and '.pdf' not in tok_str :
                for char in sind:
                    tok_str=tok_str.replace(char,"")
                # tok_end.append(tok_str.translate(str.maketrans("\n\t\r", "   ")))
                if len(tok_str)!=0:
                    tok_end.append(tok_str)
        return tok_end  
