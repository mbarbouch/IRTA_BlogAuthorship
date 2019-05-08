from classification import train_model
from data import load_corpus, from_xml_to_plain_text

if __name__ == '__main__':
    #from_xml_to_plain_text(lower_case=True, remove_punctuation=False, remove_stopwors=True)

    corpus = load_corpus("./data/plainblogs")
    print(len(corpus), corpus[0:10])
    train_model(corpus)
