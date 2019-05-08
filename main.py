from classification import train_model
from data import load_corpus, from_xml_to_plain_text

if __name__ == '__main__':
    # from_xml_to_plain_text()

    corpus = load_corpus("./data/plainblogs")
    print(len(corpus), corpus[0:10])
    train_model(corpus)
