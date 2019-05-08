import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import NuSVC
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
# import pylab as pl
from sklearn.model_selection import train_test_split


# https://www.slideshare.net/PyData/authorship-attribution-forensic-linguistics-with-python-scikit-learn-pandas-kostas-perifanos
from helper import print_and_plot_confusion_matrix


def train_model(trainset):
    # create two blocks of features, word anc character ngrams, size of 2
    # we can also append here multiple other features in general
    word_vector = TfidfVectorizer(analyzer="word", ngram_range=(2, 2), binary=False, max_features=2000)
    char_vector = TfidfVectorizer(ngram_range=(2, 3), analyzer="char", binary=False, min_df=0, max_features=2000)

    # ur vectors are the feature union of word/char ngrams
    vectorizer = FeatureUnion([("chars", char_vector), ("words", word_vector)])

    # corpus is a list with the n-word chunks
    corpus = []
    # classes is the labels of each chunk
    classes = []

    # load training sets, for males & females

    for item in trainset:
        corpus.append(item['text'])
        classes.append(item['label'])

    print("num of training instances: ", len(classes))
    print("num of training classes: ", len(set(classes)))

    # fit the model of tfidf vectors for the coprus
    matrix = vectorizer.fit_transform(corpus)

    print("num of features: ", len(vectorizer.get_feature_names()))

    print("training model")
    X = matrix.toarray()
    y = np.asarray(classes)

    model = LinearSVC(loss='hinge', dual=True)
    # model = KNeighborsClassifier()

    # scores = cross_val_score(estimator=model,
    #                          X=matrix.toarray(),
    #                          y=np.asarray(classes), cv=10)
    #
    # # http://scikit-learn.org/stable/auto_examples/plot_confusion_matrix.html
    # print(scores)
    #
    # print("10-fold cross validation results:", "mean score = ", scores.mean(), "std=", scores.std(), ", num folds =",
    #       len(scores))
    #
    # model.fit(X=matrix.toarray(), y=np.asarray(classes))
    #
    # predicted = model.predict(matrix.toarray())
    # cm = confusion_matrix(classes, predicted)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    y_pred = model.fit(X_train, y_train).predict(X_test)

    # cm = confusion_matrix(y_test, y_pred)
    # print(cm)

    model_name = str(model).split('(')[0]
    unique_labels = np.unique(y_test)
    unique_labels = ['.'.join(label.split('.')[:4]) for label in unique_labels]
    print_and_plot_confusion_matrix(y_test, y_pred, unique_labels, True, model_name + " Confusion Matrix")

    print(classification_report(y_test, y_pred))
    print("Accuracy", accuracy_score(y_test, y_pred))

