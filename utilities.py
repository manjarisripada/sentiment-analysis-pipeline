import pickle
from sentiment_analyzer import SentimentAnalyzer

class Utility:

    def __init__(self):
        self.sentiment = SentimentAnalyzer()
        self.clf = self.sentiment.clf

    def classifiers_vs_features(self):
        with open('pickled/features_train.pickle', 'rb') as features_train:
            X_train = pickle.load(features_train)
        with open('pickled/features_test.pickle', 'rb') as features_test:
            X_test = pickle.load(features_test)
        with open('pickled/labels_train.pickle', 'rb') as labels_train:
            y_train = pickle.load(labels_train)
        with open('pickled/labels_test.pickle', 'rb') as labels_test:
            y_test = pickle.load(labels_test)

        num_features = [10000, 50000, 100000, 500000, 1000000]
        acc = [[] for _ in range(len(self.clf))]

        for k in num_features:
            _, model = self.sentiment.trainData(X_train, y_train, self.clf, k)
            prediction = self.sentiment.predictData(X_test, model)
            clf_metrics = self.sentiment.evaluate(y_test, prediction)

            for j in range(len(self.clf)):
                acc[j].append(clf_metrics[0][j])

        return [{'x': num_features, 'y': acc[i]} for i in range(len(self.clf))]

    def show_top_features(self, pipeline, n=20):
        vectorizer = pipeline.named_steps['vect']
        clf = pipeline.named_steps['clf']
        feature_names = vectorizer.get_feature_names_out()

        coefs = sorted(zip(clf.coef_[0], feature_names), reverse=True)
        topn = zip(coefs[:n], coefs[-n:][::-1])

        return '\n'.join('{:0.4f}{: >25}    {:0.4f}{: >25}'.format(coef_p, feature_p, coef_n, feature_n) 
                         for (coef_p, feature_p), (coef_n, feature_n) in topn)
