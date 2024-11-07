import time
import ast
import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from preprocessor import NltkPreprocessor

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import nltk

# Download necessary NLTK resources
nltk.download('averaged_perceptron_tagger')

class SentimentAnalyzer:
    def __init__(self):
        # Initialize classifiers
        self.clf = [
            ('MNB', MultinomialNB(alpha=1.0, fit_prior=False)),
            ('LR', LogisticRegression(C=5.0, penalty='l2', solver='liblinear', max_iter=100, dual=True)),
            ('SVM', LinearSVC(C=0.55, penalty='l2', max_iter=1000, dual=True)),
            ('RF', RandomForestClassifier(n_jobs=-1, n_estimators=100, min_samples_split=40, max_depth=90, min_samples_leaf=3))
        ]
        self.clf_names = ['Multinomial NB', 'Logistic Regression', 'Linear SVC', 'Random Forest']

    def get_initial_data(self, data_file, do_pickle):
        """Load and optionally pickle initial data from file."""
        print('Fetching initial data...')
        start_time = time.time()

        # Read data
        with open(data_file, 'r') as file_handler:
            df = {i: ast.literal_eval(line) for i, line in enumerate(file_handler)}

        reviews_df = pd.DataFrame.from_dict(df, orient='index')
        if do_pickle:
            reviews_df.to_pickle('pickled/product_reviews.pickle')

        print(f'Data fetched in {round(time.time() - start_time, 3)}s\n')
        return reviews_df

    def preprocess_data(self, reviews_df, do_pickle):
        """Preprocess reviews and optionally pickle the result."""
        print('Preprocessing data...')
        start_time = time.time()

        # Convert 'reviewRating' to numeric, coercing errors to NaN (useful if there are non-numeric values)
        reviews_df['reviewRating'] = pd.to_numeric(reviews_df['reviewRating'], errors='coerce')

        # Drop rows where 'reviewRating' is NaN (optional, if you want to remove them)
        reviews_df = reviews_df.dropna(subset=['reviewRating'])

        # Ignore neutral 3-star reviews
        reviews_df = reviews_df[reviews_df['reviewRating'] != 3]

        # Assign sentiment based on 'reviewRating'
        reviews_df = reviews_df.assign(sentiment=(reviews_df['reviewRating'] >= 4).astype(int))

        # Tokenize in parallel
        nltk_preprocessor = NltkPreprocessor()
        with mp.Pool() as pool:
            reviews_df['cleaned'] = pool.map(nltk_preprocessor.tokenize, reviews_df['reviewText'])

        if do_pickle:
            reviews_df.to_pickle('pickled/product_reviews_preprocessed.pickle')

        print(f'Data preprocessed in {round(time.time() - start_time, 3)}s\n')
        return reviews_df


    def split_data(self, reviews_df, method='holdout', k_folds=5):
        """Split data into training and test sets."""
        print(f'Splitting data using {method.capitalize()} strategy...')
        start_time = time.time()

        X, y = reviews_df['cleaned'].values, reviews_df['sentiment'].values
        if method == 'holdout':
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
            split_data = (X_train, X_test, y_train, y_test)
        else:
            kf = KFold(n_splits=k_folds, random_state=42, shuffle=True)
            split_data = [(train_idx, test_idx) for train_idx, test_idx in kf.split(X, y)]

        print(f'Data split completed in {round(time.time() - start_time, 3)}s\n')
        return split_data, X, y

    def train_model(self, X_train, y_train, classifier, num_features=1000000):
        """Train the classifier pipeline."""
        pipeline = []
        trained_models = []

        # Define pipeline steps
        steps = [
            ('vect', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=False)),
            ('select_best', SelectKBest(chi2, k=num_features))
        ]

        for name, clf in classifier:
            print(f'Training {name}...')
            start_time = time.time()
            
            pl = Pipeline(steps + [('clf', clf)])
            trained_model = pl.fit(X_train, y_train)
            trained_models.append((name, trained_model))
            
            print(f'{name} trained in {round(time.time() - start_time, 3)}s\n')

        return trained_models

    def evaluate_model(self, y_test, predictions):
        """Evaluate classifiers on test data."""
        metrics = {
            'Accuracy': [], 'Precision': [], 'Recall': [], 
            'F1': [], 'ROC AUC': [], 'Confusion Matrix': [], 'Classification Report': []
        }

        for name, pred in predictions:
            print(f'Evaluating {name}...')
            metrics['Accuracy'].append(accuracy_score(y_test, pred))
            metrics['Precision'].append(precision_score(y_test, pred))
            metrics['Recall'].append(recall_score(y_test, pred))
            metrics['F1'].append(f1_score(y_test, pred))
            metrics['ROC AUC'].append(roc_auc_score(y_test, pred))
            metrics['Confusion Matrix'].append(confusion_matrix(y_test, pred))
            metrics['Classification Report'].append(classification_report(y_test, pred, target_names=['negative', 'positive']))
            
        return metrics

    def save_to_pickle(self, filename, data):
        """Save data to a pickle file."""
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_pickle(self, filename):
        """Load data from a pickle file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)
