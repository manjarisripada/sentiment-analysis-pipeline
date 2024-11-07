from sentiment_analyzer import SentimentAnalyzer
from data_visualizer import DataVisualizer
from utilities import Utility
from pathlib import Path
import pickle
import pandas as pd

def analyzeVisualize(sentiment):
    clf_names = sentiment.clf_names
    labels = ['negative', 'positive']
    visualizer = DataVisualizer()
    
    try:
        with open('pickled/metrics_dataframe.pickle', 'rb') as df:
            metrics_df = pickle.load(df)
        print("Metrics data loaded successfully.")
        
        # Prepare data for visualization
        metrics_df.rename(columns={
            "Accuracy": "value_Accuracy", 
            "Precision": "value_Precision", 
            "Recall": "value_Recall", 
            "F1-score": "value_F1-score", 
            "ROC AUC": "value_ROC AUC"
        }, inplace=True)
        metrics_df['id'] = metrics_df.index
        metrics_df_long = pd.wide_to_long(metrics_df, stubnames='value', i='id', j='id_m', sep='_', suffix=r'[a-zA-Z0-9_\- ]+')
        metrics_df_long['Metrics'] = metrics_df_long.index.get_level_values('id_m')

        # Add 'Classifier' column for each entry in metrics_df_long
        metrics_df_long['Classifier'] = [clf_names[i // len(metrics_df_long) * len(clf_names)] for i in range(len(metrics_df_long))]
        
        visualizer.plotClassifierPerformanceComparison(metrics_df_long, clf_names, 'Metrics Comparison')

    except FileNotFoundError:
        print("Metrics data file not found. Skipping metrics visualization.")

    # Additional visualizations or feature importances if applicable
    util = Utility()
    data = util.classifiers_vs_features()
    colors = ['blue', 'yellow', 'red', 'green']
    visualizer.plotClassifiersVsFeatures(data, clf_names, colors)

    # Display top features if classifier exists
    if hasattr(sentiment, 'clf') and sentiment.clf:
        top_features = util.show_top_features(sentiment.clf, n=30)
        print('The 30 most informative features for both positive and negative coefficients:\n')
        print(top_features)
    else:
        print("Classifier not available for top feature visualization.")



if __name__ == "__main__":
    do_pickle = True
    do_train_data = True
    do_fetch_data = True
    do_preprocess_data = True
    do_cross_validation_strategy = True
    do_holdout_strategy = True
    do_analyze_visualize = True

    Path('./pickled').mkdir(exist_ok=True)
    Path('./plots').mkdir(exist_ok=True)

    if do_fetch_data or do_preprocess_data or do_cross_validation_strategy or do_train_data:
        sentiment = SentimentAnalyzer()

    if do_fetch_data:
        sentiment.get_initial_data('product_reviews.json', do_pickle)

    if do_preprocess_data:
        reviews_df = pd.read_pickle('pickled/product_reviews.pickle')
        sentiment.preprocess_data(reviews_df, do_pickle)

    # if do_cross_validation_strategy or do_holdout_strategy:
    #     reviews_df_preprocessed = pd.read_pickle('pickled/product_reviews_preprocessed.pickle')
    #     print(reviews_df_preprocessed.isnull().values.sum())

    # if do_cross_validation_strategy:
    #     sentiment.crossValidationStrategy(reviews_df_preprocessed, do_pickle)

    # if do_holdout_strategy:
    #     sentiment.holdoutStrategy(reviews_df_preprocessed, do_pickle, do_train_data)
    # Train the model and evaluate it using split data
    if do_train_data:
        reviews_df_preprocessed = pd.read_pickle('pickled/product_reviews_preprocessed.pickle')
        split_data, X, y = sentiment.split_data(reviews_df_preprocessed, method='holdout')
        X_train, X_test, y_train, y_test = split_data
        trained_models = sentiment.train_model(X_train, y_train, sentiment.clf)
        
        # Evaluating and saving metrics
        predictions = [(name, model.predict(X_test)) for name, model in trained_models]
        metrics = sentiment.evaluate_model(y_test, predictions)
        if do_pickle:
            sentiment.save_to_pickle('pickled/metrics_dataframe.pickle', pd.DataFrame(metrics))
            sentiment.save_to_pickle('pickled/model_holdout.pickle', trained_models)

    if do_analyze_visualize:
        analyzeVisualize(sentiment)

    with open('pickled/model_holdout.pickle', 'rb') as model_holdout:
        model = pickle.load(model_holdout)

    model_svc = model[2][1]
    print('\nEnter your review:')
    user_review = input()
    verdict = 'Positive' if model_svc.predict([user_review]) == 1 else 'Negative'
    print(f'\nPredicted sentiment: {verdict}')
