# Sentiment Analysis and Visualization Pipeline for Product Reviews

This repository provides a complete pipeline for performing sentiment analysis on product reviews using various classifiers. The project includes modules for data preprocessing, training, evaluation, and visualization of model performance.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing)
- [License](#license)

## Overview
This project uses several natural language processing (NLP) and machine learning techniques to classify product reviews as either positive or negative. The pipeline includes:
1. Text preprocessing with `nltk` to clean and tokenize review data.
2. Classifier training and feature selection using models like Multinomial Naive Bayes, Logistic Regression, Linear SVC, and Random Forest.
3. Visualization of model performance through confusion matrices, classification reports, and feature importance plots.

## Features
- **Data Preprocessing**: Tokenization, lemmatization, and stopword removal.
- **Classifier Training**: Support for multiple classifiers and feature selection.
- **Evaluation**: Metrics such as accuracy, precision, recall, F1-score, and ROC AUC.
- **Visualization**: Confusion matrix, classification report, and classifier performance comparison.

## Installation
1. Clone this repository:
    ```bash
    git clone https://github.com/yourusername/sentiment-analysis-pipeline.git
    cd sentiment-analysis-pipeline
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download NLTK data:
    ```python
    import nltk
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    ```

## Usage
1. Place your review data in `product_reviews.json`.
2. Run the pipeline from `main.py`:
    ```bash
    python main.py
    ```
3. Enter a review to predict sentiment when prompted.

## Modules

### 1. Preprocessing (`preprocessor.py`)
- `NltkPreprocessor`: Tokenizes and lemmatizes text, removing stopwords and punctuation.

### 2. Utilities (`utilities.py`)
- `Utility`: Trains classifiers on different feature sets and displays feature importance.

### 3. Visualization (`data_visualizer.py`)
- `DataVisualizer`: Generates visualizations for model evaluation, including confusion matrices, classification reports, and feature importance.

### 4. Main Script (`main.py`)
- Controls data loading, training, and visualizations, with options to select data processing and analysis steps.

### 5. Sentiment Analysis (`sentiment_analyzer.py`)
- `SentimentAnalyzer`: Manages data preprocessing, model training, evaluation, and saving results for text classification.

## Contributing
Contributions are welcome! Please submit a pull request with a description of your changes.

## License
This project is licensed under the MIT License.
