# Financial Sentiment Analysis with Natural Language Processing

This project applies Natural Language Processing (NLP) and machine learning techniques to classify financial tweets into three sentiment categories: **bullish**, **bearish**, and **neutral**.

The analysis evaluates multiple text representation techniques and machine learning models to determine which approaches perform best for financial sentiment classification. In particular, the project compares traditional machine learning methods with modern deep learning and transformer-based models.

The goal is to investigate how textual market sentiment can be extracted from financial news and social media data to support data-driven decision-making in finance.



# Authors

This project was developed as part of the **Text Mining course at NOVA Information Management School (NOVA IMS)**.

- Leopold Kaiser  
- Marta Andres Rodrigues



# Project Structure

```
NLP_18.ipynb
---
Main notebook containing the full NLP pipeline including preprocessing,
feature engineering, model training, and evaluation.

Report_18.pdf
---
Detailed project report explaining methodology, model comparison,
evaluation metrics, and results.
```



# Dataset

The dataset consists of financial tweets labeled according to sentiment:

- **Bullish**
- **Bearish**
- **Neutral**

Each observation contains:

- the tweet text  
- the corresponding sentiment label

The dataset is **imbalanced**, with neutral tweets being the most frequent class, followed by bullish and bearish tweets.



# Data Preprocessing

Several preprocessing techniques were applied to clean and standardize the textual data before model training.

Key preprocessing steps include:

> removing links and URLs  
> handling stock ticker symbols (e.g. `$AAPL`)  
> replacing price expressions with standardized tokens  
> lowercasing all text  
> removing punctuation  
> expanding contractions  
> removing stopwords (with custom adjustments)  
> lemmatization for grammatical normalization  

These steps reduce noise and help models better capture the semantic meaning of tweets.



# Feature Engineering

To convert text into numerical features, multiple NLP encoding techniques were implemented.

### Bag of Words (BoW)

Represents sentences as vectors of word frequencies without preserving word order.

Advantages:
> simple and computationally efficient

Limitations:
> ignores context and semantic relationships between words.



### TF-IDF

Term Frequency–Inverse Document Frequency adjusts word importance based on how informative a word is within the entire dataset.

Advantages:
> improves over BoW by reducing the weight of common words.



### Word2Vec

Word embeddings that capture semantic relationships between words using neural networks.

Advantages:
> similar words have similar vector representations.



### Sentence2Vec / Doc2Vec

Extends Word2Vec by learning vector representations for entire sentences.



### Sentence Transformers

Pre-trained transformer-based models generate dense embeddings capturing contextual meaning.

In this project the **`all-MiniLM-L6-v2`** model was used.



# Classification Models

Several machine learning models were trained and evaluated.

### K-Nearest Neighbors (KNN)

Classifies tweets based on similarity to neighboring samples in feature space.

Different values of *k* were tested.



### Logistic Regression

A probabilistic linear classifier using the softmax function for multi-class prediction.



### Multi-Layer Perceptron (MLP)

A feedforward neural network capable of modeling nonlinear relationships between features and sentiment labels.



### Naïve Bayes

A probabilistic classifier based on Bayes' theorem commonly used for text classification tasks.



### Long Short-Term Memory (LSTM)

A recurrent neural network designed to capture sequential patterns in text data.



### Fine-tuned FinBERT

A transformer-based language model pretrained on financial text and further fine-tuned on the project dataset.

FinBERT captures contextual relationships between words and significantly improves classification performance.



# Evaluation Metrics

Models were evaluated using several classification metrics:

> Accuracy  
> Precision  
> Recall  
> F1-score  

These metrics provide complementary perspectives on model performance, particularly important in imbalanced datasets.



# Results

The comparison across models shows clear performance differences between classical machine learning methods and transformer-based models.

Key findings:

> Logistic Regression combined with TF-IDF produced strong baseline results.

> Word2Vec-based approaches performed worse than expected due to the short length of tweets.

> Transformer-based embeddings improved overall classification performance.

The **best performing model** was the **fine-tuned FinBERT model**, achieving approximately:

> **Accuracy: 87%**

This demonstrates the effectiveness of transformer architectures for financial sentiment analysis.



# Requirements

The project uses standard Python libraries for machine learning and natural language processing:

```
numpy
pandas
scikit-learn
matplotlib
gensim
sentence-transformers
torch
transformers
```



# Reproducibility

All scripts and notebooks required to reproduce the analysis are included in this repository.

Running the notebook will reproduce the preprocessing pipeline, model training, and evaluation results described in the project report.



# Academic Context

This project was completed as part of the **Text Mining course in the Postgraduate Program in Data Science for Finance at NOVA IMS**.

The assignment focuses on applying modern NLP techniques to real-world financial data and comparing traditional machine learning methods with state-of-the-art transformer models.



# License

This repository is provided for academic and educational purposes.
