# Summative Lab: Analytics Firm Clients (Amazon Reviews)

A data science project that uses one dataset of Amazon customer reviews for three tasks: text analysis (NLP), time series forecasting, and a neural network that predicts ratings.

## Files

| File | Description |
|---|---|
| `C09_M08.ipynb` | The notebook with all three parts |
| `amazon_reviews_lab.csv` | The dataset (997 reviews, 9 columns) |

## Dataset

Main columns used:
- `review_text`: the review
- `rating`: 1 to 5 stars
- `timestamp`: date of the review
- `sentiment_label`: positive, neutral or negative

Most reviews are positive (580 of 997 are 5 stars).

## How to Run (Google Colab)

1. Open `C09_M08.ipynb` in Google Colab.
2. Run the first code cell and upload `amazon_reviews_lab.csv` when asked.
3. Run the rest of the cells in order.

Colab already has the libraries installed: pandas, numpy, matplotlib, seaborn, nltk, statsmodels, scikit-learn and TensorFlow.

## What Was Done

### Part 1: NLP
- Explored the data (shape, ratings, review length)
- Built a `preprocess_text` function: lowercase, remove HTML tags, URLs, punctuation and numbers, tokenize, remove stopwords (kept "no", "not", "nor"), lemmatize with POS tags
- Counted the most common words and bigrams
- Compared words in positive and negative reviews

### Part 2: Time Series
- Resampled the reviews by month (count and average rating)
- Used January 2010 to June 2014 because there are almost no reviews before 2010
- Tested stationarity (rolling statistics and ADF test), applied log and differencing, and decomposed the series
- Compared six ARIMA models by AIC, BIC, MAE and RMSE, and forecasted the last 12 months

### Part 3: Neural Network
- Turned review text into TF-IDF features (2,000 words)
- Built a network with two hidden layers (128 and 64 neurons), dropout and a softmax output
- Compared three settings (Adam with dropout, Adam without dropout, RMSprop with dropout)
- Evaluated with a classification report and confusion matrix, and tried class weights

## Key Results

| Part | Result |
|---|---|
| NLP | Reviews are mostly about Nook and Kindle devices. Negative reviews mention customer service, credit card and "not work". Positive reviews mention "great", "easy" and "love". |
| Time series | Reviews grew from 131 (2010) to 325 (2013). The average rating rose from 3.91 to 4.30. Best model by AIC: ARIMA(1,1,1), test MAE 6.43, RMSE 8.04, MAPE 27.6%. |
| Neural network | Test accuracy 0.59 (always guessing 5 stars gives 0.58). Macro F1 0.18. Class weights raised macro F1 to 0.25 and lowered accuracy to 0.48. |

## Limitations

- The ARIMA forecast is about as good as a simple baseline (MAE 6.38), and there are only 54 months of data.
- The neural network mostly predicts 5 stars, and the 1, 2 and 3 star ratings are hard to predict. More data, or grouping ratings into negative, neutral and positive, would likely help.

## Libraries

Python, pandas, numpy, matplotlib, seaborn, NLTK, statsmodels, scikit-learn, TensorFlow/Keras
