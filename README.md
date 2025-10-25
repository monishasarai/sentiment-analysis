# sentiment-analysis

Sentiment Analysis of Tweets using RNN and LSTM.

This project implements and compares two deep learning models, a Simple Recurrent Neural Network (SimpleRNN) and a Long Short-Term Memory (LSTM) network, to classify the sentiment of tweets into three categories: **positive, negative, or neutral**.

The notebook is structured into two main parts:

* **Model Training:** Preprocesses the text data, builds, trains, and evaluates the RNN and LSTM models. The best-performing model (LSTM) is saved.
* **Sentiment Prediction:** Loads the saved LSTM model to perform sentiment prediction on new, user-provided text.

## 📊 Dataset

The model is trained on the `Tweets_Data.csv` dataset. This dataset contains tweets along with their corresponding sentiment.

* **`text`**: The full text of the tweet.
* **`sentiment`**: The target label (positive, negative, neutral).

## ⚙️ Methodology

The project follows a standard natural language processing (NLP) workflow:

### Data Loading & Cleaning
The `Tweets_Data.csv` file is loaded into a pandas DataFrame.

### Text Preprocessing
* **Cleaning**: Non-alphabetic characters are removed, and the text is converted to lowercase.
* **Stop Word Removal**: Common English stop words (e.g., "the", "a", "is") are removed using NLTK's `stopwords` corpus.
* **Lemmatization**: Words are reduced to their base or dictionary form (e.g., "going" -> "go") using NLTK's `WordNetLemmatizer`.

### Feature Engineering
* **Tokenization**: The cleaned text is converted into sequences of integers using `tensorflow.keras.preprocessing.text.Tokenizer`, with a vocabulary size of 5000 words.
* **Padding**: All sequences are padded to a uniform length (`max_len`) to ensure consistent input size for the models.
* **Label Encoding**: The categorical sentiment labels ('negative', 'neutral', 'positive') are one-hot encoded into numerical format.

### Model Building
Two sequential models are constructed using Keras:

* **SimpleRNN Model**: `Embedding` -> `SimpleRNN` -> `Dropout` -> `Dense` (output)
* **LSTM Model**: `Embedding` -> `LSTM` -> `Dropout` -> `Dense` (output)

Both models use `categorical_crossentropy` as the loss function and the `adam` optimizer.

### Training & Evaluation
* The data is split into 80% for training and 20% for validation.
* The models are trained for 5 epochs with a batch size of 64. `EarlyStopping` is used to prevent overfitting.
* The performance is evaluated using a classification report and a confusion matrix.

## 📈 Results

The project compares the performance of the SimpleRNN and LSTM models. The LSTM model demonstrated superior performance and was chosen as the final model.

| Model      | Accuracy | Key F1-Scores                                  |
| :--------- | :------: | :--------------------------------------------- |
| SimpleRNN  |  ~65%    | 0.61 (negative), 0.62 (neutral), 0.71 (positive) |
| **LSTM** |  **~69%**| **0.66 (negative), 0.67 (neutral), 0.75 (positive)** |
