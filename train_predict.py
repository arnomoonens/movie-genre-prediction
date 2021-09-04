"""Train and predict on the given data."""

from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from preprocessing import clean_data

def train(df: pd.DataFrame) -> Tuple[MultiLabelBinarizer, Pipeline]:
    """Train a predictive model to rank movie genres based on their synopsis."""
    print(df)

    # Data preprocessing
    df.genres = df.genres.apply(lambda x: x.split(" "))  # Convert string of genres to list of genres per movie
    df["synopsis"] = clean_data(df["synopsis"])

    # Extract input and output
    X = df[["synopsis", "year"]].to_numpy()
    y = df.genres.to_list()

    # Transforms genres of movie to list of 1's and 0's:
    # For each genre, 1 if movie has it, 0 if not.
    multilabel_binarizer = MultiLabelBinarizer()
    y_learner = multilabel_binarizer.fit_transform(y)

    # Pipeline to fit, transform and predict on input data
    pipe = Pipeline((
        # Transform text to numerical features and concatenate one-hot encoding of year
        ("transformer", ColumnTransformer([("text", TfidfVectorizer(sublinear_tf=True,
                                                                    ngram_range=(1, 2)), 0),
                                           ("year", OneHotEncoder(handle_unknown="ignore"), [1])])),
        # Multi-label Logistic Regression classifier
        ("clf", OneVsRestClassifier(LogisticRegression(C=20, solver="sag", max_iter=300),
                                    n_jobs=-1))))

    pipe.fit(X, y_learner)  # Learn model
    return multilabel_binarizer, pipe


def predict(multilabel_binarizer: MultiLabelBinarizer, pipe: Pipeline, df: pd.DataFrame) -> PlainTextResponse:
    """Predict the movie genres for each synopsis in the given file."""
    # Load and clean data like in train function
    df["synopsis"] = clean_data(df["synopsis"])
    X = df[["synopsis", "year"]].to_numpy()

    classes = multilabel_binarizer.classes_
    probabilities = pipe.predict_proba(X)  # Predict probability of each genre belonging to movie, per movie

    # Get indices of the genres with the 5 highest probabilities, per sample
    genres_ids = np.argsort(-probabilities, axis=1)[:, :5]
    output_genres_list = classes[genres_ids]  # Go from indices to list of genres per sample
    output_genres = [" ".join(genres) for genres in output_genres_list]  # Now go to string per sample

    # Make DataFrame to send back (as csv)
    result_df = pd.DataFrame({"movie_id": df.movie_id,
                              "genres": output_genres})

    return result_df
