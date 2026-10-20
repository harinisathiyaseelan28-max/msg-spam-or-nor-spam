
"""Train an LSTM SMS spam classifier and save model/tokenizer artifacts."""

from pathlib import Path
import io
import json
import zipfile

import requests
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_URL = (
    "https://archive.ics.uci.edu/static/public/"
    "228/sms+spam+collection.zip"
)

DATA_FILE = DATA_DIR / "SMSSpamCollection"

VOCAB_SIZE = 10000
MAX_LEN = 100
SEED = 42


# --------------------------------------------------
# 2. Download and load the dataset
# --------------------------------------------------

def get_dataset():
    if DATA_FILE.exists():
        print("Loading existing SMS dataset...")

        return pd.read_csv(
            DATA_FILE,
            sep="\t",
            header=None,
            names=["label", "message"],
            encoding="utf-8",
            dtype=str,
            keep_default_na=False,
        )

    print("Downloading SMS Spam Collection dataset...")

    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        member = next(
            (
                name
                for name in zf.namelist()
                if name.endswith("SMSSpamCollection")
            ),
            None,
        )

        if member is None:
            raise FileNotFoundError(
                "SMSSpamCollection was not found in the downloaded ZIP."
            )

        with zf.open(member) as source:
            df = pd.read_csv(
                source,
                sep="\t",
                header=None,
                names=["label", "message"],
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
            )

    df.to_csv(
        DATA_FILE,
        sep="\t",
        index=False,
        header=False,
    )

    print(f"Dataset saved to: {DATA_FILE}")

    return df


# --------------------------------------------------
# 3. Prepare and validate the data
# --------------------------------------------------

def main():
    tf.keras.utils.set_random_seed(SEED)

    print("\nLoading and preparing dataset...")

    df = get_dataset()

    df = df[["label", "message"]].copy()

    df["label"] = (
        df["label"].astype(str).str.strip().str.lower()
    )

    df["message"] = df["message"].astype(str).str.strip()

    df = df[df["label"].isin(["ham", "spam"])]

    df = df[df["message"] != ""]

    df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "The dataset is empty. Check data/SMSSpamCollection."
        )

    df["target"] = (df["label"] == "spam").astype("int32")

    if df["target"].nunique() != 2:
        raise ValueError(
            "The dataset must contain both ham and spam messages."
        )

    print(f"Total messages: {len(df)}")
    print(f"Normal messages: {(df['target'] == 0).sum()}")
    print(f"Spam messages: {(df['target'] == 1).sum()}")

    # Use a Python list for messages and a NumPy array for labels.
    messages = df["message"].tolist()
    targets = df["target"].to_numpy(dtype=np.int32)

    # Split the data into training and testing sets.
    train_texts, test_texts, y_train, y_test = train_test_split(
        messages,
        targets,
        test_size=0.2,
        random_state=SEED,
        stratify=targets,
    )

    print(f"\nTraining messages: {len(train_texts)}")
    print(f"Testing messages: {len(test_texts)}")


    # --------------------------------------------------
    # 4. Convert text into numerical sequences
    # --------------------------------------------------

    print("\nPreparing text tokenizer...")

    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>",
    )

    tokenizer.fit_on_texts(train_texts)

    X_train = pad_sequences(
        tokenizer.texts_to_sequences(train_texts),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    X_test = pad_sequences(
        tokenizer.texts_to_sequences(test_texts),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    y_train = np.asarray(y_train, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32)

    print(f"Training input shape: {X_train.shape}")
    print(f"Testing input shape: {X_test.shape}")


    # --------------------------------------------------
    # 5. Build the LSTM deep learning model
    # --------------------------------------------------

    print("\nBuilding LSTM model...")

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(MAX_LEN,)),

        tf.keras.layers.Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=64,
            mask_zero=True,
        ),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),

        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(
            16,
            activation="relu",
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            1,
            activation="sigmoid",
        ),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()


    # --------------------------------------------------
    # 6. Train the model
    # --------------------------------------------------

    print("\nStarting model training...")

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        epochs=8,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1,
    )


    # --------------------------------------------------
    # 7. Evaluate the trained model
    # --------------------------------------------------

    print("\nEvaluating model...")

    probabilities = model.predict(
        X_test,
        verbose=0,
    ).ravel()

    predictions = (probabilities >= 0.5).astype(np.int32)

    accuracy = accuracy_score(y_test, predictions)

    print(f"\nTest accuracy: {accuracy:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["ham", "spam"],
            digits=4,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")

    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


    # --------------------------------------------------
    # 8. Save the model and supporting files
    # --------------------------------------------------

    print("\nSaving model artifacts...")

    model.save(ARTIFACTS_DIR / "sms_spam_lstm.keras")

    with open(
        ARTIFACTS_DIR / "tokenizer.json",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(tokenizer.to_json())

    config = {
        "max_len": MAX_LEN,
        "vocab_size": VOCAB_SIZE,
        "threshold": 0.5,
    }

    with open(
        ARTIFACTS_DIR / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(config, file, indent=2)

    print("\nTraining completed successfully!")

    print(f"Model artifacts saved in:\n{ARTIFACTS_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nERROR: {type(error).__name__}: {error}")
        raise
