# SMS and Message Spam Detection Using Deep Learning (LSTM)

Beginner-friendly mini project using Python, TensorFlow/Keras, and the UCI SMS Spam Collection dataset.

## Files
- `app.py`: Streamlit web app for predicting spam/not spam.
- `train_model.py`: Downloads the dataset, trains an LSTM model, evaluates it, and saves model/tokenizer files.
- `requirements.txt`: Python dependencies.
- `data/README.md`: Dataset notes.

## Quick start (Google Colab)
1. Upload `train_model.py` and run its cells in a notebook, or run it as a script in an environment with internet access.
2. Install packages:
   `pip install -r requirements.txt`
3. Train:
   `python train_model.py`
4. Start the web app:
   `streamlit run app.py`

Training downloads the UCI dataset from its official repository. If the download is unavailable, download the SMS Spam Collection zip from:
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
and place/extract `SMSSpamCollection` at `data/SMSSpamCollection`.

## Notes
- Labels are `ham` (not spam) and `spam`.
- This is an educational classifier, not a guarantee that a message is safe.
- Do not enter private messages or sensitive personal information into a demo app.
