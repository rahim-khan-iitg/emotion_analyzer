import os
import sys
from src.logger import logging
from src.exceptions import CustomException
from src.utils import load_object
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from src.components.data_transformation import DataTransformation
import numpy as np

class PredictPipeline:
    def __init__(self) -> None:
        pass
    def Predict(self,text):
        logging.info("predicting for new input")
        try:
            model=load_model(os.path.join("artifacts","model.h5"))
            tokenizer=load_object(os.path.join("artifacts","tokenizer.pkl"))
            pre_process_data=DataTransformation().preprocess_data(text)
            tokenized_text=tokenizer.texts_to_sequences([pre_process_data])
            padded_sequence=pad_sequences(tokenized_text,padding='post',maxlen=100)
            pred=model.predict(padded_sequence)

            return pred
        except Exception as e:
            logging.info("error occured during prediction")
            raise CustomException(e,sys)

