import os
import sys
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from dataclasses import dataclass
from src.exceptions import CustomException
from src.logger import logging
from src.utils import save_objects
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
@dataclass 
class DataTransformationConfig:
    tokenizer_path=os.path.join("artifacts","tokenizer.pkl")

class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config=DataTransformationConfig()
        self.max_length=100

    def remove_tags(self,text):
        tag_remover=re.compile(r'<[^>]+>')
        return tag_remover.sub(" ",text)
    def preprocess_data(self,sent):
        sentence=sent.lower()
        sentence=self.remove_tags(sentence)
        sentence=re.sub(r'[^a-zA-Z]',' ',sentence)
        sentence=re.sub(r'\s+[a-zA-Z]\s+',' ',sentence)
        sentence=re.sub(r'\s+'," ",sentence)
        # remove stopwords
        pattern=re.compile(r'\b('+r'|'.join(stopwords.words('english'))+r')\b\s*')
        sentence=pattern.sub(" ",sentence)
        return sentence
    def create_tokenizer(self,x_train):
        logging.info("creating word tokenizer")
        word_tokenizer=Tokenizer()
        word_tokenizer.fit_on_texts(x_train)
        logging.info("saving word tokenizer")
        save_objects(word_tokenizer,DataTransformationConfig().tokenizer_path)
        return word_tokenizer
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            logging.info("reading data in data frame")
            train_data=pd.read_csv(train_data_path)
            test_data=pd.read_csv(test_data_path)
            logging.info(f"train data head\n{train_data.head(2).to_string()}")
            logging.info(f"test data head\n{test_data.head(2).to_string()}")
            logging.info("cleaning data")
            train_text=train_data['review'].apply(self.preprocess_data)
            test_text=test_data['review'].apply(self.preprocess_data)
            tokenizer=self.create_tokenizer(train_text)
            x_train=tokenizer.texts_to_sequences(train_text)
            x_test=tokenizer.texts_to_sequences(test_text)
            x_train=pad_sequences(x_train,padding='post',maxlen=self.max_length)
            x_test=pad_sequences(x_test,padding='post',maxlen=self.max_length)
            y_train=train_data['sentiment']
            y_test=test_data['sentiment']
            return(
                x_train,x_test,y_train,y_test,
                DataTransformationConfig().tokenizer_path
            )
        except Exception as e:
            logging.info("error occured dunring data transformation")
            raise CustomException(e,sys)

    
