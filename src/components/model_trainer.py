import os
import sys
import numpy as np
from src.logger import logging
from src.exceptions import CustomException
from src.utils import save_objects,load_object
from dataclasses import dataclass

from keras.layers import Dense,Embedding,LSTM
from keras.models import Sequential
import keras


@dataclass 
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.h5")

class ModelTrainer:
    def __init__(self) -> None:
        self.trained_model_config=ModelTrainerConfig()

    def get_embeddings(self,tokenizer_path):
        logging.info("making embedding matrix")
        try:
            embedding_dict=dict()
            logging.info("reading glove")
            glove_file=open(os.path.join("notebooks/data","glove.6B.100d.txt"),encoding='utf8')
            for line in glove_file:
                records=line.split()
                word=records[0]
                vector_dimensions=np.asarray(records[1:],dtype='float32')
                embedding_dict[word]=vector_dimensions
            glove_file.close()
            tokenizer=load_object(tokenizer_path)
            vocab_length=len(tokenizer.index_word)+1
            embedding_matrix=np.zeros((vocab_length,100))
            for word,index in tokenizer.word_index.items():
                embedding_vector=embedding_dict.get(word)
                if embedding_vector is not None:
                    embedding_matrix[index]=embedding_vector
            logging.info("embedding matrix obtained successfully")
            return embedding_matrix

        except Exception as e:
            logging.info("error occured during making of embedding matrix")
            raise CustomException(e,sys)
        
    def get_model(self,tokenizer_path):
        embedding_matrix=self.get_embeddings(tokenizer_path)
        tokenizer=load_object(tokenizer_path)
        vocab_length=len(tokenizer.word_index)+1
        lstm_model=Sequential()
        embedding_layer=Embedding(input_dim=vocab_length,output_dim=100,weights=[embedding_matrix],input_length=100,trainable=False)
        lstm_model.add(embedding_layer)
        lstm_model.add(LSTM(128))
        lstm_model.add(Dense(1,activation='sigmoid'))
        lstm_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
        return lstm_model
    def initiate_model_training(self,x_train,x_test,y_train,y_test,tokenizer_path,epochs,batch_size):
        logging.info("model training initiated")
        try:
            logging.info("getting model")
            model=self.get_model(tokenizer_path)
            logging.info("fitting the model")
            model.fit(x_train,y_train,batch_size=batch_size,verbose=2,epochs=epochs)
            logging.info("model fitting completed")
            logging.info('saving the model.h5')
            model.save(self.trained_model_config.trained_model_path)
            logging.info("testing the model performance")
            model.evaluate(x_test,y_test,verbose=2)
            return self.trained_model_config.trained_model_path
        except Exception as e:
            logging.info("error occured during model training")
            raise CustomException(e,sys)
