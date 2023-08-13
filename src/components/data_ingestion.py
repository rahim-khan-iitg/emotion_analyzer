import os
import sys
from src.logger import logging
from src.exceptions import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join("artifacts",'train.csv')
    test_data_path=os.path.join("artifacts",'test.csv')
    raw_data_path=os.path.join("artifacts",'raw.csv')

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion initiated")
        try:
            df=pd.read_csv(os.path.join("notebooks/data",'IMDB_Dataset.csv'))
            df['sentiment']=df['sentiment'].apply(lambda x:1 if x=='positive' else 0)
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("train test split")
            train_set,test_set=train_test_split(df,test_size=0.05,random_state=45)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("data ingestion is completed")
            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
        except Exception as e:
            logging.info("error occured during data ingestion")
            raise CustomException(e,sys)