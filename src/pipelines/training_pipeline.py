from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=="__main__":
    data_ingest=DataIngestion()
    train,test=data_ingest.initiate_data_ingestion()
    print(train,test)
    data_trans=DataTransformation()
    x_train,x_test,y_train,y_test,tokenizer_path=data_trans.initiate_data_transformation(train,test)
    print(tokenizer_path)
    trainer=ModelTrainer()
    trainer.initiate_model_training(x_train,x_test,y_train,y_test,tokenizer_path,10,batch_size=100)
    print("model training completed")