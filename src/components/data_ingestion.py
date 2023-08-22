import sys
import os
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformaion
from src.components.data_transformation import DataTransformaionConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# this is important

# any input that we need for data ingestin, we give it here
# output that we will also see
# with dataclass decorator you are able to directly define your class variables, without init.
@dataclass
class DataIngestionConfig:
    # this is the path, the output files will be also saved in this path.
    # for training and testing data path
    # we can also create another config folder for this, but fine for now
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")

class DataIngestion:
    def __init__(self) -> None:
        # when you call this, last three test, train and raw will be created.
        self.ingestion_config = DataIngestionConfig() 

    def initiate_data_ingestion(self):
        logging.info("entered the data ingestion method or component")
        try:
            #here you can replace with db or apit
            df = pd.read_csv("Notebook\Data\stud.csv")
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size = 0.2, random_state = 42)

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info("Ingestion of the data is started")

            return(
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


#if __name__=="__main__":
    #obj = DataIngestion()
    #obj.initiate_data_ingestion()


# to test data transformation
#if __name__=="__main__":
    #obj = DataIngestion()
    #train_data, test_data = obj.initiate_data_ingestion()

    #data_transfromation = DataTransformaion()
    #data_transfromation.initiate_data_transformation(train_data, test_data)



if __name__=="__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transfromation = DataTransformaion()
    train_arr, test_arr, _ = data_transfromation.initiate_data_transformation(train_data, test_data)
    modeltrainer= ModelTrainer()
    print(modeltrainer.initaite_model_trainer(train_arr, test_arr))