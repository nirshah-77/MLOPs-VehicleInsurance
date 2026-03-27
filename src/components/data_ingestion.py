import os
import sys

from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import Proj1Data

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    # bringing the data from mongoDB.
    def export_data_into_feature_store(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file
        
        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info(f"Exporting data from mongodb")
            my_data = Proj1Data()  #brings the data from proj1 file and this proj1 files establishes mongoDB connection first via a different file
            dataframe = my_data.export_collection_as_dataframe(collection_name=
                                                                   self.data_ingestion_config.collection_name)
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path  = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info(f"Saving exported data into feature store file path: {feature_store_file_path}")
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            return dataframe

        except Exception as e:
            raise MyException(e,sys)

    def export_data_into_feature_store_from_dir(self)->DataFrame:
        """
        Method Name :   export_data_into_feature_store_from_dir
        Description :   This method reads data from an existing feature store file if it exists, bypassing MongoDB.
                        It searches the 'artifact' directory for the latest 'data_ingestion/feature_store/data.csv'.
        Output      :   data is returned as artifact of data ingestion components
        """
        try:
            import os
            import glob
            import pandas as pd
            
            logging.info("Checking for existing data ingestion artifacts...")
            
            artifact_dir = "artifact"
            if os.path.exists(artifact_dir):
                search_pattern = os.path.join(artifact_dir, "*", "data_ingestion", "feature_store", "data.csv")
                existing_files = glob.glob(search_pattern)
                
                if existing_files:
                    latest_file = max(existing_files, key=os.path.getmtime)
                    logging.info(f"Found existing feature store at: {latest_file}. Reading it...")
                    
                    dataframe = pd.read_csv(latest_file)
                    logging.info(f"Shape of dataframe from existing artifact: {dataframe.shape}")
                    
                    current_feature_store_path = self.data_ingestion_config.feature_store_file_path
                    os.makedirs(os.path.dirname(current_feature_store_path), exist_ok=True)
                    dataframe.to_csv(current_feature_store_path, index=False, header=True)
                    
                    return dataframe

            logging.info("No existing feature store found. Fetching from MongoDB.")
            return self.export_data_into_feature_store()

        except Exception as e:
            raise MyException(e, sys)

    # applies train test split and stores in the given loication and this output is called artifacts that will be used in the future components.
    def split_data_as_train_test(self,dataframe: DataFrame) ->None:
        """
        Method Name :   split_data_as_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio 
        
        Output      :   Folder is created in s3 bucket
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered split_data_as_train_test method of Data_Ingestion class")

        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise MyException(e, sys) from e

    # this does the real work. calls the above functions (connect with mongoDB and split train and test and then store it at a particular location)
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion components of training pipeline 
        
        Output      :   train set and test set are returned as the artifacts of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Entered initiate_data_ingestion method of Data_Ingestion class")

        try:
            # dataframe = self.export_data_into_feature_store()
            dataframe = self.export_data_into_feature_store_from_dir()

            logging.info("Got the data from mongodb")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train test split on the dataset")

            logging.info(
                "Exited initiate_data_ingestion method of Data_Ingestion class"
            )

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path,
            test_file_path=self.data_ingestion_config.testing_file_path)
            
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e, sys) from e