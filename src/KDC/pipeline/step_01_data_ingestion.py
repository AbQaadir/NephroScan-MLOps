from KDC import logger
from KDC.components.data_ingestion import DataIngestion
from KDC.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion step"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info("Data Ingestion Pipeline started")
        logger.info(f"{STAGE_NAME} started.")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"{STAGE_NAME} completed.")
    except Exception as e:
        logger.exception(e)
        raise e
