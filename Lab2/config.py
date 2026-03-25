import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)


class Config:
    """Load configuration from environment variables"""
    
    @staticmethod
    def get(key, default=None, required=False):
        """Get environment variable or use default"""
        value = os.environ.get(key, default)
        if required and value is None:
            raise ValueError(f"Missing required config: {key}")
        return value
    
    @staticmethod
    def get_int(key, default=None, required=False):
        """Get as integer"""
        value = Config.get(key, default, required)
        return int(value) if value is not None else value
    
    @staticmethod
    def get_float(key, default=None, required=False):
        """Get as float"""
        value = Config.get(key, default, required)
        return float(value) if value is not None else value
    
    @staticmethod
    def get_bool(key, default=False, required=False):
        """Get boolean environment variable"""
        value = Config.get(key, str(default), required).lower()
        return value in ('true', '1', 'yes', 'on')


# Application configuration
class AppConfig:
    """Main application configuration"""
    
    # Flask Configuration
    FLASK_ENV = Config.get('FLASK_ENV', 'production')
    PORT = Config.get_int('PORT', 5000)
    DEBUG = FLASK_ENV == 'development'
    
    # Model Configuration
    MODEL_NAME = Config.get('MODEL_NAME', 'unet-house-segmentation')
    MODEL_PATH = Config.get('MODEL_PATH', 'models/segmentation_model.keras')
    
    # Dataset Configuration
    DATASET_PATH = Config.get('DATASET_PATH', 'data/satellite_dataset')
    
    # Training Configuration
    EPOCHS = Config.get_int('EPOCHS', 50)
    BATCH_SIZE = Config.get_int('BATCH_SIZE', 32)
    LEARNING_RATE = Config.get_float('LEARNING_RATE', 0.001)
    VALIDATION_SPLIT = Config.get_float('VALIDATION_SPLIT', 0.2)
    
    # Logging
    LOG_LEVEL = Config.get('LOG_LEVEL', 'INFO')
    
    # API Keys
    HUGGING_FACE_TOKEN = Config.get('HUGGING_FACE_TOKEN', None)


def setup_logging(level=AppConfig.LOG_LEVEL):
    """Setup application logging"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
