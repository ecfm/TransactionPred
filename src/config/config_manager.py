import inspect
import logging
from typing import Any, Dict, Type, get_args, get_origin

from pydantic import BaseModel
from pydantic.error_wrappers import ValidationError

from src.config.grid_search_config import LoggerConfig


class ConfigManager:
    @staticmethod
    def load_and_validate_config(config_dict: Dict[str, Any], config_class: Type[BaseModel]) -> BaseModel:
        # Apply default values recursively
        config_dict = ConfigManager._apply_default_values(config_dict, config_class)
        
        try:
            # Create the config instance
            config = config_class(**config_dict)
            return config
        except ValidationError as e:
            print(f"Config validation failed: {e}")
            raise

    @staticmethod
    def _apply_default_values(config_dict: Dict[str, Any], model_class: Type[BaseModel]) -> Dict[str, Any]:
        for field_name, field in model_class.__fields__.items():
            if field_name not in config_dict:
                if field.default is not None:
                    config_dict[field_name] = field.default
                elif field.default_factory is not None:
                    config_dict[field_name] = field.default_factory()
            elif config_dict[field_name] is not None:  # Only process non-None values
                field_type = field.outer_type_  # Use outer_type_ to get the full type, including generics
                
                # Handle Dict[str, BaseModel] case
                if get_origin(field_type) is dict:
                    key_type, value_type = get_args(field_type)
                    if key_type is str and inspect.isclass(value_type) and issubclass(value_type, BaseModel):
                        config_dict[field_name] = {
                            k: ConfigManager._apply_default_values(v, value_type)
                            for k, v in config_dict[field_name].items()
                        }
                # Check if the field is a Pydantic model
                elif inspect.isclass(field_type) and issubclass(field_type, BaseModel):
                    config_dict[field_name] = ConfigManager._apply_default_values(config_dict[field_name], field_type)
                
                # Handle List[BaseModel] case
                elif get_origin(field_type) is list:
                    args = get_args(field_type)
                    if args and inspect.isclass(args[0]) and issubclass(args[0], BaseModel):
                        if isinstance(config_dict[field_name], list):
                            config_dict[field_name] = [
                                ConfigManager._apply_default_values(item, args[0])
                                for item in config_dict[field_name]
                            ]

        return config_dict

    @staticmethod
    def setup_logging(logger_config: LoggerConfig):
        logger = logging.getLogger()
        logger.setLevel(logger_config.level)

        # Remove all handlers associated with the root logger object
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Create formatter
        formatter = logging.Formatter(fmt=logger_config.format, datefmt=logger_config.date_format)

        # Always add a stream handler for stdout
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        # If a file path is provided, add a file handler
        if logger_config.file_path:
            file_handler = logging.FileHandler(logger_config.file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logging.info("Logging setup complete")