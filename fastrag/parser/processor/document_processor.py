from typing import Union
import pathlib
from abc import abstractmethod
from pydantic import BaseModel, field_validator

class DocumentProcessorInterface(BaseModel):   
    """
    Provides abstract interface and shared functionality for document of different types
    """
    file_path: pathlib.Path 
    
    # Runs file_path validation before any processing
    @field_validator("file_path")
    def validate_file_path(cls, value: Union[str, pathlib.Path]) -> pathlib.Path:
        if isinstance(value, str):
            return pathlib.Path(value)
        return value

    @property
    def filename(self) -> str:
        return self.file_path.name
    
    @abstractmethod
    def cleanup(self) -> None:
        """Closes documents manually"""
        raise NotImplementedError
    
    @abstractmethod
    def extract_text(self, format: str) -> str:
        raise NotImplementedError    
    
