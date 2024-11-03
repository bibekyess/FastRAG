import fitz
from PIL import Image
from fastrag.parser.processor.document_processor import DocumentProcessorInterface
from enum import Enum
from typing import Union, Literal, List

class TextExtractor(Enum):
    FITZ = "fitz"
    OCR = "ocr"


class PDFDocumentProcessor(DocumentProcessorInterface):
    """
    Provides abstract interface and shared functionality for PDF Documents Processors
    """
    
    fitz_doc: fitz.Document = None
    
    class Config:
        arbitrary_types_allowed = True  # Skip validation for unsupported types like fitz.Document, fitz.Page

    def open_fitz_doc(self) -> None:
        if self.fitz_doc is None:
            self.fitz_doc =fitz.open(self.file_path)

    def cleanup(self) -> None:
        if self.fitz_doc:
            self.fitz_doc.close()

    def get_fitz_page(self, page_number: int) -> fitz.Page:
        if not self.fitz_doc:
            self.open_fitz_doc()
        return self.fitz_doc.load_page(page_number)
    
    def get_page_image(self, page_number: int) -> Image.Image:
        if not self.fitz_doc:
            self.open_fitz_doc()
        page = self.fitz_doc.load_page(page_number)
        pixmap = page.get_pixmap()
        image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)       
        return image
    
    def get_text_extractor(self) -> TextExtractor:
        # TODO OCR Implementation
        return TextExtractor.FITZ
        # return TextExtractor.OCR
        
    
    def extract_text(self, format: Literal['raw', 'md']='raw', pagewise: bool=False) -> Union[List[str], str]:
        text_extractor = self.get_text_extractor()
        assert text_extractor == TextExtractor.FITZ, "Only Fitz Text extractor is supported as of now!! Thank you for your patience."
        
        if format=='raw':
            self.open_fitz_doc()
            if pagewise:
                document_text = []
                for page in self.fitz_doc.pages():
                    page_text = page.get_text()
                    document_text.append(page_text)
                return document_text                
            else:
                document_text=""
                for page in self.fitz_doc.pages():
                    page_text = page.get_text()
                    document_text += page_text +"\n\n-<PAGE_BREAK>-\n\n"
                return document_text
        elif format=="md":
            import pymupdf4llm
            md_text = pymupdf4llm.to_markdown(self.file_path)
            return md_text
        else:
            raise ValueError("Only 'raw' and 'md formats are supported as of now!!")
        
