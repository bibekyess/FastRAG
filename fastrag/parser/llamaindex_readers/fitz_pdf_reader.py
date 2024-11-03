from llama_index.core.readers.base import BaseReader
from typing import Any, List, Optional, Dict
from pathlib import Path
from llama_index.core.schema import Document
from typing import Union
from fastrag.parser.processor.pdf_document_processor import PDFDocumentProcessor


class FitzPDFReader(BaseReader):
    def __init__(
        self, chunk_size: int=512, chunk_stride:int = 256, pagenum: int=-1, 
        split_documents: bool=True, page_wise: bool=False, overlap: bool=True, 
        *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        if chunk_size <= 0: chunk_size=256
        if chunk_stride < 0: chunk_stride=chunk_size//2
        if chunk_stride > chunk_size: chunk_stride=chunk_size//2
        self.chunk_size = chunk_size
        self.chunk_stride = chunk_stride
        self.pagenum = pagenum
        self.split_documents = split_documents
        self.page_wise = page_wise
        self.overlap = overlap
        if not self.overlap:
            self.chunk_stride=0


    def load_data(
        self, file: Union[str, Path], extra_info: Optional[Dict] = {}
    ) -> List[Document]:
        """
        Load data and extract document chunks from corresponding file contents.
        Please don't modify the arguments passed here as LlamaIndex's SimpleDirectoryReader expects these arguments only.

        Args:
            file (str): A url or file path pointing to the document
            extra_info (Optional[Dict]): Additional information that might be needed for loading the data, by default None.
                LlamaIndex SimpleDirectoryReader by default provides some basic File metadatas with this parameter
        Returns:
            List[Document]: List of documents.
        """

        def get_sentence_chunks(text, chunk_size, chunk_stride):
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size = chunk_size,
                        chunk_overlap  = chunk_stride,
                        length_function = lambda x: len(x.split()),
                        is_separator_regex = False,
                    )

            texts = text_splitter.create_documents([text])
            return [t.page_content for t in texts]
        
        pdf_processor = PDFDocumentProcessor(file_path=str(file))
        page_wise_texts = pdf_processor.extract_text(pagewise=True)

        results = []
        prev_chunk_preceding_content = None

        entire_text = ""
        document = None
        
        for pagenum, document_chunks in enumerate(page_wise_texts):

            if not self.split_documents:
                entire_text += document_chunks.strip() + "\n\n----\n\n" # FIXME '----' represents page splits
                continue

            if not self.page_wise:
                texts = get_sentence_chunks(text=document_chunks, chunk_size=self.chunk_size, chunk_stride=self.chunk_stride)
            else:
                texts = [document_chunks]  
        
            for idx, text in enumerate(texts):
                if self.pagenum != -1:
                    extra_info["page_label"] = self.pagenum # This is being inquired about the specific page
                else:
                    extra_info["page_label"] = pagenum
                document = Document(
                    text=text, extra_info=extra_info
                )
                results.append(document)

            if prev_chunk_preceding_content is not None and len(texts) > 0:
                current_whole_text = document.text
                current_head_text = get_sentence_chunks(text=current_whole_text, chunk_size=self.chunk_size//2, chunk_stride=0)[0]
                previous_whole_text = prev_chunk_preceding_content.text
                previous_tail_text = get_sentence_chunks(text=previous_whole_text, chunk_size=self.chunk_size//2, chunk_stride=0)[-1]
                overlapping_text = previous_tail_text + current_head_text
                metadata = prev_chunk_preceding_content.metadata
                metadata["overlapping"] = True
                overlapping_document = Document(
                    text=overlapping_text, extra_info=metadata
                )
                results.append(overlapping_document)
            
            if self.overlap and document is not None:
                prev_chunk_preceding_content = document
        
        if not self.split_documents:
            results = [Document(
                text=entire_text, extra_info=extra_info
            )]

        return results
    