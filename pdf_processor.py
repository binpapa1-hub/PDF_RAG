"""
PDF 처리 모듈
PDF 파일에서 텍스트를 추출하고 청킹하는 기능 제공
"""

import os
from typing import List, Dict, Optional
from pathlib import Path
import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class PDFProcessor:
    """PDF 파일을 처리하고 텍스트를 추출하는 클래스"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Args:
            chunk_size: 청크의 최대 크기 (문자 수)
            chunk_overlap: 청크 간 겹치는 문자 수
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, any]:
        """
        PDF 파일에서 텍스트를 추출합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            텍스트와 메타데이터를 포함한 딕셔너리
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
        
        doc = fitz.open(pdf_path)
        text_content = []
        metadata = {
            "source": os.path.basename(pdf_path),
            "file_path": pdf_path,
            "total_pages": len(doc),
            "file_size": os.path.getsize(pdf_path)
        }
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            text_content.append(text)
        
        doc.close()
        
        full_text = "\n".join(text_content)
        
        return {
            "text": full_text,
            "metadata": metadata
        }
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Document]:
        """
        텍스트를 청크로 분할합니다.
        
        Args:
            text: 분할할 텍스트
            metadata: 각 청크에 추가할 메타데이터
            
        Returns:
            Document 객체 리스트
        """
        chunks = self.text_splitter.create_documents(
            texts=[text],
            metadatas=[metadata]
        )
        
        # 각 청크에 페이지 정보 추가
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        
        return chunks
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        PDF 파일을 처리하여 Document 리스트로 반환합니다.
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Document 객체 리스트
        """
        # 텍스트 추출
        extracted_data = self.extract_text_from_pdf(pdf_path)
        
        # 청킹
        documents = self.chunk_text(
            extracted_data["text"],
            extracted_data["metadata"]
        )
        
        return documents
    
    def process_multiple_pdfs(self, pdf_directory: str) -> List[Document]:
        """
        디렉토리 내의 모든 PDF 파일을 처리합니다.
        
        Args:
            pdf_directory: PDF 파일들이 있는 디렉토리 경로
            
        Returns:
            모든 PDF에서 추출한 Document 객체 리스트
        """
        pdf_files = list(Path(pdf_directory).glob("*.pdf"))
        all_documents = []
        
        for pdf_file in pdf_files:
            try:
                documents = self.process_pdf(str(pdf_file))
                all_documents.extend(documents)
                print(f"[OK] 처리 완료: {pdf_file.name} ({len(documents)}개 청크)")
            except Exception as e:
                print(f"[ERROR] 처리 실패: {pdf_file.name} - {str(e)}")
        
        return all_documents

