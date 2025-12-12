"""
RAG 파이프라인 모듈
벡터 저장소 관리, 임베딩 생성, 검색 및 LLM 통합 기능 제공
"""

import os
from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


class RAGPipeline:
    """RAG 파이프라인을 관리하는 클래스"""
    
    def __init__(
        self,
        persist_directory: str = "./chroma_pdf_cache",
        collection_name: str = "pdf_documents",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "llama3"
    ):
        """
        Args:
            persist_directory: ChromaDB 데이터 저장 디렉토리
            collection_name: ChromaDB 컬렉션 이름
            ollama_base_url: Ollama 서버 URL
            ollama_model: 사용할 Ollama 모델 이름
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        # 임베딩 모델 초기화
        self.embeddings = OllamaEmbeddings(
            base_url=ollama_base_url,
            model=ollama_model
        )
        
        # LLM 초기화
        self.llm = OllamaLLM(
            base_url=ollama_base_url,
            model=ollama_model,
            temperature=0.7
        )
        
        # 벡터 스토어 초기화
        self.vectorstore: Optional[Chroma] = None
        self._load_or_create_vectorstore()
    
    def _load_or_create_vectorstore(self):
        """기존 벡터 스토어를 로드하거나 새로 생성합니다."""
        try:
            # 기존 벡터 스토어 로드 시도
            if os.path.exists(self.persist_directory):
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                print(f"[OK] 기존 벡터 스토어 로드 완료: {self.collection_name}")
            else:
                # 새로 생성
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings
                )
                print(f"[OK] 새 벡터 스토어 생성 완료: {self.collection_name}")
        except Exception as e:
            print(f"[ERROR] 벡터 스토어 로드 실패: {str(e)}")
            # 새로 생성
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
    
    def add_documents(self, documents: List[Document]):
        """
        문서를 벡터 스토어에 추가합니다.
        
        Args:
            documents: 추가할 Document 객체 리스트
        """
        if not self.vectorstore:
            self._load_or_create_vectorstore()
        
        if documents:
            self.vectorstore.add_documents(documents)
            # 최신 Chroma는 자동으로 persist되므로 persist() 호출 불필요
            # persist_directory를 지정했으면 자동 저장됨
            print(f"[OK] {len(documents)}개 문서가 벡터 스토어에 추가되었습니다.")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        쿼리와 유사한 문서를 검색합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            검색된 Document 객체 리스트
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        return self.vectorstore.similarity_search(query, k=k)
    
    def search_with_score(self, query: str, k: int = 5) -> List[tuple]:
        """
        쿼리와 유사한 문서를 유사도 점수와 함께 검색합니다.
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 수
            
        Returns:
            (Document, score) 튜플 리스트
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        return self.vectorstore.similarity_search_with_score(query, k=k)
    
    def _format_docs(self, docs: List[Document]) -> str:
        """문서 리스트를 컨텍스트 문자열로 변환"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변을 생성합니다.
        
        Args:
            question: 사용자 질문
            
        Returns:
            답변과 소스 문서를 포함한 딕셔너리
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 초기화되지 않았습니다.")
        
        # 검색기 생성
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        
        # 관련 문서 검색
        source_documents = retriever.invoke(question)
        
        # 컨텍스트 생성
        context = self._format_docs(source_documents)
        
        # 프롬프트 템플릿
        prompt_template = """다음 컨텍스트를 사용하여 질문에 답변하세요. 
컨텍스트에서 답을 찾을 수 없으면 "모르겠습니다"라고 답변하세요.

컨텍스트: {context}

질문: {question}

답변:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 체인 구성 - 더 간단하고 안정적인 방식
        def format_retriever_input(inputs: Dict) -> str:
            return inputs["question"]
        
        # 컨텍스트와 질문을 함께 전달
        formatted_prompt = prompt.format(context=context, question=question)
        
        # LLM 호출
        answer = self.llm.invoke(formatted_prompt)
        
        return {
            "answer": answer,
            "source_documents": source_documents
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        벡터 스토어 컬렉션 정보를 반환합니다.
        
        Returns:
            컬렉션 정보 딕셔너리
        """
        if not self.vectorstore:
            return {"error": "벡터 스토어가 초기화되지 않았습니다."}
        
        collection = self.vectorstore._collection
        count = collection.count()
        
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "model": self.ollama_model
        }

