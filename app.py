"""
FastAPI 애플리케이션
REST API를 통해 PDF RAG 시스템에 접근할 수 있는 엔드포인트 제공
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import os
import tempfile
from pathlib import Path

from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="PDF RAG API",
    description="PDF 문서 기반 RAG(Retrieval-Augmented Generation) 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
pdf_processor = PDFProcessor()
rag_pipeline = RAGPipeline(
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_pdf_cache"),
    collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "PDF RAG API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "query": "/query",
            "search": "/search",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        info = rag_pipeline.get_collection_info()
        return {
            "status": "healthy",
            "vectorstore": info
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    PDF 파일을 업로드하고 벡터 스토어에 추가합니다.
    
    Args:
        file: 업로드할 PDF 파일
        
    Returns:
        처리 결과
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name
    
    try:
        # PDF 처리
        documents = pdf_processor.process_pdf(tmp_path)
        
        # 벡터 스토어에 추가
        rag_pipeline.add_documents(documents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "chunks": len(documents),
            "message": f"{len(documents)}개의 청크가 벡터 스토어에 추가되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 중 오류 발생: {str(e)}")
    finally:
        # 임시 파일 삭제
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query")
async def query(question: str):
    """
    질문에 대한 답변을 생성합니다.
    
    Args:
        question: 사용자 질문
        
    Returns:
        답변과 소스 문서 정보
    """
    try:
        result = rag_pipeline.query(question)
        
        # 소스 문서 정보 추출
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return {
            "answer": result["answer"],
            "sources": sources,
            "source_count": len(sources)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"질의 처리 중 오류 발생: {str(e)}")


@app.get("/search")
async def search(query: str, k: int = 5):
    """
    쿼리와 유사한 문서를 검색합니다.
    
    Args:
        query: 검색 쿼리
        k: 반환할 문서 수 (기본값: 5)
        
    Returns:
        검색된 문서 리스트
    """
    try:
        results = rag_pipeline.search_with_score(query, k=k)
        
        search_results = []
        for doc, score in results:
            search_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
        
        return {
            "query": query,
            "results": search_results,
            "count": len(search_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")


@app.get("/info")
async def get_info():
    """벡터 스토어 정보를 반환합니다."""
    try:
        info = rag_pipeline.get_collection_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"정보 조회 중 오류 발생: {str(e)}")


@app.post("/process-directory")
async def process_directory(directory_path: str):
    """
    디렉토리 내의 모든 PDF 파일을 처리합니다.
    
    Args:
        directory_path: PDF 파일들이 있는 디렉토리 경로
        
    Returns:
        처리 결과
    """
    if not os.path.exists(directory_path):
        raise HTTPException(status_code=404, detail="디렉토리를 찾을 수 없습니다.")
    
    try:
        documents = pdf_processor.process_multiple_pdfs(directory_path)
        rag_pipeline.add_documents(documents)
        
        return {
            "status": "success",
            "directory": directory_path,
            "total_chunks": len(documents),
            "message": f"{len(documents)}개의 청크가 벡터 스토어에 추가되었습니다."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"디렉토리 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", 8000)),
        reload=os.getenv("DEBUG", "True").lower() == "true"
    )

