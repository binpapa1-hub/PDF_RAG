"""
메인 실행 파일
PDF RAG 시스템을 초기화하고 실행하는 진입점
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline

load_dotenv()


def process_pdfs_from_directory(directory_path: str):
    """디렉토리 내의 모든 PDF를 처리"""
    print(f"\n[INFO] 디렉토리 처리 시작: {directory_path}")
    
    pdf_processor = PDFProcessor()
    rag_pipeline = RAGPipeline(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_pdf_cache"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    
    documents = pdf_processor.process_multiple_pdfs(directory_path)
    
    if documents:
        rag_pipeline.add_documents(documents)
        print(f"\n[OK] 총 {len(documents)}개의 청크가 벡터 스토어에 추가되었습니다.")
    else:
        print("\n[ERROR] 처리할 PDF 파일이 없습니다.")


def interactive_query():
    """대화형 질의 모드"""
    print("\n[INFO] 대화형 질의 모드 (종료하려면 'quit' 또는 'exit' 입력)")
    print("=" * 60)
    
    rag_pipeline = RAGPipeline(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_pdf_cache"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    
    # 벡터 스토어 정보 확인
    info = rag_pipeline.get_collection_info()
    if info.get("document_count", 0) == 0:
        print("[WARNING] 벡터 스토어에 문서가 없습니다. 먼저 PDF를 처리해주세요.")
        return
    
    print(f"[INFO] 벡터 스토어에 {info.get('document_count', 0)}개의 문서가 있습니다.\n")
    
    while True:
        try:
            question = input("\n질문: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n[INFO] 종료합니다.")
                break
            
            if not question:
                continue
            
            print("\n[INFO] 답변 생성 중...")
            result = rag_pipeline.query(question)
            
            print(f"\n[ANSWER] 답변:\n{result['answer']}\n")
            
            if result.get("source_documents"):
                print("[INFO] 참고 문서:")
                for i, doc in enumerate(result["source_documents"][:3], 1):
                    source = doc.metadata.get("source", "Unknown")
                    print(f"  [{i}] {source}")
        
        except KeyboardInterrupt:
            print("\n\n[INFO] 종료합니다.")
            break
        except Exception as e:
            print(f"\n[ERROR] 오류 발생: {str(e)}")


def show_info():
    """시스템 정보 표시"""
    rag_pipeline = RAGPipeline(
        persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_pdf_cache"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
    )
    
    info = rag_pipeline.get_collection_info()
    
    print("\n" + "=" * 60)
    print("[INFO] 시스템 정보")
    print("=" * 60)
    print(f"컬렉션 이름: {info.get('collection_name', 'N/A')}")
    print(f"문서 수: {info.get('document_count', 0)}")
    print(f"저장 경로: {info.get('persist_directory', 'N/A')}")
    print(f"모델: {info.get('model', 'N/A')}")
    print("=" * 60 + "\n")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="PDF RAG 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 디렉토리 내 PDF 처리
  python main.py --process-dir ./pdfs
  
  # 대화형 질의 모드
  python main.py --query
  
  # 시스템 정보 확인
  python main.py --info
        """
    )
    
    parser.add_argument(
        "--process-dir",
        type=str,
        help="PDF 파일들이 있는 디렉토리 경로"
    )
    parser.add_argument(
        "--query",
        action="store_true",
        help="대화형 질의 모드 실행"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="시스템 정보 표시"
    )
    
    args = parser.parse_args()
    
    if args.process_dir:
        if not os.path.exists(args.process_dir):
            print(f"[ERROR] 오류: 디렉토리를 찾을 수 없습니다: {args.process_dir}")
            sys.exit(1)
        process_pdfs_from_directory(args.process_dir)
    elif args.query:
        interactive_query()
    elif args.info:
        show_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

