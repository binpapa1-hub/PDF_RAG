"""
Gradio 웹 UI 애플리케이션
PDF RAG 시스템에 대한 사용자 친화적인 웹 인터페이스 제공
"""

import gradio as gr
import os
from pathlib import Path
from pdf_processor import PDFProcessor
from rag_pipeline import RAGPipeline
from dotenv import load_dotenv

load_dotenv()

# 전역 변수 초기화
pdf_processor = PDFProcessor()
rag_pipeline = RAGPipeline(
    persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_pdf_cache"),
    collection_name=os.getenv("CHROMA_COLLECTION_NAME", "pdf_documents"),
    ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3")
)


def process_pdf_file(file):
    """PDF 파일을 처리하고 벡터 스토어에 추가"""
    if file is None:
        return "파일을 선택해주세요."
    
    try:
        documents = pdf_processor.process_pdf(file.name)
        rag_pipeline.add_documents(documents)
        return f"✓ 성공! {len(documents)}개의 청크가 벡터 스토어에 추가되었습니다."
    except Exception as e:
        return f"✗ 오류 발생: {str(e)}"


def query_rag(question, history):
    """RAG 시스템에 질문하고 답변 받기"""
    if not question.strip():
        return history, "질문을 입력해주세요."
    
    try:
        result = rag_pipeline.query(question)
        answer = result["answer"]
        
        # 소스 문서 정보
        sources_info = []
        for i, doc in enumerate(result["source_documents"][:3], 1):
            source = doc.metadata.get("source", "Unknown")
            sources_info.append(f"[{i}] {source}")
        
        sources_text = "\n".join(sources_info) if sources_info else "소스 없음"
        
        # 히스토리 업데이트
        history.append((question, f"{answer}\n\n📚 참고 문서:\n{sources_text}"))
        
        return history, ""
    except Exception as e:
        error_msg = f"오류 발생: {str(e)}"
        history.append((question, error_msg))
        return history, ""


def search_documents(query, k):
    """문서 검색"""
    if not query.strip():
        return "검색어를 입력해주세요."
    
    try:
        results = rag_pipeline.search_with_score(query, k=int(k))
        
        output = f"검색 결과 ({len(results)}개):\n\n"
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "Unknown")
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            output += f"[{i}] {source} (유사도: {score:.4f})\n"
            output += f"   {content_preview}\n\n"
        
        return output
    except Exception as e:
        return f"오류 발생: {str(e)}"


def get_collection_info():
    """벡터 스토어 정보 조회"""
    try:
        info = rag_pipeline.get_collection_info()
        return f"""벡터 스토어 정보:
- 컬렉션 이름: {info.get('collection_name', 'N/A')}
- 문서 수: {info.get('document_count', 0)}
- 저장 경로: {info.get('persist_directory', 'N/A')}
- 모델: {info.get('model', 'N/A')}
"""
    except Exception as e:
        return f"오류 발생: {str(e)}"


# Gradio 인터페이스 구성
with gr.Blocks(title="PDF RAG 시스템", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📚 PDF RAG 시스템
        PDF 문서를 업로드하고 질문에 답변받을 수 있는 RAG(Retrieval-Augmented Generation) 시스템입니다.
        """
    )
    
    with gr.Tabs():
        # 탭 1: PDF 업로드
        with gr.Tab("📄 PDF 업로드"):
            gr.Markdown("### PDF 파일을 업로드하여 벡터 스토어에 추가합니다.")
            file_input = gr.File(
                label="PDF 파일 선택",
                file_types=[".pdf"]
            )
            upload_btn = gr.Button("업로드 및 처리", variant="primary")
            upload_output = gr.Textbox(
                label="처리 결과",
                lines=3,
                interactive=False
            )
            
            upload_btn.click(
                fn=process_pdf_file,
                inputs=file_input,
                outputs=upload_output
            )
        
        # 탭 2: 질문하기
        with gr.Tab("💬 질문하기"):
            gr.Markdown("### PDF 문서에 대해 질문하고 답변을 받습니다.")
            
            chatbot = gr.Chatbot(
                label="대화",
                height=400,
                show_copy_button=True
            )
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="질문",
                    placeholder="질문을 입력하세요...",
                    scale=4
                )
                submit_btn = gr.Button("전송", variant="primary", scale=1)
            
            submit_btn.click(
                fn=query_rag,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
            
            question_input.submit(
                fn=query_rag,
                inputs=[question_input, chatbot],
                outputs=[chatbot, question_input]
            )
        
        # 탭 3: 문서 검색
        with gr.Tab("🔍 문서 검색"):
            gr.Markdown("### 유사도 기반으로 관련 문서를 검색합니다.")
            
            with gr.Row():
                search_query = gr.Textbox(
                    label="검색어",
                    placeholder="검색어를 입력하세요...",
                    scale=3
                )
                search_k = gr.Slider(
                    label="결과 수",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    scale=1
                )
                search_btn = gr.Button("검색", variant="primary", scale=1)
            
            search_output = gr.Textbox(
                label="검색 결과",
                lines=15,
                interactive=False
            )
            
            search_btn.click(
                fn=search_documents,
                inputs=[search_query, search_k],
                outputs=search_output
            )
            
            search_query.submit(
                fn=search_documents,
                inputs=[search_query, search_k],
                outputs=search_output
            )
        
        # 탭 4: 시스템 정보
        with gr.Tab("ℹ️ 시스템 정보"):
            gr.Markdown("### 벡터 스토어 및 시스템 정보를 확인합니다.")
            
            info_btn = gr.Button("정보 조회", variant="primary")
            info_output = gr.Textbox(
                label="시스템 정보",
                lines=10,
                interactive=False
            )
            
            info_btn.click(
                fn=get_collection_info,
                inputs=None,
                outputs=info_output
            )


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("HOST", "0.0.0.0"),
        server_port=int(os.getenv("GRADIO_PORT", 7860)),
        share=False
    )

