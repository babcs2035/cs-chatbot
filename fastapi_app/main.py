from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import sys

# --- パス設定 ---
# スクリプトの実行場所に関わらず、このファイルがあるディレクトリをPythonパスに追加
# これにより、'scraper'や'rag_handler'のようなローカルモジュールを安定してインポートできる
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# --- モジュールインポート ---
# rag_handlerは起動時に初期化され、ベクトルDBなどをメモリに読み込む
from rag_handler import rag_handler_instance
# scraperは/scrapeエンドポイントでのみ使用する
from scraper import crawl_website

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(
    title="RAG Chatbot API",
    description="ローカルLLMとRAG（Retrieval-Augmented Generation）を使用したチャットボットAPIです。`/docs`から対話的なAPIドキュメントを利用できます。",
    version="1.0.0"
)

# --- データモデル定義 (Pydantic) ---
# APIが受け取るリクエストボディの型を定義

class ChatQuery(BaseModel):
    """ /chat エンドポイントのリクエストボディ """
    question: str

class ScrapeRequest(BaseModel):
    """ /scrape エンドポイントのリクエストボディ """
    urls: list[str]

# --- APIエンドポイント定義 ---

@app.get("/", summary="ヘルスチェック")
async def root():
    """
    APIサーバーが正常に起動しているかを確認するためのシンプルなエンドポイント。
    """
    return {"status": "ok", "message": "API is running. Please head to /docs to test the endpoints."}

@app.post("/chat", summary="チャットボットに質問を送信")
async def chat_endpoint(query: ChatQuery):
    """
    ユーザーからの質問を受け取り、知識ベースを基に生成された回答を返します。
    """
    try:
        # rag_handlerのメソッド名を `query` から `ask` に修正
        response = rag_handler_instance.ask(query.question)
        
        # rag_handler内でエラーが発生した場合の処理
        if "error" in response:
            raise HTTPException(status_code=500, detail=response["error"])
        
        # 回答とソースドキュメントを含む完全なレスポンスを返すように修正
        return response
    except Exception as e:
        # 予期せぬエラーが発生した場合
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.post("/scrape", summary="Webサイトをスクレイピングして知識ベースを構築")
async def scrape_endpoint(request: ScrapeRequest):
    """
    指定されたURLリストをスクレイピングし、チャットボットの知識ベースを構築または更新します。
    この処理は時間がかかることがあります。完了後、RAGシステムは新しいデータで再初期化されます。
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty.")
    
    try:
        print("🚀 Starting web scraping...")
        crawl_website(request.urls) # scraper.pyの関数を実行
        
        print("🔄 Re-initializing RAG handler with new data...")
        # 新しく生成されたデータでRAGハンドラを再初期化
        rag_handler_instance._initialize()
        
        return {"message": "Scraping and knowledge base update completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape or re-initialize: {str(e)}")
