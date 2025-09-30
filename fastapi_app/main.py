import os
import sys
import uuid
import asyncio
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ローカルモジュールをインポートするためのパス設定
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_handler import rag_handler_instance
from scraper import crawl_website

# --- FastAPIアプリケーションの初期化 ---
app = FastAPI(
    title="Advanced RAG Chatbot API",
    description="長時間処理に対応した非同期タスク方式のRAGチャットボットAPIです。",
    version="2.0.0",
)

# フロントエンドからのリクエストを許可するためのCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ktak.dev", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 非同期タスクの管理 ---
# バックグラウンドで実行されるタスクの状態と結果を保存するインメモリ辞書
# key: task_id, value: {"status": "running" | "completed" | "failed", "result": ...}
tasks = {}


def run_rag_task(task_id: str, query: str):
    """バックグラウンドでRAG処理を実行し、結果をtasks辞書に保存するワーカー関数。"""
    try:
        # rag_handler.askは非同期関数のため、新しいイベントループで実行
        result = asyncio.run(rag_handler_instance.ask(query))
        tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:
        tasks[task_id] = {"status": "failed", "result": {"error": str(e)}}


# --- データモデル定義 (Pydantic) ---
class ChatQuery(BaseModel):
    """質問リクエストのボディ"""

    question: str


class ScrapeRequest(BaseModel):
    """スクレイピングリクエストのボディ"""

    urls: list[str]


class ChatTaskResponse(BaseModel):
    """チャットタスク開始時のレスポンス"""

    task_id: str


# --- APIエンドポイント定義 ---
@app.get("/", summary="ヘルスチェック")
async def root():
    """APIサーバーの稼働状況を確認する。"""
    return {"status": "ok", "message": "API is running."}


@app.post(
    "/chat/start", summary="チャット処理タスクを開始", response_model=ChatTaskResponse
)
async def start_chat_task(query: ChatQuery, background_tasks: BackgroundTasks):
    """
    時間のかかるRAG処理をバックグラウンドタスクとして開始し、即座にタスクIDを返す。
    """
    if not query.question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {"status": "running", "result": None}
    background_tasks.add_task(run_rag_task, task_id, query.question)

    return {"task_id": task_id}


@app.get("/chat/status/{task_id}", summary="チャット処理タスクの状況と結果を取得")
async def get_chat_status(task_id: str):
    """
    指定されたタスクIDの現在の状態（実行中、完了、失敗）と、完了している場合はその結果を返す。
    """
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    # 完了または失敗していれば、メモリからタスク情報を削除する
    if task["status"] in ["completed", "failed"]:
        return tasks.pop(task_id)

    return task


@app.post("/scrape", summary="知識ベースを構築")
async def scrape_endpoint(request: ScrapeRequest):
    """
    指定されたURLをスクレイピングし、知識ベースを更新する。（この処理は同期的に実行されます）
    """
    if not request.urls:
        raise HTTPException(status_code=400, detail="URL list cannot be empty.")
    try:
        crawl_website(request.urls)
        rag_handler_instance._initialize()
        return {"message": "Scraping and knowledge base update completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to scrape: {str(e)}")
