"""
高度なRAG (Retrieval-Augmented Generation) パイプラインを処理するハンドラ．
以下の先進的な技術を組み合わせて，回答の精度と効率を最大化する．
1. クエリ拡張 (Query Expansion): ユーザーの質問を多様化し，検索の網羅性を向上．
2. 文脈ウィンドウ検索 (Sentence-Window Retrieval): 検索精度と文脈の豊かさを両立．
3. RAG-Fusion (RRF): 複数の検索結果を統合し，最も信頼性の高い情報を抽出．
4. 抽出→統合パイプライン: LLMがノイズを除去してから回答を生成することで，精度を向上．
5. キャッシュ機構: 埋め込みとLLMの応答をキャッシュし，処理を高速化．
"""

import os
import jq
import logging
import hashlib
import asyncio
import json
from typing import Any, Dict, List, Generator
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# LangChainの主要コンポーネント
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.globals import set_llm_cache

# LangChainコミュニティのインテグレーション
from langchain_community.cache import InMemoryCache
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# ChromaDBの設定
from chromadb.config import Settings

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


# --- システム全体で使用する定数 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_DB_PATH = "./db"
DATA_PATH = "./data/scraped_data.json"
CACHE_PATH = "./cache"
PARENT_DOCS_PATH = "./data/parent_docs.json"
# 文脈ウィンドウ検索で使用するチャンクサイズ
PARENT_CHUNK_SIZE = 1024  # LLMに渡す，文脈が豊富な親チャンク
CHILD_CHUNK_SIZE = 256  # ベクトル検索で使う，精度の高い子チャンク
CHUNK_OVERLAP = 64
# DB構築時のバッチサイズ
BATCH_SIZE = 128

# アプリケーション起動時にLLMのインメモリキャッシュを有効化
logger.info("🧠 Activating in-memory LLM cache...")
set_llm_cache(InMemoryCache())


# --- プロンプトテンプレート定義 ---

# 方針1「クエリ拡張」で使用するプロンプト
query_expansion_template = """
### 指示
あなたは，ユーザーの質問を分析し，ベクトル検索の精度を向上させるための多様な検索クエリを生成するAIアシスタントです．
ユーザーの元の質問を基に，異なる視点やキーワードを用いた検索クエリを3つ生成してください．

### 厳格なルール
- **出力は"queries"というキーを持つJSONオブジェクトのみ**とし，前後に説明や挨拶などの余計なテキストを一切含めないでください．
- 必ず3つのクエリを生成し，すべて日本語で書いてください．
- 出力形式は必ず `{{"queries": ["クエリ1", "クエリ2", "クエリ3"]}}` のようにしてください．

### 元の質問
{question}
"""

# RAGパイプラインの「抽出」ステップで使用するプロンプト
# 検索で得られた広範なコンテキストから，質問に直接関連する情報のみをLLMに抜き出させる
extraction_template = """
### 指示
あなたは，提供された「コンテキスト情報」の中から，ユーザーの「質問」に回答するために必要不可欠な情報のみを正確に抜き出すAIアシスタントです．
以下のルールに従って，関連する文章を抽出してください．

### ルール
- 「コンテキスト情報」の各文章を評価し，「質問」への回答に直接関連する文章だけを抽出してください．
- 抽出した文章は，元の文章から一切変更を加えず，そのまま出力してください．
- 複数の関連文章がある場合は，それぞれを改行で区切って出力してください．
- 関連する情報が全くない場合は，何も出力しないでください．

### コンテキスト情報
{context}

### 質問
{question}

### 抽出結果
"""

# 最終的な回答を生成（統合ステップ）するためのメインプロンプト
answer_generation_template = """
### 指示
あなたは東京大学の情報システムに関する学生や教職員からの質問に回答する，非常に優秀で親切なサポート担当者です．
提供された「コンテキスト情報」を注意深く読み込み，以下のルールに従って回答を生成してください．

### ルール
1.  **言語**: 回答はすべて日本語で生成してください．
2.  **役割の遵守**: 必ずサポート担当者として，丁寧で分かりやすい言葉遣いをしてください．
3.  **情報の厳守**: 回答は，**提供された「コンテキスト情報」に書かれている内容のみ**を根拠としてください．コンテキストに記載のない情報や，あなた自身の知識を決して補ってはいけません．
4.  **段階的な説明**: ユーザーの問題を解決するために，具体的な手順を段階的に説明してください．
5.  **統合と要約**: 複数のコンテキスト情報を参照し，それらを自然な文章に統合・要約して回答を作成してください．情報の断片をコピーペーストしてはいけません．
6.  **情報がない場合**: 質問に対する答えがコンテキスト情報に見つからない場合は，曖昧な回答はせず，「申し訳ありませんが，ご質問に関する情報は見つかりませんでした．」と明確に回答してください．

### コンテキスト情報
{context}

### 質問
{question}

### 回答
"""


# --- ヘルパー関数群 ---


def metadata_func(record: dict, metadata: dict) -> dict:
    """JSONLoaderで文書を読み込む際に，URLをメタデータとして付与する．"""
    metadata["source"] = record.get("url")
    return metadata


def batch_generator(
    data: List[Any], batch_size: int
) -> Generator[List[Any], None, None]:
    """リストを指定されたバッチサイズに分割するジェネレータ．"""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def create_sha256_encoder(namespace: str):
    """埋め込みキャッシュ用のキーを生成するエンコーダーを作成する．"""

    def sha256_encoder(payload: str) -> str:
        # モデル名（namespace）をキーに含めることで，異なるモデルのキャッシュが衝突しないようにする
        namespaced_payload = f"{namespace}:{payload}".encode("utf-8")
        return hashlib.sha256(namespaced_payload).hexdigest()

    return sha256_encoder


class RAGHandler:
    """高度なRAGパイプラインを管理・実行するハンドラクラス．"""

    def __init__(self):
        """ハンドラのインスタンスを初期化する．"""
        self.vector_store = None
        self.parent_docstore = {}
        self.llm = None
        self.json_llm = None
        self.reranker = None
        self.query_expansion_chain = None
        self.extraction_chain = None
        self.final_answer_chain = None
        logger.info("🚦 Initializing Advanced RAGHandler...")
        self._initialize()

    def _initialize(self):
        """RAGパイプラインに必要なコンポーネントをすべて初期化する．"""
        # --- 1. モデルと基本ツールのセットアップ ---
        os.makedirs(CACHE_PATH, exist_ok=True)
        fs_store = LocalFileStore(CACHE_PATH)

        logger.info("🔌 Connecting to Ollama at %s", OLLAMA_BASE_URL)
        ollama_embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )
        key_encoder = create_sha256_encoder(namespace=EMBEDDING_MODEL)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            ollama_embedder, fs_store, key_encoder=key_encoder
        )

        self.llm = ChatOllama(
            model=LLM_MODEL, temperature=0.2, top_k=40, base_url=OLLAMA_BASE_URL
        )
        self.json_llm = ChatOllama(
            model=LLM_MODEL, format="json", temperature=0, base_url=OLLAMA_BASE_URL
        )

        logger.info("🔍 Initializing Re-ranker: %s", RERANKER_MODEL)
        self.reranker = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)

        # --- 2. データベースの読み込み，または新規作成 ---
        chroma_settings = Settings(anonymized_telemetry=False)
        if os.path.exists(CHROMA_DB_PATH) and os.path.exists(PARENT_DOCS_PATH):
            logger.info("💾 Loading existing vector store and parent docstore...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=cached_embedder,
                client_settings=chroma_settings,
            )
            with open(PARENT_DOCS_PATH, "r", encoding="utf-8") as f:
                parent_docs_data = json.load(f)
                self.parent_docstore = {
                    int(k): Document(**v) for k, v in parent_docs_data.items()
                }
            logger.info("✅ Vector store and parent docstore loaded successfully.")
        else:
            self._create_sentence_window_db(cached_embedder, chroma_settings)

        # --- 3. LangChain Expression Language (LCEL) を用いて処理フロー（チェーン）を定義 ---
        logger.info("🔗 Initializing LCEL chains...")

        # 質問から複数の検索クエリを生成するチェーン
        query_prompt = PromptTemplate(
            template=query_expansion_template, input_variables=["question"]
        )
        self.query_expansion_chain = query_prompt | self.json_llm | JsonOutputParser()

        # 検索した文脈からノイズを除去（関連部分のみ抽出）するチェーン
        extraction_prompt = PromptTemplate(
            template=extraction_template, input_variables=["context", "question"]
        )
        self.extraction_chain = (
            {
                "context": lambda x: "\n---\n".join(
                    doc.page_content for doc in x["documents"]
                ),
                "question": lambda x: x["question"],
            }
            | extraction_prompt
            | self.llm
            | StrOutputParser()
        )

        # 最終的な回答を生成するチェーン
        answer_prompt = PromptTemplate(
            template=answer_generation_template, input_variables=["context", "question"]
        )
        self.final_answer_chain = (
            {
                "context": lambda x: x["cleaned_context"],
                "question": lambda x: x["question"],
            }
            | answer_prompt
            | self.llm
            | StrOutputParser()
        )
        logger.info("🚀 Advanced RAG Handler is ready.")

    def _create_sentence_window_db(self, embedder, settings):
        """文脈ウィンドウ戦略に基づき，ベクトルデータベースと親ドキュメントストアを新規作成する．"""
        logger.warning(
            "🕳️ Vector store not found. Creating a new one with Sentence-Window strategy..."
        )
        if not os.path.exists(DATA_PATH):
            raise FileNotFoundError(f"'{DATA_PATH}' not found.")

        documents = JSONLoader(
            file_path=DATA_PATH,
            jq_schema=".[]",
            content_key="content",
            json_lines=False,
            metadata_func=metadata_func,
        ).load()
        logger.info("📄 Loaded %d raw documents.", len(documents))

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )

        parent_documents = parent_splitter.split_documents(documents)
        self.parent_docstore = {i: doc for i, doc in enumerate(parent_documents)}
        logger.info("📑 Split into %d parent chunks.", len(parent_documents))

        child_documents = []
        for i, p_doc in enumerate(parent_documents):
            _sub_docs = child_splitter.split_documents([p_doc])
            for _doc in _sub_docs:
                _doc.metadata["parent_id"] = i  # 子チャンクに親IDを紐付ける
            child_documents.extend(_sub_docs)
        logger.info("📑 Created %d child chunks for retrieval.", len(child_documents))

        logger.info(
            "⏳ Creating vector store. This may take a while (first time only)..."
        )
        self.vector_store = Chroma(
            persist_directory=CHROMA_DB_PATH,
            embedding_function=embedder,
            client_settings=settings,
        )
        doc_batches = batch_generator(child_documents, BATCH_SIZE)
        for batch in tqdm(
            doc_batches,
            desc="📦 Embedding and adding child chunks",
            total=len(child_documents) // BATCH_SIZE
            + (1 if len(child_documents) % BATCH_SIZE > 0 else 0),
        ):
            self.vector_store.add_documents(batch)

        logger.info("💾 Saving parent docstore to %s", PARENT_DOCS_PATH)
        parent_docs_data = {k: v.dict() for k, v in self.parent_docstore.items()}
        with open(PARENT_DOCS_PATH, "w", encoding="utf-8") as f:
            json.dump(parent_docs_data, f, ensure_ascii=False, indent=4)

        logger.info("✅ Vector store and parent docstore created.")

    async def ask(self, query: str) -> Dict[str, Any]:
        """ユーザーの質問に対し，高度なRAGパイプラインを実行して回答を生成する．"""
        if not self.vector_store:
            return {"error": "Vector store not initialized."}

        try:
            # --- ステップ1: クエリ拡張 ---
            logger.info("🧠 Step 1: Expanding query...")
            if self.query_expansion_chain is None:
                raise RuntimeError("Query expansion chain is not initialized.")
            response_dict = await self.query_expansion_chain.ainvoke(
                {"question": query}
            )
            expanded_queries = response_dict.get("queries", [])
            all_queries = [query] + expanded_queries
            logger.info(f"⚡ Expanded queries: {all_queries}")

            # --- ステップ2: 並列ベクトル検索 ---
            logger.info("🔍 Step 2: Retrieving documents in parallel...")
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 15})
            tasks = [retriever.ainvoke(q) for q in all_queries]
            retrieved_results = await asyncio.gather(*tasks)

            # --- ステップ3: RAG-Fusion (RRF) ---
            logger.info("✨ Step 3: Fusing results with RRF...")
            fused_docs = self._rag_fusion(retrieved_results)
            logger.info(f"🗳️ Fused to {len(fused_docs)} unique documents.")

            # --- ステップ4: 再ランク付け ---
            logger.info("🧐 Step 4: Re-ranking top documents...")
            rerank_pairs = [(query, doc.page_content) for doc in fused_docs[:20]]
            if self.reranker is None:
                logger.warning(
                    "⚠️ Reranker is not initialized. Skipping reranking step."
                )
                final_child_docs = fused_docs[:4]
            else:
                scores = self.reranker.score(rerank_pairs)
                scored_docs = sorted(
                    zip(fused_docs, scores), key=lambda x: x[1], reverse=True
                )
                final_child_docs = [doc for doc, score in scored_docs[:4]]

            # --- ステップ5: 文脈ウィンドウ取得 ---
            logger.info("📖 Step 5: Fetching full context from parent docstore...")
            parent_ids = {doc.metadata["parent_id"] for doc in final_child_docs}
            final_context_docs = [
                self.parent_docstore.get(pid)
                for pid in parent_ids
                if self.parent_docstore.get(pid) is not None
            ]
            logger.info(
                f"📚 Using {len(final_context_docs)} parent documents for initial context."
            )

            # --- ステップ6: 文脈の抽出（ノイズ除去）---
            logger.info("✂️ Step 6: Extracting relevant sentences from context...")
            if self.extraction_chain is None:
                logger.warning(
                    "⚠️ Extraction chain is not initialized. Skipping extraction step."
                )
                cleaned_context = "\n---\n".join(
                    doc.page_content for doc in final_context_docs if doc is not None
                )
            else:
                cleaned_context = await self.extraction_chain.ainvoke(
                    {"documents": final_context_docs, "question": query}
                )
                if not cleaned_context or cleaned_context.isspace():
                    logger.warning(
                        "⚠️ No relevant information found after extraction. Falling back to original context."
                    )
                    cleaned_context = "\n---\n".join(
                        doc.page_content
                        for doc in final_context_docs
                        if doc is not None
                    )  # フォールバック
                else:
                    logger.info("✅ Context cleaned successfully.")

            # --- ステップ7: 最終回答の生成（統合）---
            logger.info("✍️ Step 7: Generating final answer from cleaned context...")
            if self.final_answer_chain is None:
                logger.warning(
                    "⚠️ Final answer chain is not initialized. Returning cleaned context as answer."
                )
                answer = cleaned_context
            else:
                answer = await self.final_answer_chain.ainvoke(
                    {"cleaned_context": cleaned_context, "question": query}
                )

            # 回答に参照元URLを追記
            unique_sources = list(
                {
                    doc.metadata.get("source")
                    for doc in final_context_docs
                    if doc is not None
                }
            )

            logger.info("📬 Successfully generated an answer.")
            return {"answer": answer, "source_documents": unique_sources}

        except Exception as e:
            logger.exception(
                "🔥 Error during Advanced RAG invocation for query: '%s'", query
            )
            return {"error": f"Failed to get an answer: {str(e)}"}

    def _rag_fusion(
        self, retrieved_results: List[List[Document]], k: int = 60
    ) -> List[Document]:
        """Reciprocal Rank Fusion (RRF)アルゴリズムで，複数の検索結果を統合・ランク付けする．"""
        fused_scores = {}
        doc_map = {}
        # 各検索結果リストをループ
        for docs in retrieved_results:
            # 各文書の順位(rank)に基づいてスコアを計算
            for rank, doc in enumerate(docs):
                # 文書を一意に識別するためのIDを生成
                doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:100]}"
                doc_map[doc_id] = doc
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                # RRFのスコア計算式: 1 / (rank + k)．kはランキングの影響度を調整する定数．
                fused_scores[doc_id] += 1 / (rank + k)

        # 総合スコアで降順にソート
        reranked_results = sorted(
            fused_scores.items(), key=lambda x: x[1], reverse=True
        )
        # ソートされたIDのリストから，元のDocumentオブジェクトのリストを再構築して返す
        return [doc_map[doc_id] for doc_id, score in reranked_results]


# FastAPIアプリケーションから利用するためのシングルトンインスタンス
rag_handler_instance = RAGHandler()
