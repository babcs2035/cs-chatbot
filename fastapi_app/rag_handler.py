import os
import jq
import logging
import hashlib
from typing import Any, Dict, List, Generator
from tqdm import tqdm
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from chromadb.config import Settings

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


# --- 定数定義 ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_DB_PATH = "./db"
DATA_PATH = "./data/scraped_data.json"
CACHE_PATH = "./cache"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
BATCH_SIZE = 100

logger.info("🧠 Activating in-memory LLM cache...")
set_llm_cache(InMemoryCache())

prompt_template = """
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


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["source"] = record.get("url")
    return metadata


def batch_generator(
    data: List[Any], batch_size: int
) -> Generator[List[Any], None, None]:
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


def create_sha256_encoder(namespace: str):
    """指定された名前空間（モデル名）でプレフィックスを付けたSHA-256ハッシュを計算するエンコーダーを生成する"""

    def sha256_encoder(payload: str) -> str:
        # ペイロードの前に名前空間を追加することで，モデルごとにユニークなハッシュを保証
        namespaced_payload = namespace.encode() + b":" + payload.encode()
        return hashlib.sha256(namespaced_payload).hexdigest()

    return sha256_encoder


class RAGHandler:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        logger.info("🚦 Initializing RAGHandler...")
        self._initialize()

    def _initialize(self):
        fs_store = LocalFileStore(CACHE_PATH)

        logger.info("🔌 Connecting to Ollama at %s", OLLAMA_BASE_URL)
        ollama_embedder = OllamaEmbeddings(
            model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL
        )

        key_encoder = create_sha256_encoder(namespace=EMBEDDING_MODEL)
        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            ollama_embedder,
            fs_store,
            key_encoder=key_encoder,
        )

        chroma_settings = Settings(anonymized_telemetry=False)

        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            logger.info("💾 Loading existing vector store from %s...", CHROMA_DB_PATH)
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=cached_embedder,
                client_settings=chroma_settings,
            )
            logger.info("✅ Vector store loaded successfully.")
        else:
            logger.warning("🕳️ Vector store not found. Creating a new one...")
            if not os.path.exists(DATA_PATH):
                logger.error(
                    "🔥 '%s' not found. Please run scraper.py first.", DATA_PATH
                )
                raise FileNotFoundError(
                    f"'{DATA_PATH}' not found. Please run scraper.py first."
                )

            logger.info("📂 Loading documents from '%s'...", DATA_PATH)
            loader = JSONLoader(
                file_path=DATA_PATH,
                jq_schema=".[]",
                content_key="content",
                json_lines=False,
                metadata_func=metadata_func,
            )
            documents = loader.load()
            logger.info("📄 Loaded %d documents.", len(documents))

            logger.info(
                "✂️ Splitting documents into chunks (size: %d, overlap: %d)...",
                CHUNK_SIZE,
                CHUNK_OVERLAP,
            )
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(documents)
            logger.info("📑 Split into %d chunks.", len(docs))

            logger.info(
                "⏳ Creating vector store. This may take a while (first time only)..."
            )
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH,
                embedding_function=cached_embedder,
                client_settings=chroma_settings,
            )

            doc_batches = batch_generator(docs, BATCH_SIZE)

            for batch in tqdm(
                doc_batches,
                desc="📦 Embedding and adding chunks",
                total=len(docs) // BATCH_SIZE
                + (1 if len(docs) % BATCH_SIZE > 0 else 0),
            ):
                self.vector_store.add_documents(batch)

            logger.info("✅ Vector store created and persisted at %s", CHROMA_DB_PATH)

        llm = ChatOllama(
            model=LLM_MODEL, temperature=0.2, top_k=40, base_url=OLLAMA_BASE_URL
        )

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=prompt_template
        )

        logger.info("🔍 Initializing Re-ranker: %s", RERANKER_MODEL)
        base_retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )
        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        reranker = CrossEncoderReranker(model=model, top_n=4)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        logger.info("🔗 Initializing QA chain...")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )
        logger.info("🚀 RAG Handler is ready.")

    def ask(self, query: str) -> Dict[str, Any]:
        if self.qa_chain is None:
            logger.error("❌ QA chain is not initialized. Cannot process query.")
            return {"error": "QA chain is not initialized."}

        try:
            logger.info("📨 Received query: %s", query)
            response = self.qa_chain.invoke({"query": query})

            answer = response.get("result", "").strip()
            source_docs = response.get("source_documents", [])

            logger.info("📚 Retrieved %d documents after re-ranking.", len(source_docs))

            unique_sources = []
            if source_docs:
                seen_urls = set()
                for i, doc in enumerate(source_docs):
                    source_url = doc.metadata.get("source")
                    logger.info("  - Source %d: %s", i + 1, source_url)
                    if source_url and source_url not in seen_urls:
                        seen_urls.add(source_url)
                        unique_sources.append(source_url)

                if unique_sources:
                    answer += "\n\n---\n\n**参考にした情報源:**\n"
                    for url in unique_sources:
                        answer += f"- [{url}]({url})\n"

            logger.info("📬 Successfully generated an answer.")
            return {
                "answer": answer,
                "source_documents": unique_sources,
            }
        except Exception as e:
            logger.exception(
                "🔥 Error during QA chain invocation for query: '%s'", query
            )
            return {"error": f"Failed to get an answer from the QA chain: {str(e)}"}


rag_handler_instance = RAGHandler()
