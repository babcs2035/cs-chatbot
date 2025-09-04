import os
import jq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
# Ollama関連のクラスを新しい`langchain-ollama`パッケージからインポート
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# --- 定数定義 ---
# 環境変数から設定を読み込むか、デフォルト値を使用
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH = "./data/scraped_data.json"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class RAGHandler:
    """
    RAG (Retrieval-Augmented Generation) パイプラインを管理するクラス。
    データの読み込み、ベクトル化、QAチェーンの構築と実行を担当する。
    """
    def __init__(self):
        """
        RAGHandlerのインスタンスを初期化する。
        """
        self.vector_store = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """
        ベクトルストアとQAチェーンの初期化処理を行う。
        - 既存のベクトルストアがあれば読み込み、なければデータソースから新規作成する。
        - LLMとリトリーバーを組み合わせたQAチェーンを構築する。
        """
        # 1. 埋め込みモデルの初期化
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        # 2. ベクトルストアの読み込みまたは作成
        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            print("✅ Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH, 
                embedding_function=embeddings
            )
        else:
            print("⏳ Creating new vector store...")
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(
                    f"'{DATA_PATH}' not found. Please run scraper.py to generate the data file."
                )
            
            # JSONファイルからドキュメントを読み込む
            loader = JSONLoader(
                file_path=DATA_PATH,
                jq_schema='.[] | .content', # JSON配列から各要素の'content'キーを抽出
                text_content=True,
                json_lines=False 
            )
            documents = loader.load()
            
            # テキストをチャンク（小さな断片）に分割
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, 
                chunk_overlap=CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(documents)
            
            # チャンクをベクトル化し、ChromaDBに保存
            self.vector_store = Chroma.from_documents(
                documents=docs, 
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH
            )
            print(f"✅ Vector store created with {len(docs)} document chunks.")
        
        # 3. 大規模言語モデル（LLM）の初期化
        llm = ChatOllama(model=LLM_MODEL)
        
        # 4. QAチェーンの作成
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff", # 取得した文書をすべてコンテキストとして渡す方式
            retriever=self.vector_store.as_retriever(),
            return_source_documents=True # 回答の根拠となったソース文書も返す設定
        )
        print("🚀 RAG Handler is ready.")

    def ask(self, query: str):
        """
        ユーザーからの質問を受け取り、QAチェーンを実行して回答を生成する。
        
        Args:
            query (str): ユーザーからの質問テキスト。

        Returns:
            dict: 回答とソースドキュメントを含む辞書。
        """
        if self.qa_chain is None:
            print("❌ QA chain is not initialized.")
            return {"error": "QA chain is not initialized."}
        
        try:
            print(f"🔄 Processing query: {query}")
            response = self.qa_chain.invoke({"query": query})
            
            # ソースドキュメントのメタデータを整形
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    # メタデータが存在すればURLを取得、なければ'Unknown'
                    url = doc.metadata.get('source', 'Unknown')
                    sources.append(url)
            
            print("✅ Answer generated.")
            return {
                "answer": response.get("result"),
                # 重複を除いたソースURLのリスト
                "source_documents": list(set(sources))
            }
        except Exception as e:
            print(f"🔥 Error during QA chain invocation: {e}")
            return {"error": "Failed to get an answer from the QA chain."}

# --- シングルトンインスタンス ---
# FastAPIアプリケーション全体でRAGHandlerの単一インスタンスを共有するために、
# モジュールレベルでインスタンスを生成する。
rag_handler_instance = RAGHandler()
