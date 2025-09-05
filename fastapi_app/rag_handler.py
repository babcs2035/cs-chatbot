import os
import jq
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# --- 定数定義 ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3:latest")
RERANKER_MODEL = "BAAI/bge-reranker-base"
CHROMA_DB_PATH = "./chroma_db"
DATA_PATH = "./data/scraped_data.json"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

prompt_template = """
### 指示
あなたは、提供された「コンテキスト情報」を深く理解し、質問に対して的確に回答する優秀なアシスタントです。
以下のルールを厳守してください。

1.  回答は、必ず「コンテキスト情報」に基づいて生成してください。
2.  情報をただ抜き出すのではなく、**統合・要約し、自然で分かりやすい文章で回答を作成**してください。
3.  質問に対する答えがコンテキスト情報に存在しない場合は、「その情報は見つかりませんでした。」と明確に回答してください。自身の知識で補完してはいけません。
4.  回答は日本語で行ってください。

### コンテキスト情報
{context}

### 質問
{question}

### 回答
"""


class RAGHandler:
    def __init__(self):
        self.vector_store = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(CHROMA_DB_PATH) and os.listdir(CHROMA_DB_PATH):
            print("✅ Loading existing vector store...")
            self.vector_store = Chroma(
                persist_directory=CHROMA_DB_PATH, embedding_function=embeddings
            )
        else:
            print(f"⏳ Creating new vector store using '{EMBEDDING_MODEL}'...")
            if not os.path.exists(DATA_PATH):
                raise FileNotFoundError(
                    f"'{DATA_PATH}' not found. Please run scraper.py first."
                )

            loader = JSONLoader(
                file_path=DATA_PATH,
                jq_schema=".[] | .content",
                text_content=True,
                json_lines=False,
            )
            documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(documents)

            self.vector_store = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=CHROMA_DB_PATH
            )
            print(f"✅ Vector store created with {len(docs)} document chunks.")

        llm = ChatOllama(model=LLM_MODEL, temperature=0.2, top_k=40)

        QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        base_retriever = self.vector_store.as_retriever(
            search_type="mmr", search_kwargs={"k": 10}
        )

        model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
        reranker = CrossEncoderReranker(model=model, top_n=3)

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=base_retriever
        )

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )
        print("🚀 RAG Handler is ready with fine-tuned parameters.")

    def ask(self, query: str):
        if self.qa_chain is None:
            return {"error": "QA chain is not initialized."}

        try:
            print(f"🔍 Query: {query}")
            response = self.qa_chain.invoke({"query": query})
            sources = [
                doc.metadata.get("source", "Unknown")
                for doc in response.get("source_documents", [])
            ]

            print(f"✅ Answer: {response.get('result')}")
            return {
                "answer": response.get("result"),
                "source_documents": list(set(sources)),
            }
        except Exception as e:
            return {"error": f"Failed to get an answer from the QA chain: {str(e)}"}


rag_handler_instance = RAGHandler()
