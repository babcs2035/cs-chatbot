# cs-chatbot

FastAPI，LangChain，ChromaDB，Ollama (LLM) を用いた RAG 型チャットボット

## プロジェクト概要

cs-chatbotは，FastAPI，LangChain，ChromaDB，Ollama (LLM) を活用した RAG 型チャットボットシステムである．
Web スクレイピング，埋め込み，ベクトル検索，Next.js によるチャット UI を備えている．
Docker 化された Ollama により，GPU による高速推論も可能である．

## セットアップ

### 必要なもの

- Docker と Docker Compose  
- Python 3.11 以上 (ローカル開発用)   
- Node.js 18 以上と pnpm (フロントエンド用)   
- NVIDIA GPU (Ollama の高速化用，任意)   

### 1．リポジトリのクローン

```bash
git clone https://github.com/babcs2035/cs-chatbot.git
cd cs-chatbot
```


### 2．セットアップ・開発・デプロイ (mise推奨)

このプロジェクトは [mise](https://github.com/jdx/mise) の scripts 機能で主要なコマンドを統一管理している．

#### 依存パッケージのインストール
```bash
mise run setup
```

#### 開発サーバ起動 (バックエンド＋フロントエンド)
```bash
mise run dev
```

#### デプロイ (Docker Compose)
```bash
mise run deploy
```

## 開発方法


### 開発コマンド一覧 (mise)

| コマンド        | 内容                                       |
| --------------- | ------------------------------------------ |
| mise run setup  | フロントエンド依存パッケージインストール   |
| mise run dev    | バックエンド＋フロントエンド開発サーバ起動 |
| mise run deploy | Docker Compose による本番デプロイ          |

開発サーバ起動後，ブラウザで http://localhost:3000 を開くとチャット UI にアクセスできる．


## デプロイ

### Docker Compose (推奨) 

```bash
docker compose up --build -d
```

## 使い方

1．知識ベース構築：フロントエンドのボタンで対象 URL をスクレイピングし，知識ベースを作成・更新する．  
2．チャット：チャット UI で質問を入力すると，関連情報を検索し，LLM が回答する．

主な API エンドポイント：  
- `POST /chat` — 質問  
- `POST /scrape` — URL スクレイピングと知識ベース更新  

## ディレクトリ構成

```
cs-chatbot/
├── fastapi_app/      # FastAPI バックエンド (RAG，スクレイピング，ベクトルDB) 
├── web_app/          # Next.js フロントエンド (チャット UI) 
├── db/               # ChromaDB ベクトルデータベース
├── data/             # スクレイピングデータ
├── cache/            # 埋め込み・LLM キャッシュ
├── requirements.txt  # Python 依存パッケージ
├── Dockerfile        # バックエンド Docker ビルド
├── docker-compose.yml# サービス構成
├── pyproject.toml    # Python プロジェクト設定
└── README.md         # このファイル
```
