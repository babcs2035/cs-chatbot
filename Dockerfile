# ステージ1: ビルドステージ
# uvをインストールし、依存関係を仮想環境にインストールする
FROM python:3.11-slim as builder

# uvをインストール
RUN pip install uv

WORKDIR /app

# 依存関係ファイルをコピー
COPY requirements.txt .

# uvを使って仮想環境を作成し、依存関係をインストール
RUN uv venv && uv pip install --no-cache-dir -r requirements.txt

# ---

# ステージ2: 本番ステージ
# 最終的なアプリケーションイメージを作成する
FROM python:3.11-slim

WORKDIR /app

# 非rootユーザーを作成してセキュリティを向上
RUN useradd --create-home appuser
USER appuser

# ビルドステージから仮想環境（インストール済みパッケージ）をコピー
COPY --from=builder /app/.venv /app/.venv

# 仮想環境のパスを環境変数に追加
ENV PATH="/app/.venv/bin:$PATH"

# アプリケーションコードをコピー
COPY ./fastapi_app /app

# ポートを公開
EXPOSE 8000

# アプリケーションを起動（仮想環境内のuvicornが使用される）
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
