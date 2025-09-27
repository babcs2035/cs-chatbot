# ステージ1: 依存関係インストール用
FROM python:3.12-slim as builder

# uvインストール
RUN pip install uv

WORKDIR /app

# 依存関係ファイルコピー
COPY requirements.txt .

# 仮想環境作成＆依存関係インストール
RUN uv venv && uv pip install --no-cache-dir -r requirements.txt

# ステージ2: 実行環境
FROM python:3.12-slim

WORKDIR /app

# 非rootユーザー作成
RUN useradd --create-home appuser

# dbディレクトリ作成＆権限付与
RUN mkdir -p /app/db && chown appuser:appuser /app/db

USER appuser

# 仮想環境コピー
COPY --from=builder /app/.venv /app/.venv

# 仮想環境パス追加
ENV PATH="/app/.venv/bin:$PATH"

# アプリケーションコード・データコピー
COPY ./fastapi_app /app
COPY ./cache /app/cache
COPY ./data /app/data
COPY ./db /app/db

# 権限変更
USER root
RUN chown -R appuser:appuser /app
USER appuser

# ポート公開
EXPOSE 8100

# dataフォルダーが空ならscraper.py実行→FastAPI起動
CMD ["/bin/sh", "-c", "if [ ! \"$(ls -A /app/data 2>/dev/null)\" ]; then python /app/scraper.py; fi && uvicorn main:app --host 0.0.0.0 --port 8100"]
