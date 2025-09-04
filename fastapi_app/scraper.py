import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import json
import time
from tqdm import tqdm

# 訪問済みURLを記録するセット
visited_urls = set()

def scrape_page(url, base_domain):
    """指定されたURLのページからテキストコンテンツを抽出する"""
    if url in visited_urls:
        return None, []

    # PDFや画像などの非HTMLファイルへのリンクはリクエスト前にスキップ
    if url.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png', '.gif', '.zip', '.gz', '.rar')):
        print(f"Skipping non-HTML link: {url}")
        visited_urls.add(url)
        return None, []

    visited_urls.add(url)

    try:
        # サーバーへの負荷を軽減するために少し待機
        time.sleep(0.2)
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Content-TypeがHTMLでない場合は解析をスキップ
        content_type = response.headers.get('Content-Type', '')
        if 'text/html' not in content_type:
            print(f"Skipping non-HTML content type '{content_type}' at {url}")
            return None, []
            
        response.encoding = response.apparent_encoding # 文字化け対策
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None, []

    soup = BeautifulSoup(response.text, 'html.parser')

    # 不要なタグ（スクリプト、スタイルシート、ナビゲーション、フッターなど）を削除
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()

    # 本文テキストを取得
    text = soup.get_text(separator='\n', strip=True)

    # ページ内のリンクを収集
    links = []
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # 相対URLを絶対URLに変換
        full_url = urljoin(url, str(href))
        # URLのフラグメント（#...）を削除
        full_url = full_url.split('#')[0]
        
        # 同じドメイン内のURLのみを対象とし、"/en/"を含むURLは除外
        if urlparse(full_url).netloc == base_domain and not ("/en/" in full_url):
            links.append(full_url)

    return {"url": url, "content": text}, links

def crawl_website(start_urls, data_dir="data"):
    """ウェブサイト全体をクロールし、テキストデータをJSONファイルに保存する"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    all_scraped_data = []
    urls_to_visit = list(start_urls)
    
    # 訪問済みセットを初期化
    global visited_urls
    visited_urls = set()

    with tqdm(total=len(urls_to_visit), desc="Crawling Progress", unit="pages") as pbar:
        while urls_to_visit:
            url = urls_to_visit.pop(0)
            if url in visited_urls:
                # キューに重複があった場合、pbarの総数を調整
                pbar.total = len(visited_urls) + len(urls_to_visit)
                pbar.update(1)
                continue

            base_domain = urlparse(url).netloc
            pbar.set_description(f"Crawling {url[:70]}...")

            page_data, new_links = scrape_page(url, base_domain)

            if page_data:
                all_scraped_data.append(page_data)

            newly_discovered_links = 0
            if new_links:
                for link in new_links:
                    if link not in visited_urls and link not in urls_to_visit:
                        urls_to_visit.append(link)
                        newly_discovered_links += 1
            
            # プログレスバーの総数を更新
            pbar.total = len(visited_urls) + len(urls_to_visit)
            pbar.update(1)
            pbar.set_postfix_str(f"Queue: {len(urls_to_visit)}, Found: {newly_discovered_links}")

    # 収集したデータをファイルに保存
    output_path = os.path.join(data_dir, "scraped_data.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_scraped_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nScraping finished. {len(all_scraped_data)} pages saved to {output_path}")
    return output_path

if __name__ == '__main__':
    # このスクリプトを直接実行した場合のテスト用コード
    target_urls = [
        "https://utelecon.adm.u-tokyo.ac.jp/",
        "https://www.sodan.ecc.u-tokyo.ac.jp/"
    ]
    crawl_website(target_urls)
