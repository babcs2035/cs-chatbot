"use client";

import { useState, useRef, useEffect, FormEvent } from "react";
import ReactMarkdown from "react-markdown";

// --- 型定義 ---
// チャットのメッセージ一件を表す型
type Message = {
  id: number;
  role: "user" | "bot";
  text: string;
  sources?: string[]; // ボットの回答にのみ含まれる参照元URLのリスト
};

// --- アイコンコンポーネント ---
// アイコンをインラインSVGとして定義することで，外部依存をなくします
const SendIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="m22 2-7 20-4-9-9-4Z" />
    <path d="M22 2 11 13" />
  </svg>
);

const BotIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 8V4H8" />
    <rect width="16" height="12" x="4" y="8" rx="2" />
    <path d="M2 14h2" /><path d="M20 14h2" /><path d="M15 13v2" /><path d="M9 13v2" />
  </svg>
);

const UserIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" />
  </svg>
);

const LinkIcon = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.72" /><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.72-1.72" />
  </svg>
);


// --- メインコンポーネント ---
export default function ChatPage() {
  // --- 状態管理 ---
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 0,
      role: "bot",
      text: "こんにちは！東京大学の情報システムについて，何でも質問してください．\n\nまずは，上のボタンから知識ベースを構築・更新してください．",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isScraping, setIsScraping] = useState(false);
  const [scrapeStatus, setScrapeStatus] = useState<string | null>(null);

  const chatContainerRef = useRef<HTMLDivElement>(null);

  // 新しいメッセージが追加されたら，チャットの表示を一番下までスクロールする
  useEffect(() => {
    chatContainerRef.current?.scrollTo({
      top: chatContainerRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  // --- イベントハンドラ ---
  const handleScrape = async () => {
    setIsScraping(true);
    setScrapeStatus("スクレイピングと知識ベースの構築を開始します... (数分かかる場合があります)");
    setError(null);
    try {
      const response = await fetch("http://localhost:8000/scrape", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          urls: [
            "https://utelecon.adm.u-tokyo.ac.jp/",
            "https://www.sodan.ecc.u-tokyo.ac.jp/",
          ],
        }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "スクレイピングに失敗しました．");
      }
      const result = await response.json();
      setScrapeStatus(`✅ ${result.message}`);
    } catch (err: any) {
      setError(`スクレイピングエラー: ${err.message}`);
      setScrapeStatus(null);
    } finally {
      setIsScraping(false);
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { id: Date.now(), role: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: userMessage.text }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "APIからの応答エラーです．");
      }
      const data = await response.json();
      const botMessage: Message = {
        id: Date.now() + 1,
        role: "bot",
        text: data.answer,
        sources: data.source_documents,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };

  // --- レンダリング ---
  return (
    <div className="flex flex-col h-screen bg-gray-900 text-white font-sans">
      <header className="p-4 border-b border-gray-700 shadow-md bg-gray-800">
        <h1 className="text-xl font-bold text-center text-gray-200">
          cs-chatbot
        </h1>
      </header>

      <div className="p-4 bg-gray-800 border-b border-gray-700 text-center">
        <button
          onClick={handleScrape}
          disabled={isScraping}
          className="px-4 py-2 bg-indigo-600 rounded-lg hover:bg-indigo-700 disabled:bg-indigo-900 disabled:cursor-not-allowed transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-opacity-50"
        >
          {isScraping ? "知識ベースを構築中..." : "知識ベースを構築/更新"}
        </button>
        {scrapeStatus && <p className="mt-2 text-sm text-gray-400">{scrapeStatus}</p>}
      </div>

      <main ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg) => (
          <div key={msg.id} className={`flex items-start gap-4 ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'bot' && <BotIcon className="w-8 h-8 flex-shrink-0 text-indigo-400 mt-1" />}
            <div className={`rounded-xl p-4 max-w-2xl ${msg.role === 'user' ? 'bg-blue-600' : 'bg-gray-800'}`}>
              <div className="prose prose-invert prose-p:text-gray-300 prose-a:text-cyan-400 hover:prose-a:text-cyan-300 prose-strong:text-white">
                <ReactMarkdown
                  components={{ a: ({ node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}
                >
                  {msg.text}
                </ReactMarkdown>
              </div>

              {msg.role === 'bot' && msg.sources && msg.sources.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-600/50">
                  <h3 className="text-sm font-semibold text-gray-400 mb-2">参考にした情報源:</h3>
                  <ul className="space-y-2">
                    {msg.sources.map((source, index) => (
                      <li key={index} className="text-sm flex items-center gap-2">
                        <LinkIcon className="w-4 h-4 text-gray-500 flex-shrink-0" />
                        <a
                          href={source}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-cyan-500 hover:text-cyan-400 hover:underline truncate"
                          title={source}
                        >
                          {source}
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
            {msg.role === 'user' && <UserIcon className="w-8 h-8 flex-shrink-0 text-blue-400 mt-1" />}
          </div>
        ))}
        {isLoading && (
          <div className="flex items-start gap-4">
            <BotIcon className="w-8 h-8 flex-shrink-0 text-indigo-400 animate-pulse" />
            <div className="rounded-xl p-4 bg-gray-800">
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse delay-75"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-pulse delay-150"></div>
              </div>
            </div>
          </div>
        )}
      </main>

      <footer className="p-4 border-t border-gray-700 bg-gray-800">
        {error && <p className="text-red-500 text-center text-sm mb-2">エラー: {error}</p>}
        <form onSubmit={handleSubmit} className="flex items-center max-w-4xl mx-auto">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="質問を入力してください..."
            className="flex-1 p-3 bg-gray-700 rounded-l-lg border-none focus:ring-2 focus:ring-indigo-500 focus:outline-none placeholder-gray-400"
            disabled={isLoading}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            className="p-3 bg-indigo-600 rounded-r-lg hover:bg-indigo-700 disabled:bg-indigo-900 disabled:cursor-not-allowed transition-colors"
          >
            <SendIcon className="w-6 h-6" />
          </button>
        </form>
      </footer>
    </div>
  );
}
