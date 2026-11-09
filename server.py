"""零依赖 Web Demo：标准库 http.server + 单页前端。

用法：
    python server.py            # 打开 http://127.0.0.1:8000
"""

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import RecipeRAGSystem  # noqa: E402

RAG = None
LOCK = threading.Lock()


def get_rag() -> RecipeRAGSystem:
    global RAG
    if RAG is None:
        rag = RecipeRAGSystem()
        rag.initialize_system()
        rag.build_knowledge_base()
        RAG = rag
    return RAG


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # 静默访问日志
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path in ("/", "/index.html"):
            self._send(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/health":
            self._send(200, json.dumps({"ok": True}).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        question = (payload.get("question") or "").strip()
        if not question:
            self._send(400, json.dumps({"error": "question 不能为空"}, ensure_ascii=False).encode(),
                       "application/json")
            return
        try:
            with LOCK:
                answer = get_rag().ask_question(question, stream=False)
            body = json.dumps({"answer": answer}, ensure_ascii=False).encode("utf-8")
            self._send(200, body, "application/json; charset=utf-8")
        except Exception as exc:  # noqa: BLE001
            body = json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False).encode()
            self._send(500, body, "application/json; charset=utf-8")


def main() -> None:
    print("正在加载知识库（首次启动约 30-60 秒）...")
    get_rag()
    print("已就绪：http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
