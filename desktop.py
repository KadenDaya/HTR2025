from PySide6.QtWidgets import QApplication, QMainWindow, QTextEdit
from PySide6.QtCore import QThread, Signal
import sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from speech import tts

class LogBridge(QThread):
    message = Signal(str)

class Handler(BaseHTTPRequestHandler):
    bridge = None

    def do_GET(self):
        q = urllib.parse.urlparse(self.path).query
        msg = urllib.parse.parse_qs(q).get("msg", [""])[0]
        raw_log = f"{self.address_string()} - - [{self.log_date_time_string()}] \"{self.requestline}\" 200 -"
        if Handler.bridge:
            Handler.bridge.message.emit(raw_log)
            if msg:
                Handler.bridge.message.emit(msg)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Metal Chair 1.0")

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.setCentralWidget(self.log_view)

        self.bridge = LogBridge()
        self.bridge.message.connect(self.log)
        threading.Thread(target=self.start_server, args=(8080,), daemon=True).start()

    def log(self, text):
        self.log_view.append(text)
        if not "GET /?msg=" in text:
            tts(text)

    def start_server(self, port):
        Handler.bridge = self.bridge
        server = HTTPServer(("0.0.0.0", port), Handler)
        server.serve_forever()

app = QApplication(sys.argv)
window = MainWindow()
window.showMaximized()
sys.exit(app.exec())
