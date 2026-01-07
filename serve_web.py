#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
简单的 HTTP 服务器，用于提供 web_client.html
"""

import http.server
import socketserver
import os
import sys

PORT = 9003

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # 添加 CORS 头，允许跨域访问
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

if __name__ == "__main__":
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"HTTP 服务器启动在 http://0.0.0.0:{PORT}")
        print(f"请在浏览器中访问: http://localhost:{PORT}/web_client.html")
        print(f"或访问: http://您的服务器IP:{PORT}/web_client.html")
        print("按 Ctrl+C 停止服务器")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n服务器已停止")
            sys.exit(0)

