"""
FunASR-Nano-2512 Client Web Server
作者：凌封
来源：https://aibook.ren (AI全书)
"""
import http.server
import socketserver
import os
import socket

PORT = 9003
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    ip = get_ip_address()
    print(f"Serving HTTP on 0.0.0.0 port {PORT} ...")
    print(f"Server directory: {DIRECTORY}")
    print("\n" + "="*60)
    print(f"访问地址:")
    print(f"  Local:   http://localhost:{PORT}")
    print(f"  Network: http://{ip}:{PORT}")
    print("="*60)
    print("="*60)
    print("\n[使用说明]")
    print("1. 【推荐】本地电脑测试：")
    print(f"   请直接访问: http://localhost:{PORT}")
    print("   无需任何配置，浏览器会直接允许麦克风权限。")
    print("\n2. 【高级】远程访问测试：")
    print(f"   如果您通过 IP (http://{ip}:{PORT}) 访问，浏览器可能因非 HTTPS 禁止麦克风。")
    print("   解决办法: 在 Chrome 地址栏输入 chrome://flags/#unsafely-treat-insecure-origin-as-secure")
    print(f"   填入 http://{ip}:{PORT} 并启用。")
    print("="*60 + "\n")

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")
