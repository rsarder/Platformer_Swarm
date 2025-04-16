import http.server
import socketserver
import threading
import time
from pathlib import Path
from datetime import datetime
from playwright.sync_api import sync_playwright

class StoppableHTTPServer(socketserver.TCPServer):
    allow_reuse_address = True

def start_server(httpd):
    """Serve until shutdown is called."""
    print(f"[SERVER] Serving at http://localhost:{httpd.server_address[1]}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print("[SERVER] Server shut down.")

def capture_browser_errors(url, wait_time_ms=5000, log_path="browser_errors.md"):
    errors = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()

        def handle_console(msg):
            if msg.type == "error" and "favicon.ico" not in msg.text:
                loc = msg.location or {}
                file = loc.get("url", "Unknown file")
                line = loc.get("lineNumber", "?")
                column = loc.get("columnNumber", "?")

                error_text = f"""[CONSOLE ERROR]
                                Type: {msg.type}
                                Message: {msg.text}
                                Source: {file}:{line}:{column}
                                """
                print(error_text)
                errors.append(error_text)

        def handle_page_error(exception):
            full_error = f"""[PAGE ERROR]
                            Message: {exception.message}
                            Name: {getattr(exception, 'name', 'Unknown')}
                            Stack: {getattr(exception, 'stack', 'No stack trace')}
                            """
            print(full_error)
            errors.append(full_error)

        page.on("console", handle_console)
        page.on("pageerror", handle_page_error)

        print(f"[BROWSER] Navigating to: {url}")
        page.goto(url)
        page.wait_for_timeout(wait_time_ms)
        browser.close()
        print("[BROWSER] Closed.")

    if errors:
        log_file = Path.cwd() / log_path
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_file, "w") as f:
            f.write(f"# Browser Console Errors\n\n")
            f.write(f"_Captured at {timestamp}_\n\n")
            for err in errors:
                f.write(f"{err}\n")
        print(f"[LOG] Errors written to {log_file}")
    else:
        print("[LOG] No browser errors captured.")



def run_browser_server_and_capture_errors(
    filename="index.html",
    port=8000,
    wait_time_ms=5000
):
    """Starts a local server, launches a browser, captures console errors, and then shuts down the server."""
    url = f"http://localhost:{port}/{filename}"

    # ✅ Ensure favicon.ico exists
    favicon_path = Path.cwd() / "favicon.ico"
    if not favicon_path.exists():
        favicon_path.write_bytes(b"")
        print(f"[SETUP] Created blank favicon.ico at {favicon_path}")

    # ✅ Create stoppable server
    handler = http.server.SimpleHTTPRequestHandler
    httpd = StoppableHTTPServer(("", port), handler)

    # ✅ Start in background
    server_thread = threading.Thread(target=start_server, args=(httpd,), daemon=True)
    server_thread.start()

    # ✅ Wait for server to boot
    time.sleep(1)

    # ✅ Run browser test
    capture_browser_errors(url, wait_time_ms)

    # ✅ Shutdown server
    httpd.shutdown()
    server_thread.join()
    print("[SERVER] Server thread finished.")
