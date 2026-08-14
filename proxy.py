#!/usr/bin/env python3
"""OpenAI API proxy server. No third-party dependencies."""

import http.server
import os
import random
import threading
import time
import urllib.request
import urllib.error
from http.server import HTTPServer


# ---------------------------------------------------------------------------
# Minimal YAML parser (supports only the subset used in .env.yml)
# ---------------------------------------------------------------------------

def parse_yaml(text: str) -> dict:
    result = {}
    current_list_key = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith('#'):
            current_list_key = None
            continue
        # list item
        stripped = line.lstrip()
        if stripped.startswith('- '):
            if current_list_key is not None:
                result[current_list_key].append(stripped[2:].strip().strip('"').strip("'"))
            continue
        # key: value
        if ':' in line:
            key, _, val = line.partition(':')
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val == '':
                result[key] = []
                current_list_key = key
            else:
                current_list_key = None
                # try numeric
                try:
                    result[key] = int(val)
                except ValueError:
                    try:
                        result[key] = float(val)
                    except ValueError:
                        result[key] = val
    return result


def load_config(path: str = '.env.yml') -> dict:
    # Load from .env.yml if it exists (fallback for local dev)
    cfg = {}
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            cfg = parse_yaml(f.read())

    cfg.setdefault('base_url', 'https://api.openai.com')
    cfg.setdefault('port', 8080)
    cfg.setdefault('fail_threshold', 3)
    cfg.setdefault('ban_duration', 600)
    cfg.setdefault('api_keys', [])

    # Environment variables override config file (for Railway / cloud deploy)
    port_env = os.environ.get('PORT')
    if port_env:
        cfg['port'] = int(port_env)
    base_url_env = os.environ.get('BASE_URL')
    if base_url_env:
        cfg['base_url'] = base_url_env
    api_keys_env = os.environ.get('API_KEYS')
    if api_keys_env:
        cfg['api_keys'] = [k.strip() for k in api_keys_env.split(',') if k.strip()]
    fail_threshold_env = os.environ.get('FAIL_THRESHOLD')
    if fail_threshold_env:
        cfg['fail_threshold'] = int(fail_threshold_env)
    ban_duration_env = os.environ.get('BAN_DURATION')
    if ban_duration_env:
        cfg['ban_duration'] = int(ban_duration_env)

    return cfg


# ---------------------------------------------------------------------------
# Key pool with failure tracking and temporary ban
# ---------------------------------------------------------------------------

class KeyPool:
    def __init__(self, keys: list, fail_threshold: int, ban_duration: int):
        self._keys = list(keys)
        self._threshold = fail_threshold
        self._ban_duration = ban_duration
        self._failures: dict[str, int] = {k: 0 for k in keys}
        self._banned_until: dict[str, float] = {}
        self._success_count: dict[str, int] = {k: 0 for k in keys}
        self._total_count: dict[str, int] = {k: 0 for k in keys}
        self._lock = threading.Lock()

    def pick(self) -> str | None:
        with self._lock:
            now = time.time()
            available = [
                k for k in self._keys
                if now >= self._banned_until.get(k, 0)
            ]
            if not available:
                # all banned — return least-recently-banned as fallback
                if not self._keys:
                    return None
                return min(self._keys, key=lambda k: self._banned_until.get(k, 0))
            if len(available) == 1:
                return available[0]
            # 0.2 explore (random), 0.8 exploit (成功率前1/4中随机选)
            if random.random() < 0.2:
                return random.choice(available)
            # exploit: pick from top-1/4 by success rate to spread load
            sorted_keys = sorted(
                available,
                key=lambda k: self._success_count[k] / self._total_count[k]
                    if self._total_count[k] > 0 else 0,
                reverse=True,
            )
            top_n = max(1, len(sorted_keys) // 4)
            return random.choice(sorted_keys[:top_n])

    def report_success(self, key: str):
        with self._lock:
            self._failures[key] = 0
            self._success_count[key] = self._success_count.get(key, 0) + 1
            self._total_count[key] = self._total_count.get(key, 0) + 1

    def report_failure(self, key: str):
        with self._lock:
            self._failures[key] = self._failures.get(key, 0) + 1
            self._total_count[key] = self._total_count.get(key, 0) + 1
            if self._failures[key] >= self._threshold:
                ban_until = time.time() + self._ban_duration
                self._banned_until[key] = ban_until
                self._failures[key] = 0
                print(f"[KeyPool] key ...{key[-6:]} banned for {self._ban_duration}s")


# ---------------------------------------------------------------------------
# Proxy handler
# ---------------------------------------------------------------------------

class ProxyHandler(http.server.BaseHTTPRequestHandler):
    # set by main()
    key_pool: KeyPool | None = None
    base_url: str = ''

    def log_message(self, format, *args):
        print(f"[{self.address_string()}] {format % args}")

    def _proxy(self):
        assert self.key_pool is not None
        pool = self.key_pool
        key = pool.pick()
        if key is None:
            self.send_error(503, "No available API keys")
            return

        target_url = self.base_url.rstrip('/') + self.path

        # collect request body
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # build upstream request headers
        headers = {}
        for k, v in self.headers.items():
            lower = k.lower()
            if lower in ('host', 'authorization', 'content-length'):
                continue
            headers[k] = v
        headers['Authorization'] = f'Bearer {key}'
        if body:
            headers['Content-Length'] = str(len(body))

        req = urllib.request.Request(target_url, data=body, headers=headers, method=self.command)

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                pool.report_success(key)
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() == 'transfer-encoding':
                        continue
                    self.send_header(k, v)
                self.end_headers()

                # stream response
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()

        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429):
                pool.report_failure(key)
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() == 'transfer-encoding':
                    continue
                self.send_header(k, v)
            self.end_headers()
            body_err = e.read()
            if body_err:
                self.wfile.write(body_err)
        except Exception as e:
            pool.report_failure(key)
            self.send_error(502, f"Upstream error: {e}")

    def do_GET(self):    self._proxy()
    def do_POST(self):   self._proxy()
    def do_PUT(self):    self._proxy()
    def do_DELETE(self): self._proxy()
    def do_PATCH(self):  self._proxy()
    def do_OPTIONS(self):self._proxy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    cfg_path = os.environ.get('PROXY_CONFIG', '.env.yml')
    cfg = load_config(cfg_path)

    if not cfg['api_keys']:
        raise SystemExit("No api_keys configured in .env.yml")

    pool = KeyPool(cfg['api_keys'], cfg['fail_threshold'], cfg['ban_duration'])

    ProxyHandler.key_pool = pool
    ProxyHandler.base_url = cfg['base_url']

    port = int(cfg['port'])
    server = HTTPServer(('0.0.0.0', port), ProxyHandler)
    print(f"OpenAI proxy listening on http://0.0.0.0:{port}")
    print(f"Upstream: {cfg['base_url']}")
    print(f"Keys: {len(cfg['api_keys'])}, fail_threshold={cfg['fail_threshold']}, ban_duration={cfg['ban_duration']}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == '__main__':
    main()
