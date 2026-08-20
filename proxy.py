#!/usr/bin/env python3
"""OpenAI API proxy server. No third-party dependencies."""

import http.server
import os
import random
import threading
import time
import urllib.request
import urllib.error
import collections
import json
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
    cfg.setdefault('ban_step', 3)
    cfg.setdefault('max_ban', 86400)
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
    ban_step_env = os.environ.get('BAN_STEP')
    if ban_step_env:
        cfg['ban_step'] = int(ban_step_env)
    max_ban_env = os.environ.get('MAX_BAN')
    if max_ban_env:
        cfg['max_ban'] = int(max_ban_env)

    return cfg

# ---------------------------------------------------------------------------
# Key pool with failure tracking and temporary ban
# ---------------------------------------------------------------------------

class KeyPool:
    # EWMA + epsilon-greedy parameters
    _ALPHA = 0.1       # learning rate
    _EPSILON = 0.02    # exploration probability
    _DELTA = 0.02      # min weight floor
    _INIT_P = 0.9      # initial success rate estimate
    _INIT_LATENCY = 1.0 # initial latency estimate (seconds)
    _HISTORY_SIZE = 20  # max stored call records per key

    def __init__(self, keys: list, fail_threshold: int, ban_duration: int, storage_path: str | None = None, ban_step: int = 3, max_ban: int = 86400):
        self._keys = list(keys)
        self._threshold = fail_threshold
        self._ban_duration = ban_duration
        self._ban_step = ban_step
        self._max_ban = max_ban
        self._failures: dict[str, int] = {k: 0 for k in keys}
        self._banned_until: dict[str, float] = {}
        self._ewma_p: dict[str, float] = {k: self._INIT_P for k in keys}
        self._ewma_latency: dict[str, float] = {k: self._INIT_LATENCY for k in keys}
        self._history: dict[str, collections.deque] = {
            k: collections.deque(maxlen=self._HISTORY_SIZE) for k in keys
        }
        self._lock = threading.Lock()
        self._storage_path = storage_path
        self._history_file: str | None = None
        if storage_path:
            os.makedirs(storage_path, exist_ok=True)
            self._history_file = os.path.join(storage_path, 'history.json')
            self._load()

    def _load(self):
        """Load state from storage directory."""
        if not self._history_file or not os.path.exists(self._history_file):
            return
        try:
            with open(self._history_file, encoding='utf-8') as f:
                data = json.load(f)
            for key, records in data.get('history', {}).items():
                if key in self._history:
                    for ts, status, latency in records:
                        self._history[key].append((ts, status, latency))
                    self._recalc_ewma(key)
            for key, until in data.get('banned_until', {}).items():
                if key in self._failures:
                    self._banned_until[key] = until
            for key, count in data.get('failures', {}).items():
                if key in self._failures:
                    self._failures[key] = count
            print(f"[KeyPool] loaded state from {self._history_file}")
        except (json.JSONDecodeError, OSError) as e:
            print(f"[KeyPool] warning: failed to load state: {e}")

    def _recalc_ewma(self, key: str):
        """Recalculate EWMA success rate and latency from history."""
        records = list(self._history[key])
        if not records:
            return
        p = self._INIT_P
        lat = self._INIT_LATENCY
        for _, status, latency in records:
            if status:
                p = (1 - self._ALPHA) * p + self._ALPHA
            else:
                p = (1 - self._ALPHA) * p
            lat = (1 - self._ALPHA) * lat + self._ALPHA * latency
        self._ewma_p[key] = p
        self._ewma_latency[key] = lat

    def _persist(self):
        """Persist state to storage directory (called within lock)."""
        if not self._history_file:
            return
        data = {
            'history': {key: list(records) for key, records in self._history.items()},
            'banned_until': dict(self._banned_until),
            'failures': dict(self._failures),
        }
        tmp = self._history_file + '.tmp'
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            os.replace(tmp, self._history_file)
        except OSError as e:
            print(f"[KeyPool] warning: failed to persist state: {e}")

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
            # epsilon-greedy: explore vs exploit
            random.seed = now
            if random.random() < self._EPSILON:
                return random.choice(available)
            # exploit: weighted random by EWMA success rate / latency
            weights = [
                max(self._DELTA, self._ewma_p[k] / (self._ewma_latency[k] + 0.1))
                for k in available
            ]
            total = sum(weights)
            r = random.uniform(0, total)
            cumulative = 0
            for k, w in zip(available, weights):
                cumulative += w
                if r <= cumulative:
                    return k
            return available[-1]

    def report(self, key: str, success: bool, latency: float):
        with self._lock:
            now = time.time()
            self._history[key].append((now, success, latency))
            if success:
                self._failures[key] = 0
                self._ewma_p[key] = (1 - self._ALPHA) * self._ewma_p.get(key, self._INIT_P) + self._ALPHA
            else:
                self._failures[key] = self._failures.get(key, 0) + 1
                self._ewma_p[key] = (1 - self._ALPHA) * self._ewma_p.get(key, self._INIT_P)
                if self._failures[key] >= self._threshold:
                    ban_secs = min(int(self._failures[key] / self._ban_step) * self._ban_duration, self._max_ban)
                    ban_until = time.time() + ban_secs
                    self._banned_until[key] = ban_until
                    print(f"[KeyPool] key ...{key[-6:]} banned for {ban_secs}s")
            self._ewma_latency[key] = (
                (1 - self._ALPHA) * self._ewma_latency.get(key, self._INIT_LATENCY)
                + self._ALPHA * latency
            )
            self._persist()


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

        t0 = time.time()
        try:
            with urllib.request.urlopen(req) as resp:
                pool.report(key, True, time.time() - t0)
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
                pool.report(key, False, time.time() - t0)
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
            pool.report(key, False, time.time() - t0)
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

    pool = KeyPool(cfg['api_keys'], cfg['fail_threshold'], cfg['ban_duration'], os.environ.get('STORAGE_PATH'), cfg['ban_step'], cfg['max_ban'])

    ProxyHandler.key_pool = pool
    ProxyHandler.base_url = cfg['base_url']

    port = int(cfg['port'])
    server = HTTPServer(('0.0.0.0', port), ProxyHandler)
    print(f"OpenAI proxy listening on http://0.0.0.0:{port}")
    print(f"Upstream: {cfg['base_url']}")
    print(f"Keys: {len(cfg['api_keys'])}, fail_threshold={cfg['fail_threshold']}, ban_duration={cfg['ban_duration']}s, ban_step={cfg['ban_step']}, max_ban={cfg['max_ban']}s")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == '__main__':
    main()
