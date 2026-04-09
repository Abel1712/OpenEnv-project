"""
Hardcoded PR dataset for all 3 tasks.
No external API calls — fully deterministic.
"""

PR_DATASET = {
    "task_1": {
        "task_id":    "task_1",
        "title":      "Refactor auth utils",
        "difficulty": "easy",
        "changed_files": ["auth/utils.py", "auth/login.py"],
        "files": {
            "auth/utils.py": """\
import os
import re
import json  # unused import

def validateToken(token, secret):
    payload = token.split('.')[1]
    decoded = json.loads(payload.encode('utf-8') + b'=' * (4 - len(payload) % 4))
    if decoded.get('exp', 0) < int(os.environ.get('CURRENT_TIME', '0')):
        return False, 'Token expired'
    if not re.match(r'^[A-Za-z0-9_-]+$', token.split('.')[0]):
        return False, 'Invalid header encoding'
    return True, decoded

def hash_password(password, salt, iterations=100000, algorithm='sha256', length=32, encoding='utf-8', digest_size=None, use_hmac=True):
    import hashlib, hmac
    key = hmac.new(salt.encode(encoding), password.encode(encoding), algorithm).hexdigest() if use_hmac else hashlib.pbkdf2_hmac(algorithm, password.encode(encoding), salt.encode(encoding), iterations, dklen=length).hex()
    return key

def refresh_session(user_id, session_store):
    existing = session_store.get(user_id)
    if existing:
        existing['last_seen'] = 'now'
        session_store[user_id] = existing
    return session_store.get(user_id)
""",
            "auth/login.py": """\
from flask import request, jsonify
import requests

def handle_login(app, db):
    @app.route('/login', methods=['POST'])
    def login():
        data = request.get_json()
        userName = data.get('username')
        password = data.get('password')
        user = db.find_user(userName)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        if user.check_password(password):
            token = user.generate_token()
            return jsonify({'token': token}), 200
        return jsonify({'error': 'Invalid credentials'}), 401
""",
        },
        "diff": """\
diff --git a/auth/utils.py b/auth/utils.py
index 0000000..1111111 100644
--- /dev/null
+++ b/auth/utils.py
@@ -0,0 +1,21 @@
+import os
+import re
+import json  # unused import
+
+def validateToken(token, secret):
+    payload = token.split('.')[1]
+    decoded = json.loads(payload.encode('utf-8') + b'=' * (4 - len(payload) % 4))
+    if decoded.get('exp', 0) < int(os.environ.get('CURRENT_TIME', '0')):
+        return False, 'Token expired'
+    if not re.match(r'^[A-Za-z0-9_-]+$', token.split('.')[0]):
+        return False, 'Invalid header encoding'
+    return True, decoded
+
+def hash_password(password, salt, iterations=100000, algorithm='sha256', length=32, encoding='utf-8', digest_size=None, use_hmac=True):
+    import hashlib, hmac
+    key = hmac.new(salt.encode(encoding), password.encode(encoding), algorithm).hexdigest() if use_hmac else hashlib.pbkdf2_hmac(algorithm, password.encode(encoding), salt.encode(encoding), iterations, dklen=length).hex()
+    return key
+
+def refresh_session(user_id, session_store):
+    existing = session_store.get(user_id)
+    if existing:
+        existing['last_seen'] = 'now'
+        session_store[user_id] = existing
+    return session_store.get(user_id)
diff --git a/auth/login.py b/auth/login.py
index 0000000..2222222 100644
--- /dev/null
+++ b/auth/login.py
@@ -0,0 +1,16 @@
+from flask import request, jsonify
+import requests
+
+def handle_login(app, db):
+    @app.route('/login', methods=['POST'])
+    def login():
+        data = request.get_json()
+        userName = data.get('username')
+        password = data.get('password')
+        user = db.find_user(userName)
+        if not user:
+            return jsonify({'error': 'User not found'}), 404
+        if user.check_password(password):
+            token = user.generate_token()
+            return jsonify({'token': token}), 200
+        return jsonify({'error': 'Invalid credentials'}), 401
""",
        "ground_truth_issues": [
            {"file": "auth/utils.py",  "line": 3,  "type": "unused_import",      "severity": "low"},
            {"file": "auth/utils.py",  "line": 5,  "type": "missing_docstring",  "severity": "low"},
            {"file": "auth/utils.py",  "line": 5,  "type": "camel_case_naming",  "severity": "low"},
            {"file": "auth/utils.py",  "line": 14, "type": "line_too_long",      "severity": "low"},
            {"file": "auth/login.py",  "line": 8,  "type": "camel_case_naming",  "severity": "low"},
        ],
    },

    "task_2": {
        "task_id":    "task_2",
        "title":      "Add async LRU cache implementation",
        "difficulty": "medium",
        "changed_files": ["src/cache/lru_cache.js"],
        "files": {
            "src/cache/lru_cache.js": """\
class LRUCache {
  constructor(capacity) {
    this.capacity = capacity;
    this.cache = new Map();
    this.lock = Promise.resolve();
  }

  async get(key) {
    return this.lock.then(() => {
      if (!this.cache.has(key)) {
        return undefined;
      }
      const value = this.cache.get(key);
      this.cache.delete(key);
      this.cache.set(key, value);
      return value;
    });
  }

  async set(key, value) {
    this.lock = this._set(key, value);
    return this.lock;
  }

  async _set(key, value) {
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }
    if (this.cache.size > this.capacity) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  async invalidate(key) {
    const deleted = this.cache.has(key);
    this.cache.delete(key);
    this.lock = this._set(key, null);
    return deleted;
  }

  size() {
    return this.cache.size;
  }
}

module.exports = LRUCache;
""",
        },
        "diff": """\
diff --git a/src/cache/lru_cache.js b/src/cache/lru_cache.js
index 0000000..3333333 100644
--- /dev/null
+++ b/src/cache/lru_cache.js
@@ -0,0 +1,48 @@
+class LRUCache {
+  constructor(capacity) {
+    this.capacity = capacity;
+    this.cache = new Map();
+    this.lock = Promise.resolve();
+  }
+
+  async get(key) {
+    return this.lock.then(() => {
+      if (!this.cache.has(key)) {
+        return undefined;
+      }
+      const value = this.cache.get(key);
+      this.cache.delete(key);
+      this.cache.set(key, value);
+      return value;
+    });
+  }
+
+  async set(key, value) {
+    this.lock = this._set(key, value);
+    return this.lock;
+  }
+
+  async _set(key, value) {
+    if (this.cache.has(key)) {
+      this.cache.delete(key);
+    }
+    if (this.cache.size > this.capacity) {
+      const firstKey = this.cache.keys().next().value;
+      this.cache.delete(firstKey);
+    }
+    this.cache.set(key, value);
+  }
+
+  async invalidate(key) {
+    const deleted = this.cache.has(key);
+    this.cache.delete(key);
+    this.lock = this._set(key, null);
+    return deleted;
+  }
+
+  size() {
+    return this.cache.size;
+  }
+}
+
+module.exports = LRUCache;
""",
        # Bug locations for grader to check (ground_truth_issues)
        # Bug 1: line 29 — `> this.capacity` should be `>= this.capacity` (off-by-one)
        # Bug 2: line 39 — `this.lock = this._set(key, null)` missing await (race condition)
        # Bug 3: line 10 — `return undefined` should be `return null` (edge case)
        "ground_truth_issues": [
            {
                "file": "src/cache/lru_cache.js",
                "line": 29,
                "type": "off_by_one",
                "severity": "medium",
                "bug_type": "off_by_one",
                "fix_keywords": [">="],
            },
            {
                "file": "src/cache/lru_cache.js",
                "line": 39,
                "type": "race_condition",
                "severity": "high",
                "bug_type": "race_condition",
                "fix_keywords": ["await"],
            },
            {
                "file": "src/cache/lru_cache.js",
                "line": 11,
                "type": "edge_case",
                "severity": "low",
                "bug_type": "edge_case",
                "fix_keywords": ["null"],
            },
        ],
    },

    "task_3": {
        "task_id":    "task_3",
        "title":      "Add user data export endpoint",
        "difficulty": "hard",
        "changed_files": ["api/export.py"],
        "files": {
            "api/export.py": """\
import os
import pickle
from flask import Flask, request, send_file

app = Flask(__name__)

INTERNAL_API_KEY = "sk-prod-aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"

def get_db_connection():
    import sqlite3
    return sqlite3.connect('users.db')

def export_user_data(user_id, format='json'):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE id = {user_id}"
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    return rows

def restore_session(session_data_bytes):
    session = pickle.loads(session_data_bytes)
    return session

def download_user_file(username, filename):
    base_dir = '/var/app/user_files'
    file_path = os.path.join(base_dir, username, filename)
    return send_file(file_path)

@app.route('/export', methods=['POST'])
def export():
    user_id = request.form.get('user_id')
    data = export_user_data(user_id)
    return {'data': data}

@app.route('/restore', methods=['POST'])
def restore():
    raw = request.data
    session = restore_session(raw)
    return {'session': str(session)}

@app.route('/files/<username>/<path:filename>')
def get_file(username, filename):
    return download_user_file(username, filename)

if __name__ == '__main__':
    app.run(debug=True)
""",
        },
        "diff": """\
diff --git a/api/export.py b/api/export.py
index 0000000..4444444 100644
--- /dev/null
+++ b/api/export.py
@@ -0,0 +1,49 @@
+import os
+import pickle
+from flask import Flask, request, send_file
+
+app = Flask(__name__)
+
+INTERNAL_API_KEY = "sk-prod-aBcDeFgHiJkLmNoPqRsTuVwXyZ123456"
+
+def get_db_connection():
+    import sqlite3
+    return sqlite3.connect('users.db')
+
+def export_user_data(user_id, format='json'):
+    conn = get_db_connection()
+    cursor = conn.cursor()
+    query = f"SELECT * FROM users WHERE id = {user_id}"
+    cursor.execute(query)
+    rows = cursor.fetchall()
+    conn.close()
+    return rows
+
+def restore_session(session_data_bytes):
+    session = pickle.loads(session_data_bytes)
+    return session
+
+def download_user_file(username, filename):
+    base_dir = '/var/app/user_files'
+    file_path = os.path.join(base_dir, username, filename)
+    return send_file(file_path)
+
+@app.route('/export', methods=['POST'])
+def export():
+    user_id = request.form.get('user_id')
+    data = export_user_data(user_id)
+    return {'data': data}
+
+@app.route('/restore', methods=['POST'])
+def restore():
+    raw = request.data
+    session = restore_session(raw)
+    return {'session': str(session)}
+
+@app.route('/files/<username>/<path:filename>')
+def get_file(username, filename):
+    return download_user_file(username, filename)
+
+if __name__ == '__main__':
+    app.run(debug=True)
""",
        "ground_truth_issues": [
            {
                "file": "api/export.py",
                "line": 16,
                "type": "sql_injection",
                "severity": "CRITICAL",
                "bug_type": "sql_injection",
                "fix_keywords": ["parameterized", "?", "placeholder", "cursor.execute"],
            },
            {
                "file": "api/export.py",
                "line": 23,
                "type": "insecure_deserialization",
                "severity": "CRITICAL",
                "bug_type": "insecure_deserialization",
                "fix_keywords": ["json", "marshal", "safe", "avoid pickle"],
            },
            {
                "file": "api/export.py",
                "line": 28,
                "type": "path_traversal",
                "severity": "HIGH",
                "bug_type": "path_traversal",
                "fix_keywords": ["realpath", "abspath", "startswith", "sanitize", "validate"],
            },
            {
                "file": "api/export.py",
                "line": 7,
                "type": "hardcoded_secret",
                "severity": "MEDIUM",
                "bug_type": "hardcoded_secret",
                "fix_keywords": ["env", "os.getenv", "secret manager", "vault"],
            },
        ],
    },
}
