#!/usr/bin/env python3
"""Simple local web frontend for participant response prediction."""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from chose_expert import predict_response


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Participant Expert</title>
  <style>
    :root { color-scheme: light; font-family: system-ui, sans-serif; }
    body { background: #f4f7fb; color: #172033; margin: 0; }
    main { max-width: 760px; margin: 7vh auto; padding: 0 20px; }
    section { background: white; border-radius: 14px; box-shadow: 0 8px 30px #17203318; padding: 28px; }
    h1 { margin-top: 0; }
    label { display: block; font-weight: 600; margin: 18px 0 8px; }
    textarea { box-sizing: border-box; min-height: 130px; resize: vertical; width: 100%; }
    textarea, input, button { border: 1px solid #c8d1df; border-radius: 8px; font: inherit; padding: 11px 12px; }
    button { background: #2457d6; border: 0; color: white; cursor: pointer; margin-top: 18px; }
    button:disabled { cursor: wait; opacity: .65; }
    #status { color: #52617a; margin-top: 16px; }
    #error { color: #b42318; white-space: pre-wrap; }
    #result { border-left: 4px solid #2457d6; display: none; margin-top: 20px; padding: 4px 16px; }
    .expert { font-weight: 700; }
    pre { font-family: inherit; white-space: pre-wrap; }
  </style>
</head>
<body>
<main>
  <section>
    <h1>Participant response predictor</h1>
    <p>Ask a question and the system will select the most relevant expert.</p>
    <form id="question-form">
      <label for="question">Question</label>
      <textarea id="question" required
        placeholder="Example: Would the participant support a government welfare scheme?"></textarea>
      <button id="submit" type="submit">Ask participant</button>
    </form>
    <div id="status"></div>
    <div id="error"></div>
    <div id="result">
      <p>Expert: <span class="expert" id="expert"></span></p>
      <p>Reason: <span id="reason"></span></p>
      <h2>Predicted response</h2>
      <pre id="prediction"></pre>
    </div>
  </section>
</main>
<script>
const form = document.getElementById("question-form");
const button = document.getElementById("submit");
const status = document.getElementById("status");
const error = document.getElementById("error");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  result.style.display = "none";
  error.textContent = "";
  status.textContent = "Generating response...";
  button.disabled = true;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question: document.getElementById("question").value})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Request failed.");
    document.getElementById("expert").textContent = data.selected_expert;
    document.getElementById("reason").textContent = data.classification_reason;
    document.getElementById("prediction").textContent = data.prediction;
    result.style.display = "block";
    status.textContent = "";
  } catch (requestError) {
    status.textContent = "";
    error.textContent = requestError.message;
  } finally {
    button.disabled = false;
  }
});
</script>
</body>
</html>
"""


class FrontendHandler(BaseHTTPRequestHandler):
    """Serve the page and the prediction endpoint."""

    def _send_json(self, payload: dict, status: HTTPStatus) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if urlparse(self.path).path != "/":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = HTML_PAGE.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/predict":
            self._send_json({"error": "Endpoint not found."}, HTTPStatus.NOT_FOUND)
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length > 100_000:
                raise ValueError("Request is too large.")
            payload = json.loads(self.rfile.read(content_length))
            question = payload.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError("Please enter a question.")
            result = predict_response(question)
            self._send_json(result, HTTPStatus.OK)
        except (ValueError, FileNotFoundError, EnvironmentError) as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json(
                {"error": f"Prediction failed: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: object) -> None:
        print(f"[Frontend] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(f"Open http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping frontend.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
