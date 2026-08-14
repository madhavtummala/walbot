"""One-shot local Schwab consent, for machines that are not the deployment host.

Schwab only redirects to HTTPS, and the dashboard serves plain HTTP, so a loopback callback
has nowhere to land. This starts a throwaway HTTPS listener on the registered loopback URL,
completes the code exchange, and exits. The refresh token lands in the same state store the
dashboard reads, so the machine that runs this is the machine that ends up authorized.

    python -m src.brokerages.schwab_local_auth

The certificate is self-signed and generated fresh into a temp directory each run, so the
browser will warn once. That is expected: the connection is to your own loopback interface,
and nothing sensitive crosses it except the one-time authorization code.
"""

from __future__ import annotations

import argparse
import datetime as dt
import http.server
import logging
import ssl
import sys
import tempfile
import threading
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from ..core.config import get_config
from .schwab_auth import begin_authorization, complete_authorization
from .schwab_client import SchwabAuthError

logger = logging.getLogger(__name__)

DEFAULT_CALLBACK = "https://127.0.0.1:8182"


def _self_signed_cert(directory: Path) -> tuple[Path, Path]:
    """Generate a short-lived certificate for 127.0.0.1."""
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")])
    now = dt.datetime.now(dt.timezone.utc)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.IPAddress(__import__("ipaddress").ip_address("127.0.0.1"))]), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_path = directory / "cert.pem"
    key_path = directory / "key.pem"
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    return cert_path, key_path


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    result: dict[str, Any] = {}
    done = threading.Event()

    def do_GET(self) -> None:  # noqa: N802 - http.server's required name
        query = parse_qs(urlparse(self.path).query)
        code = (query.get("code") or [""])[0]
        state = (query.get("state") or [""])[0]
        error = (query.get("error") or [""])[0]

        if error:
            _CallbackHandler.result = {"ok": False, "message": f"Schwab denied the request: {error}"}
        elif not code:
            # Browsers ask for /favicon.ico; ignore anything that is not the real redirect.
            self.send_response(204)
            self.end_headers()
            return
        else:
            try:
                complete_authorization(get_config(), code=code, returned_state=state)
                _CallbackHandler.result = {"ok": True, "message": "Schwab connected. You can close this tab."}
            except Exception as exc:  # noqa: BLE001 - report it on the page and in the shell
                _CallbackHandler.result = {"ok": False, "message": str(exc)}

        ok = _CallbackHandler.result.get("ok", False)
        body = f"<h2>{'Connected' if ok else 'Authorization failed'}</h2><p>{_CallbackHandler.result['message']}</p>"
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(body.encode("utf-8"))
        _CallbackHandler.done.set()

    def log_message(self, *args: Any) -> None:
        """Silence the default stderr access log."""


def run(callback_url: str = DEFAULT_CALLBACK, open_browser: bool = True, timeout: float = 300.0) -> int:
    config = get_config()
    configured = str(getattr(config, "schwab_callback_url", "") or "")
    if configured != callback_url:
        print(f"SCHWAB_CALLBACK_URL is {configured or '(unset)'}, but this helper serves {callback_url}.")
        print("Schwab requires the redirect_uri to match exactly, so set it and re-run:")
        print(f"  SCHWAB_CALLBACK_URL={callback_url}")
        return 2

    parsed = urlparse(callback_url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 443

    try:
        started = begin_authorization(config)
    except SchwabAuthError as error:
        print(f"Cannot start authorization: {error}")
        return 2

    with tempfile.TemporaryDirectory() as directory:
        cert_path, key_path = _self_signed_cert(Path(directory))
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))

        server = http.server.HTTPServer((host, port), _CallbackHandler)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        print(f"Listening on {callback_url}")
        print("\nOpen this URL, log in, and approve:\n")
        print(started["authorize_url"])
        print("\nYour browser will warn about the self-signed certificate on the redirect.")
        print("That is this script's own listener -- proceed past it.\n")
        if open_browser:
            webbrowser.open(started["authorize_url"])

        finished = _CallbackHandler.done.wait(timeout=timeout)
        server.shutdown()

    if not finished:
        print(f"Timed out after {int(timeout)}s without a callback.")
        return 1
    result = _CallbackHandler.result
    print(result.get("message", ""))
    return 0 if result.get("ok") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--callback-url", default=DEFAULT_CALLBACK, help="Registered loopback callback URL")
    parser.add_argument("--no-browser", action="store_true", help="Print the URL instead of opening it")
    parser.add_argument("--timeout", type=float, default=300.0, help="Seconds to wait for the redirect")
    args = parser.parse_args()
    return run(args.callback_url, open_browser=not args.no_browser, timeout=args.timeout)


if __name__ == "__main__":
    sys.exit(main())
