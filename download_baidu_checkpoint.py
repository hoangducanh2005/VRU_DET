"""Download the CSP checkpoint from the original Baidu share.

Baidu allows the public share to be inspected anonymously, but its direct
download endpoint requires a logged-in Baidu session. Keep BDUSS local: pass
it through the environment or --bduss, never commit it or paste it into chat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests


SHARE_URL = "https://pan.baidu.com/s/1SSPQnbDP6zf9xf8eCDi3Fw"
SHARE_PASSWORD = "jcgd"
REMOTE_PATH = "/research/models_CSP/cityperson/withoffset/net_e121_l0.hdf5"
EXPECTED_SIZE = 160_609_096
EXPECTED_MD5 = "464fadd6a04527d00be91fe7769bdde1"
USER_AGENT = "Mozilla/5.0 CSP-checkpoint-downloader/1.0"


class BaiduDownloadError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download net_e121_l0.hdf5 from the original Baidu share."
    )
    parser.add_argument("--output", type=Path, default=Path("net_e121_l0.hdf5"))
    parser.add_argument(
        "--bduss",
        default=os.environ.get("BAIDU_BDUSS"),
        help="Baidu BDUSS cookie; prefer BAIDU_BDUSS over shell history.",
    )
    parser.add_argument("--share-url", default=SHARE_URL)
    parser.add_argument("--password", default=SHARE_PASSWORD)
    parser.add_argument("--force", action="store_true", help="Replace a valid output file.")
    return parser.parse_args()


def response_json(response: requests.Response, operation: str) -> dict:
    try:
        payload = response.json()
    except ValueError as exc:
        raise BaiduDownloadError(
            f"Baidu returned non-JSON data during {operation}: HTTP {response.status_code}"
        ) from exc
    errno = int(payload.get("errno", 0))
    if errno != 0:
        message = payload.get("errmsg") or payload.get("show_msg") or "unknown error"
        raise BaiduDownloadError(f"Baidu {operation} failed: errno={errno}, {message}")
    return payload


def discover_share(session: requests.Session, share_url: str) -> tuple[str, str, str]:
    response = session.get(share_url, allow_redirects=True, timeout=30)
    response.raise_for_status()
    final_url = response.url
    match = re.search(r"/share/init\?surl=([A-Za-z0-9_-]+)", final_url)
    if match:
        surl = match.group(1)
    else:
        match = re.search(r"/s/1([A-Za-z0-9_-]+)", urlparse(share_url).path)
        if not match:
            raise BaiduDownloadError(f"Could not extract Baidu surl from {share_url}")
        surl = match.group(1)

    html = response.text
    share_uk = re.search(r'share_uk["\s:]+["\']?(\d+)', html)
    share_id = re.search(r'shareid["\s:]+["\']?(\d+)', html)
    if not share_uk or not share_id:
        raise BaiduDownloadError("Could not discover shareid/uk from the Baidu share page")
    return surl, share_id.group(1), share_uk.group(1)


def verify_share(
    session: requests.Session, share_id: str, share_uk: str, password: str
) -> str:
    response = session.post(
        "https://pan.baidu.com/share/verify",
        params={"shareid": share_id, "uk": share_uk},
        data={"pwd": password, "vcode": "", "vcode_str": ""},
        timeout=30,
    )
    payload = response_json(response, "password verification")
    return unquote(payload["randsk"])


def get_remote_file(
    session: requests.Session, share_id: str, share_uk: str, sekey: str
) -> dict:
    parent = str(Path(REMOTE_PATH).parent).replace("\\", "/")
    response = session.get(
        "https://pan.baidu.com/share/list",
        params={
            "shareid": share_id,
            "uk": share_uk,
            "dir": parent,
            "page": "1",
            "num": "100",
            "root": "0",
            "sekey": sekey,
        },
        timeout=30,
    )
    payload = response_json(response, "file listing")
    filename = Path(REMOTE_PATH).name
    for item in payload.get("list", []):
        if item.get("server_filename") == filename or item.get("path") == REMOTE_PATH:
            return item
    raise BaiduDownloadError(f"Checkpoint not found in Baidu directory {parent}")


def get_direct_link(
    session: requests.Session, share_id: str, share_uk: str, fs_id: str, sekey: str
) -> str:
    sign_response = session.get(
        "https://pan.baidu.com/share/tplconfig",
        params={
            "shareid": share_id,
            "uk": share_uk,
            "fields": "sign,timestamp",
            "channel": "chunlei",
            "web": "1",
            "app_id": "250528",
            "clienttype": "0",
        },
        timeout=30,
    )
    sign_data = response_json(sign_response, "download-sign request")["data"]
    response = session.post(
        "https://pan.baidu.com/api/sharedownload",
        params={
            "app_id": "250528",
            "channel": "chunlei",
            "clienttype": "12",
            "web": "1",
            "sign": sign_data["sign"],
            "timestamp": sign_data["timestamp"],
        },
        data={
            "encrypt": "0",
            "extra": json.dumps({"sekey": sekey}, separators=(",", ":")),
            "fid_list": f"[{fs_id}]",
            "primaryid": share_id,
            "product": "share",
            "type": "nolimit",
            "uk": share_uk,
        },
        timeout=30,
    )
    payload = response_json(response, "direct-link request")
    try:
        return payload["list"][0]["dlink"]
    except (KeyError, IndexError) as exc:
        raise BaiduDownloadError("Baidu did not return a direct download link") from exc


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(session: requests.Session, dlink: str, output: Path, size: int) -> None:
    part = output.with_suffix(output.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"Referer": SHARE_URL}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    with session.get(dlink, headers=headers, stream=True, timeout=(30, 120)) as response:
        if existing and response.status_code == 200:
            existing = 0
        response.raise_for_status()
        mode = "ab" if existing and response.status_code == 206 else "wb"
        written = existing if mode == "ab" else 0
        with part.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
                print(f"\rDownloaded {written:,}/{size:,} bytes", end="", flush=True)
    print()
    part.replace(output)


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists() and not args.force:
        if output.stat().st_size == EXPECTED_SIZE and md5_file(output) == EXPECTED_MD5:
            print(f"Checkpoint already verified: {output}")
            return 0
        raise BaiduDownloadError(
            f"Output exists but is not the expected checkpoint: {output}. Use --force."
        )
    if not args.bduss:
        raise BaiduDownloadError(
            "Baidu blocks anonymous direct downloads (errno=9019). Set BAIDU_BDUSS "
            "in this PowerShell session or pass --bduss; do not put it in chat or git."
        )

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.cookies.set("BDUSS", args.bduss, domain=".baidu.com", path="/")
    _surl, share_id, share_uk = discover_share(session, args.share_url)
    session.get(args.share_url, timeout=30)
    sekey = verify_share(session, share_id, share_uk, args.password)
    metadata = get_remote_file(session, share_id, share_uk, sekey)
    print(
        f"Found {metadata.get('path')} size={metadata.get('size')} "
        f"md5={metadata.get('md5')}"
    )
    if int(metadata.get("size", 0)) != EXPECTED_SIZE or metadata.get("md5") != EXPECTED_MD5:
        raise BaiduDownloadError("Remote metadata does not match the expected checkpoint")
    dlink = get_direct_link(session, share_id, share_uk, str(metadata["fs_id"]), sekey)
    output.parent.mkdir(parents=True, exist_ok=True)
    download_file(session, dlink, output, EXPECTED_SIZE)
    actual_size = output.stat().st_size
    actual_md5 = md5_file(output)
    if actual_size != EXPECTED_SIZE or actual_md5 != EXPECTED_MD5:
        raise BaiduDownloadError(
            f"Checksum mismatch: size={actual_size}, md5={actual_md5}; file rejected"
        )
    print(f"Verified checkpoint: {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BaiduDownloadError, requests.RequestException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
