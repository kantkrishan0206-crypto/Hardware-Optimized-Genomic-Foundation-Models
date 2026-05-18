from __future__ import annotations

import argparse
import json
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Call HOGFM inference API.")
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--api-key", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.dumps({"sequence": args.sequence, "max_length": 512}).encode()
    request = urllib.request.Request(
        args.url,
        data=payload,
        headers={"content-type": "application/json"},
        method="POST",
    )
    if args.api_key:
        request.add_header("x-api-key", args.api_key)
    with urllib.request.urlopen(request) as response:
        print(response.read().decode())


if __name__ == "__main__":
    main()
