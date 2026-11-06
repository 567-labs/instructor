import requests


if __name__ == "__main__":
    with requests.post(
        "http://127.0.0.1:3000/extract",
        json={
            "query": "Alice and Bob are best friends. They are currently 32 and 43 respectively. "
        },
        stream=True,
        timeout=(5, 120),
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                print(str(chunk, encoding="utf-8"), end="\n")
