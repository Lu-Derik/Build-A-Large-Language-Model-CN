"""Query a locally running Ollama chat model."""

import json
import urllib.error
import urllib.request


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat",
                print_response_details=True):
    payload = json.dumps({
        "model": model,
        "seed": 123,
        "temperature": 0,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response_parts = []
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue
                original_response = raw_line.decode("utf-8").rstrip("\r\n")
                if print_response_details:
                    print("\nOriginal response line:")
                    print(original_response)

                response_json = json.loads(original_response)
                if print_response_details:
                    print("\nParsed response_json:")
                    print(json.dumps(response_json, ensure_ascii=False, indent=2))

                response_parts.append(response_json.get("message", {}).get("content", ""))
                if print_response_details:
                    print("\nresponse_parts so far:")
                    print(repr(response_parts))

                if response_json.get("done"):
                    break
    except urllib.error.URLError as error:
        raise RuntimeError(
            "Could not reach Ollama at http://localhost:11434. "
            "Start the Ollama app/service and try again."
        ) from error

    return "".join(response_parts)


if __name__ == "__main__":
    model = "llama3"
    prompt = "What do llamas eat?"
    print(query_model(prompt, model))
