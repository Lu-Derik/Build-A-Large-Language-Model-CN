import json
import os
import unicodedata
import urllib.request

import tiktoken


# DeepSeek-V4.1-Flash uses this publicly released tokenizer.
TOKENIZER_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/tokenizer.json"
)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "deepseek-tokenizer")
TOKENIZER_PATH = os.path.join(CACHE_DIR, "deepseek-v4.1-flash-tokenizer.json")


def get_tokenizer_data():
    """Load the tokenizer JSON, downloading it once into the local cache."""
    if not os.path.exists(TOKENIZER_PATH):
        os.makedirs(CACHE_DIR, exist_ok=True)
        print("Downloading DeepSeek tokenizer...")
        urllib.request.urlretrieve(TOKENIZER_URL, TOKENIZER_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def make_byte_decoder():
    """Build the byte-level BPE Unicode-to-byte mapping used by the tokenizer."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    codepoints = byte_values.copy()
    extra = 0

    for byte_value in range(256):
        if byte_value not in byte_values:
            byte_values.append(byte_value)
            codepoints.append(256 + extra)
            extra += 1

    return {chr(codepoint): byte_value for byte_value, codepoint in zip(byte_values, codepoints)}


def is_single_han_character(text):
    if len(text) != 1:
        return False

    name = unicodedata.name(text, "")
    return "CJK UNIFIED IDEOGRAPH" in name or "CJK COMPATIBILITY IDEOGRAPH" in name


def han_characters(text):
    """Return the Han characters found in text, preserving their order."""
    return [character for character in text if is_single_han_character(character)]


def add_token_to_groups(token_id, token_text, single_tokens, multi_tokens, pure_multi_tokens):
    characters = han_characters(token_text)
    if len(characters) == 1 and len(token_text) == 1:
        single_tokens.append((token_id, token_text))
    if len(characters) >= 2:
        multi_tokens.append((token_id, token_text))
        if len(characters) == len(token_text):
            pure_multi_tokens.append((token_id, token_text))


def count_deepseek_single_han_tokens():
    """Classify Han-character tokens in the DeepSeek-V4.1-Flash vocabulary."""
    data = get_tokenizer_data()
    vocab = data["model"]["vocab"]
    byte_decoder = make_byte_decoder()
    single_han_tokens = []
    multi_han_tokens = []
    pure_multi_han_tokens = []

    for token, token_id in vocab.items():
        try:
            token_bytes = bytes(byte_decoder[character] for character in token)
            token_text = token_bytes.decode("utf-8")
        except (KeyError, UnicodeDecodeError):
            # Some entries are special tokens or incomplete UTF-8 sequences.
            continue

        add_token_to_groups(
            token_id,
            token_text,
            single_han_tokens,
            multi_han_tokens,
            pure_multi_han_tokens,
        )

    # Added tokens are separate from byte-level BPE vocabulary entries.
    for added_token in data.get("added_tokens", []):
        token_text = added_token["content"]
        add_token_to_groups(
            added_token["id"],
            token_text,
            single_han_tokens,
            multi_han_tokens,
            pure_multi_han_tokens,
        )

    single_han_tokens.sort()
    multi_han_tokens.sort()
    pure_multi_han_tokens.sort()
    return len(vocab), single_han_tokens, multi_han_tokens, pure_multi_han_tokens


def count_tiktoken_single_han_tokens(encoding_name):
    """Classify Han-character tokens in a public OpenAI tiktoken encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    single_han_tokens = []
    multi_han_tokens = []
    pure_multi_han_tokens = []

    # The mergeable-ranks keys are the exact byte strings for ordinary BPE tokens.
    # Special tokens are structural markers and are not single Han characters.
    for token_bytes, token_id in encoding._mergeable_ranks.items():
        try:
            token_text = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue

        add_token_to_groups(
            token_id,
            token_text,
            single_han_tokens,
            multi_han_tokens,
            pure_multi_han_tokens,
        )

    single_han_tokens.sort()
    multi_han_tokens.sort()
    pure_multi_han_tokens.sort()
    return (
        encoding.n_vocab,
        single_han_tokens,
        multi_han_tokens,
        pure_multi_han_tokens,
    )


if __name__ == "__main__":
    print("Han-character token counts (same Unicode-based counting rule):")

    # GPT-2 is the original functionality; r50k_base is its equivalent encoding.
    tiktoken_encodings = [
        ("GPT-2 / r50k_base (GPT-3 legacy)", "gpt2"),
        ("p50k_base (legacy Codex)", "p50k_base"),
        ("cl100k_base (GPT-3.5, GPT-4)", "cl100k_base"),
        ("o200k_base (GPT-4o, GPT-4.1, GPT-5.x, o-series)", "o200k_base"),
    ]

    for label, encoding_name in tiktoken_encodings:
        vocab_size, singles, multi, pure_multi = count_tiktoken_single_han_tokens(
            encoding_name
        )
        print(f"{label}: {len(singles)} single-Han tokens / vocabulary size {vocab_size}")
        if encoding_name == "gpt2":
            print("  GPT-2 examples:", singles[:30])
        print("  前 50 个汉字:", [character for _, character in singles[:50]])
        print(f"  含两个或更多汉字的 token 数: {len(multi)}")
        print("  多汉字 token examples:", multi[:30])
        print(f"  纯汉字组成的多汉字 token 数: {len(pure_multi)}")
        print("  纯汉字 token examples:", pure_multi[:30])

    (
        deepseek_vocab_size,
        deepseek_tokens,
        deepseek_multi_tokens,
        deepseek_pure_multi_tokens,
    ) = count_deepseek_single_han_tokens()
    print(
        "DeepSeek-V4.1-Flash: "
        f"{len(deepseek_tokens)} / base vocabulary size {deepseek_vocab_size}"
    )
    print("  DeepSeek examples:", deepseek_tokens[:30])
    print(
        "  前 50 个汉字:",
        [character for _, character in deepseek_tokens[:50]],
    )
    print(
        "  含两个或更多汉字的 token 数:",
        len(deepseek_multi_tokens),
    )
    print("  DeepSeek multi-Han examples:", deepseek_multi_tokens[:30])
    print(
        "  纯汉字组成的多汉字 token 数:",
        len(deepseek_pure_multi_tokens),
    )
    print("  DeepSeek pure-Han examples:", deepseek_pure_multi_tokens[:30])
