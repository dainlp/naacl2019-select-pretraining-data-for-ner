'''Reference url: https://github.com/tensorflow/nmt/blob/tf-1.4/nmt/utils/misc_utils.py
Update date: 2019-April-29'''
def format_bpe_text(symbols, delimiter=b"@@"):
    tokens = []
    token = b""

    if isinstance(symbols, str): symbols = symbols.encode()

    delimiter_len = len(delimiter)
    for symbol in symbols:
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            token += symbol[:-delimiter_len]
        else:
            token += symbol
            tokens.append(word)
            word = b""

    return b" ".join(tokens)