def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    chunks = []
    step = chunk_size - overlap
    # tokens = ["a", "b", "c", "d", "e", "f", "g"], chunk_size = 3, overlap = 1
    # (start, stop, step) -> (0,7,2)
    # (a, b, c), (d, e,f)...
    for token in range(0, len(tokens), step):
        chunks.append(tokens[token: token + chunk_size])
        if token + chunk_size >= len(tokens):
            break
    return chunks
        