import numpy as np
def ordinal_encoding(values, ordering):
    """
    Encode categorical values using the provided ordering.
    """
    # Write code here
    vector = []
    ordering_mapping = {cate: ind for ind, cate in enumerate(ordering)}
    for val in values:
        index = ordering_mapping.get(val)
        vector.append(index)
    return vector
    