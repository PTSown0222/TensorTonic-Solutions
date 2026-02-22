import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    # A = (N,M)
    # convert to np.array
    A = np.array(A)
    # get shape of rows and cols with (N,M) => rows = N, cols = M
    rows, cols = A.shape
    # create a zeros matrix by rows and cols => AT = zeros(N,M )
    AT = np.zeros((cols, rows))
    
    #AT = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            AT[j, i] = A[i, j]
    return AT
