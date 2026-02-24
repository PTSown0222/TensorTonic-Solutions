import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    c_arr = np.asarray(centroids)
    p_arr = np.asarray(points)
    D = c_arr.shape[1]
    pos = []

    for p in p_arr:
        min_distance = float("inf")
        closest_centroid_index = -1
        
        for id, c in enumerate(c_arr):
            squared_distance = sum((p[d] - c[d])**2 for d in range(D))
    
            if squared_distance < min_distance:
                min_distance = squared_distance
                closest_centroid_index = id
        pos.append(closest_centroid_index)
    return pos
        
        