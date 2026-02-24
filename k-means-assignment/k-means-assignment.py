import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # c_arr = np.asarray(centroids)
    # p_arr = np.asarray(points)
    # D = c_arr.shape[1]
    # pos = []

    # for p in p_arr:
    #     min_distance = float("inf")
    #     closest_centroid_index = -1
        
    #     for id, c in enumerate(c_arr):
    #         squared_distance = sum((p[d] - c[d])**2 for d in range(D))
    
    #         if squared_distance < min_distance:
    #             min_distance = squared_distance
    #             closest_centroid_index = id
    #     pos.append(closest_centroid_index)
    # return pos

    # BROADCASTING
    p = np.asarray(points)     # Kích thước: (N, D) - N điểm, D chiều
    c = np.asarray(centroids)  # Kích thước: (K, D) - K tâm cụm, D chiều
    distances = np.sum((p[:, np.newaxis, :] - c[np.newaxis, :, :]) ** 2, axis=-1)
    res = np.argmin(distances, axis=-1)
    return list(res)
        
        