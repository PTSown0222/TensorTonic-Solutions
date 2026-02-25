def rank_transform(values):
    """
    Replace each value with its average rank.
    """
    rank_dict = {}
    for id, val in enumerate(sorted(values)):
        rank = id + 1
        if val not in rank_dict:
            rank_dict[val] = [rank]
        else:
            rank_dict[val].append(rank)
    
    avg_rank_dict = {}
    for val, ranks in rank_dict.items():
        avg_rank_dict[val] = sum(ranks) / len(ranks)
    
    return [avg_rank_dict[val] for val in values]
    