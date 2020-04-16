import numpy as np


def compute_feature_distances(features1, features2):
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.
    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second set
      features (m not necessarily equal to n)

    Returns:
    - dists: A numpy array of shape (n,m) which holds the distances from each
      feature in features1 to each feature in features2
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

    dists = np.zeros((features1.shape[0],features2.shape[0]))
    for i in range(features1.shape[0]):
        for j in range(features2.shape[0]):
            dists[i][j] = np.linalg.norm(features1[i]-features2[j])
    

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dists


def match_features(features1, features2, x1, y1, x2, y2):
    """
    This function does not need to be symmetric (e.g. it can produce
    different numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 4.18 in
    section 4.1.3 of Szeliski. There are a lot of repetitive features in
    these images, and all of their descriptors will look similar. The
    ratio test helps us resolve this issue (also see Figure 11 of David
    Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
    - features1: A numpy array of shape (n,feat_dim) representing one set of
      features, where feat_dim denotes the feature dimensionality
    - features2: A numpy array of shape (m,feat_dim) representing a second
      set of features (m not necessarily equal to n)
    - x1: A numpy array of shape (n,) containing the x-locations of features1
    - y1: A numpy array of shape (n,) containing the y-locations of features1
    - x2: A numpy array of shape (m,) containing the x-locations of features2
    - y2: A numpy array of shape (m,) containing the y-locations of features2

    Returns:
    - matches: A numpy array of shape (k,2), where k is the number of matches.
      The first column is an index in features1, and the second column is an
      index in features2
    - confidences: A numpy array of shape (k,) with the real valued confidence
      for every match

    'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
    """

    ###########################################################################
    # TODO: YOUR CODE HERE                                                    #
    ###########################################################################

#     dists = compute_feature_distances(features1, features2)
#     sorted_dists = sorted(dists, key=lambda a: a[0])
#     matches = np.array([[item[1], item[2]] for item in sorted_dists[:100]])
#     confidences = np.array(item[0] for item in sorted_dists[:100])
#     matches = matches.astype(int)
    matches = []
    conf = []
    dists = compute_feature_distances(features1, features2)
    
    for i in range(len(dists)):
        dist1 = np.copy(dists[i])
        index = np.argmin(dist1)
        mindist = dist1[index]
        dist1 = np.delete(dist1, index)
        mindist2 = dist1[np.argmin(dist1)]
        ratio = mindist /mindist2
        if ratio < 0.77:
                matches.append([i,index])
                conf.append(mindist)
    matches = np.array(matches)
    confidences = np.array(conf)
#         dists.append([ratio, i, dist_index])
#         dists = np.array(dists)
#         # < 0.76
#         for each in dists:
#             if each[0] < 0.76:
#                 matches.append(each)
#         matches = np.array(matches)
                
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return matches, confidences
