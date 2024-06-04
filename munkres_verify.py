from itertools import permutations
import numpy as np

cost_matrix_orig = np.empty((5,5))
perms = list(permutations([0,1,2,3,4]))
costs = [cost_matrix_orig[(0,1,2,3,4), perm].sum() for perm in perms]
lowest_indx = costs.index(min(costs))
lowest_cost = costs[lowest_indx]
lowest_perm = perms[lowest_indx]