def apply_permutation(module, X, permutation):
    return module(X.permute(permutation)).permute(permutation)
