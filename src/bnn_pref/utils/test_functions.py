from bnn_pref.utils.type import Q1, Q2, Q2D, SD, D, Q


def linear_reward_fn(features_D: D, param_D: D) -> D:
    return features_D @ param_D


def polynomial_reward_fn(features_D, param_D):
    features_D = features_D**2
    return features_D @ param_D**2


test_functions_dict = {
    "linear": linear_reward_fn,
    "poly": polynomial_reward_fn,
}
