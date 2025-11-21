from typing import Union

from bnn_pref.alg.agent_dropout import DropoutAgent, DropoutBeliefState
from bnn_pref.alg.agent_ekf import EKFAgent, EKFBeliefState
from bnn_pref.alg.agent_ensemble import EnsembleAgent, EnsembleBeliefState
from bnn_pref.alg.agent_laplace import LaplaceAgent, LaplaceBeliefState
from bnn_pref.alg.agent_llmcmc import LMCMCAgent, LMCMCBeliefState

AgentState = Union[
    EKFBeliefState,
    EnsembleBeliefState,
    DropoutBeliefState,
    LMCMCBeliefState,
    LaplaceBeliefState,
]
alg_classes = {
    "dropout": DropoutAgent,
    "ekf": EKFAgent,
    "ensemble": EnsembleAgent,
    "laplace": LaplaceAgent,
    "llmcmc": LMCMCAgent,
}
