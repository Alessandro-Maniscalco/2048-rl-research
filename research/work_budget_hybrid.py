"""Repeatable search allocation for the existing hybrid player.

Replace only CoreAILogic's wall-clock iterative-deepening controller on this
instance. Complete layers from depth three; stop when a layer's native node
count reaches the work target. This is a soft target, not a hard node cap:
the final layer can overshoot. Fixed-depth embedded-table validation remains.
No native binary, heuristic, solved table, or game rule is changed.
"""
from types import MethodType

from rl2048.agents.hybrid_endgame import HybridEndgameAgent


def complete_layers(logic, player, initial_depth, max_depth, time_limit):
    # Ignore the upstream proposed initial depth and wall-clock allowance. Both
    # can reflect timing history. Begin with a cheap completed fallback instead.
    upper = max(1, int(max_depth))
    first = min(3, upper)
    layers = []
    total = 0
    best = 0
    scores = []
    final_depth = 0
    for depth in range(first, upper + 1):
        player.stop_search = False
        player.start_search(depth)
        if player.stop_search:
            raise RuntimeError('A deterministic search layer was interrupted')
        best = int(player.best_operation)
        scores = list(player.scores)
        final_depth = depth
        nodes = int(player.node)
        total += nodes
        layers.append(dict(depth=depth, nodes=nodes, action=best, scores=scores))
        if best == 0 or nodes >= logic.work_target:
            break
    logic.work_calls.append(dict(requested_initial_depth=int(initial_depth),
        requested_max_depth=int(max_depth), ignored_seconds=float(time_limit),
        soft_layer_node_target=logic.work_target, summed_layer_nodes=total, layers=layers))
    return best, final_depth, scores


class WorkBudgetHybridAgent(HybridEndgameAgent):
    name = 'hybrid_complete_layer_work_budget'
    display_name = 'Frozen formations + repeatable work-budget search'

    def __init__(self, work_target=1000000):
        if int(work_target) != work_target or work_target < 1:
            raise ValueError('Positive integer work target required')
        super().__init__(1.)
        self.work_target = int(work_target)
        self.logic.work_target = self.work_target
        self.logic.work_calls = []
        self.logic.perform_iterative_search = MethodType(complete_layers, self.logic)

    def act(self, board, action_mask):
        self.logic.work_calls = []
        action = super().act(board, action_mask)
        self.last_decision.update(work_target=self.work_target,
            work_calls=self.logic.work_calls,
            summed_iterative_layer_nodes=sum(c['summed_layer_nodes'] for c in self.logic.work_calls))
        return action
