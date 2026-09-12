from types import SimpleNamespace

from research.work_budget_hybrid import complete_layers


class Player:
    def __init__(self):
        self.stop_search = False
        self.seen = []

    def start_search(self, depth):
        self.seen.append(depth)
        self.node = 10**depth
        self.best_operation = 2
        self.scores = [1, 4, 2, 3]


def test_finishes_layers_and_discloses_soft_budget_overshoot():
    logic = SimpleNamespace(work_target=1500, work_calls=[])
    player = Player()
    action, depth, scores = complete_layers(logic, player, 24, 60, .0001)
    assert (action, depth) == (2, 4)
    assert player.seen == [3, 4]
    assert logic.work_calls[0]['summed_layer_nodes'] == 11000
    second = Player()
    assert complete_layers(logic, second, 5, 60, 10) == (action, depth, scores)
    assert second.seen == player.seen


def test_respects_small_maximum_depth_without_searching_extra_layers():
    logic = SimpleNamespace(work_target=1000000, work_calls=[])
    player = Player()
    assert complete_layers(logic, player, 3, 3, 1)[1] == 3
    assert player.seen == [3]
