from __future__ import annotations
import collections
import random
import csv
from math import fabs

import numpy as np
import sklearn.preprocessing as skl_preprocessing

from problem import Action, available_actions, Corner, Driver, Experiment, Environment, State

ALMOST_INFINITE_STEP = 10000
MAX_LEARNING_STEPS = 500


class RandomDriver(Driver):
    def __init__(self):
        self.current_step: int = 0

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        return random.choice(available_actions(state))

    def control(self, state: State, last_reward: int) -> Action:
        self.current_step += 1
        return random.choice(available_actions(state))

    def finished_learning(self) -> bool:
        return self.current_step > MAX_LEARNING_STEPS


class OffPolicyNStepSarsaDriver(Driver):
    def __init__(self, step_size: float, step_no: int, experiment_rate: float, discount_factor: float) -> None:
        self.step_size: float = step_size
        self.step_no: int = step_no
        self.experiment_rate: float = experiment_rate
        self.discount_factor: float = discount_factor
        self.q: dict[tuple[State, Action], float] = collections.defaultdict(float)
        self.current_step: int = 0
        self.final_step: int = ALMOST_INFINITE_STEP
        self.finished: bool = False
        self.states: dict[int, State] = dict()
        self.actions: dict[int, Action] = dict()
        self.rewards: dict[int, int] = dict()

    def start_attempt(self, state: State) -> Action:
        self.current_step = 0
        self.states[self._access_index(self.current_step)] = state
        action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
        self.actions[self._access_index(self.current_step)] = action
        self.final_step = ALMOST_INFINITE_STEP
        self.finished = False
        return action

    def control(self, state: State, last_reward: int) -> Action:
        if self.current_step < self.final_step:
            self.rewards[self._access_index(self.current_step + 1)] = last_reward
            self.states[self._access_index(self.current_step + 1)] = state
            if self.final_step == ALMOST_INFINITE_STEP and (
                    last_reward == 0 or self.current_step == MAX_LEARNING_STEPS
            ):
                self.final_step = self.current_step
            action = self._select_action(self.epsilon_greedy_policy(state, available_actions(state)))
            self.actions[self._access_index(self.current_step + 1)] = action
        else:
            action = Action(0, 0)

        update_step = self.current_step - self.step_no + 1
        if update_step >= 0:
            return_value_weight = self._return_value_weight(update_step)
            return_value = self._return_value(update_step)
            state_t1 = self.states[self._access_index(update_step)]
            action_t1 = self.actions[self._access_index(update_step)]
            qt = self.q[state_t1, action_t1]
            if update_step + self.step_no < self.final_step:
                state_t2 = self.states[self._access_index(update_step + self.step_no)]
                action_t2 = self.actions[self._access_index(update_step + self.step_no)]
                return_value += (self.discount_factor ** self.step_no) * self.q[state_t2, action_t2]
            self.q[state_t1, action_t1] = qt + self.step_size * return_value_weight * (return_value - qt)
        if update_step == self.final_step - 1:
            self.finished = True

        self.current_step += 1
        return action

    def _return_value(self, update_step):
        # TODO: Tutaj trzeba policzy?? zwrot G
        G = 0.0
        for i in range((update_step + 1), min(update_step + self.step_no, self.final_step)):
            G += (self.discount_factor ** (i - update_step - 1)) * self.rewards[self._access_index(i)]
        return G

    def _return_value_weight(self, update_step):
        # TODO: Tutaj trzeba policzy?? korekt?? na r????ne prawdopodobie??stwa ?? (poniewa?? uczymy poza-polityk??)
        p = 1.0
        for i in range((update_step + 1), min(update_step + self.step_no, self.final_step)):
            state = self.states[self._access_index(i)]
            action = self.actions[self._access_index(i)]
            actions = available_actions(state)
            eps_greedy = self.epsilon_greedy_policy(state, actions)[action]
            greedy = self.greedy_policy(state, actions)[action]
            p *= greedy / eps_greedy
        return p

    def finished_learning(self) -> bool:
        return self.finished

    def _access_index(self, index: int) -> int:
        return index % (self.step_no + 1)

    @staticmethod
    def _select_action(actions_distribution: dict[Action, float]) -> Action:
        actions = list(actions_distribution.keys())
        probabilities = list(actions_distribution.values())
        i = np.random.choice(list(range(len(actions))), p=probabilities)
        return actions[i]

    def epsilon_greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        # TODO: tutaj trzeba ustalic prawdopodobie??stwa wyboru akcji wed??ug polityki ??-zach??annej
        probabilities = self.experiment_rate * self._random_probabilities(actions) + (1 - self.experiment_rate) \
                            * self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def greedy_policy(self, state: State, actions: list[Action]) -> dict[Action, float]:
        probabilities = self._greedy_probabilities(state, actions)
        return {action: probability for action, probability in zip(actions, probabilities)}

    def _greedy_probabilities(self, state: State, actions: list[Action]) -> np.ndarray:
        values = [self.q[state, action] for action in actions]
        maximal_spots = (values == np.max(values)).astype(float)
        return self._normalise(maximal_spots)

    @staticmethod
    def _random_probabilities(actions: list[Action]) -> np.ndarray:
        maximal_spots = np.array([1.0 for _ in actions])
        return OffPolicyNStepSarsaDriver._normalise(maximal_spots)

    @staticmethod
    def _normalise(probabilities: np.ndarray) -> np.ndarray:
        return skl_preprocessing.normalize(probabilities.reshape(1, -1), norm='l1')[0]


def main() -> float:
    experiment = Experiment(
        environment=Environment(
            corner=Corner(
                name='corner_d'
            ),
            steering_fail_chance=0.01,
        ),
        driver=OffPolicyNStepSarsaDriver(
            step_no=4,
            step_size=0.6,
            experiment_rate=0.05,
            discount_factor=1.00
        ),
        number_of_episodes=10000,
    )

    return experiment.run()


if __name__ == '__main__':
    main()
