# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
               first='MyCustomAgent', second='MyCustomAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers. isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class MyCustomAgent(CaptureAgent):
    """
    A custom Capture Agent that can be tailored for offensive or defensive strategies.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.walls = None
        self.width = 0
        self.height = 0

    def register_initial_state(self, game_state):
        """
        Initializes the agent's starting position and game map information.
        """
        self.start = game_state.get_agent_position(self.index)
        self.walls = game_state.get_walls()
        self.width = game_state.data.layout.width
        self.height = game_state.data.layout.height
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Chooses the best action based on evaluation of possible moves.
        """
        actions = game_state.get_legal_actions(self.index)
        # Remove STOP action to encourage movement
        actions = [action for action in actions if action != Directions.STOP]

        # Evaluate each action
        values = [self.evaluate(game_state, action) for action in actions]

        # Find the maximum evaluation value
        max_value = max(values)

        # Choose actions that have the maximum evaluation value
        best_actions = [action for action, value in zip(actions, values) if value == max_value]

        # Select randomly among the best actions to introduce variability
        chosen_action = random.choice(best_actions)

        # Debugging: Print chosen action
        # print(f"Agent {self.index} chooses action {chosen_action}")

        return chosen_action

    def evaluate(self, game_state, action):
        """
        Evaluates the desirability of a given action.
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Extracts features from the game state after taking an action.
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        # Example feature: distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        # Example feature: number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Example feature: whether the action stops Pacman from moving
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0

        # Example feature: whether the action reverses the current direction
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        if action == Directions.REVERSE.get(current_direction, None):
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def get_weights(self, game_state, action):
        """
        Assigns weights to each feature for evaluation.
        """
        return {
            'distance_to_food': -1.5,
            'num_invaders': -1000,
            'stop': -100,
            'reverse': -2
        }

    def get_successor(self, game_state, action):
        """
        Generates the successor game state after taking an action.
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()

        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    # Optional: Add additional methods for enhanced behavior

    def get_opponents(self, game_state):
        """
        Returns a list of opponent agent indices.
        """
        return self.get_opponents_indices()

    def get_opponents_indices(self):
        """
        Retrieves the indices of opponent agents.
        """
        num_agents = game_state.get_num_agents()
        opponents = []
        for i in range(num_agents):
            if i != self.index and i not in self.get_team_indices():
                opponents.append(i)
        return opponents

    def get_team_indices(self):
        """
        Retrieves the indices of agents on the same team.
        """
        team_indices = [self.index]
        # Assuming you have a method to get team members; replace with actual implementation
        for agent in self.get_team(game_state=None):
            if agent != self.index:
                team_indices.append(agent)
        return team_indices

    # Add more helper methods as needed for complex strategies
