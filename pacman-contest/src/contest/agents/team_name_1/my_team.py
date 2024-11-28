import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from queue import PriorityQueue
import numpy as np

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as sus números de índice de agente.
    """
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
       # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
       # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

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

        # Feature: distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if food_list:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        # Feature: number of invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        # Feature: whether the action stops Pacman from moving
        features['stop'] = 1 if action == Directions.STOP else 0

        # Feature: whether the action reverses the current direction
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        successor_direction = successor.get_agent_state(self.index).configuration.direction
        features['reverse'] = 1 if successor_direction == Directions.REVERSE[current_direction] else 0

        # Feature: distance to the nearest capsule
        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_distance = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_capsule'] = min_capsule_distance
        else:
            features['distance_to_capsule'] = 0

        return features

    def get_weights(self, game_state, action):
        """
        Returns the weights for each feature.
        """
        return {
            'distance_to_food': -1.0,
            'num_invaders': -1000.0,
            'stop': -100.0,
            'reverse': -2.0,
            'distance_to_capsule': -1.0
        }




class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that seeks food. This is an agent
    we give you to get an idea of what an offensive agent might look like,
    but it is by no means the best or only way to build an offensive agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)

        # Posición actual del agente
        my_pos = successor.get_agent_state(self.index).get_position()

        # Verificar si la posición actual es válida
        if not self.is_position_valid(my_pos, game_state):
            return features

        # Distancia a la comida más cercana
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Distancia a la cápsula más cercana
        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_distance = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_capsule'] = min_capsule_distance
        else:
            features['distance_to_capsule'] = 0

        # Distancia al fantasma más cercano
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if ghosts:
            min_ghost_distance = min(self.get_maze_distance(my_pos, ghost.get_position()) for ghost in ghosts)
            features['distance_to_ghost'] = min_ghost_distance
        else:
            features['distance_to_ghost'] = 0

        # Distancia al punto más cercano de su lado del campo (retorno a casa)
        if len(food_list) <= 2:  # Si tiene 2 comidas o menos
            home_boundary = self.get_home_boundary(successor)
            valid_home_positions = [pos for pos in home_boundary if self.is_position_valid(pos, game_state)]
            if valid_home_positions:
                min_home_distance = min(self.get_maze_distance(my_pos, home) for home in valid_home_positions)
                features['distance_to_home'] = min_home_distance
            else:
                features['distance_to_home'] = 0
        else:
            features['distance_to_home'] = 0

        # Verificar si debe regresar a casa (obligatorio)
        if len(food_list) <= 2:
            nearby_food = [food for food in food_list if self.get_maze_distance(my_pos, food) <= 5]
            if not nearby_food:  # No hay comida cerca
                features['return_home'] = 1
            else:
                features['return_home'] = 0
        else:
            features['return_home'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'successor_score': 100,
            'distance_to_food': -1,
            'distance_to_capsule': -1,
            'distance_to_ghost': 10,
            'distance_to_home': -2,
            'return_home': 2000  # Alta prioridad para retornar a casa
        }

    def get_home_boundary(self, game_state):
        """
        Returns the positions on the home boundary.
        """
        mid_x = game_state.data.layout.width // 2
        if self.red:
            mid_x -= 1
        else:
            mid_x += 1
        boundary_positions = [(mid_x, y) for y in range(game_state.data.layout.height)]
        return boundary_positions

    def is_position_valid(self, pos, game_state):
        """
        Verifica si una posición está dentro de la cuadrícula del juego.
        """
        x, y = pos
        return 0 <= x < game_state.data.layout.width and 0 <= y < game_state.data.layout.height






class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A defensive agent with improved strategies including Manhattan distance,
    priority queue-based decision making, and enemy movement prediction.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # On defense or offense
        features['on_defense'] = 1 if not my_state.is_pacman else 0

        # Distance to visible invaders
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if invaders:
            min_invader_dist = min(self.get_maze_distance(my_pos, a.get_position()) for a in invaders)
            features['invader_distance'] = min_invader_dist

        # Predict next positions of invaders based on directions
        if invaders:
            predicted_positions = [
                self.predict_invader_next_position(inv, game_state) for inv in invaders
            ]
            predicted_dist = min(self.get_manhattan_distance(my_pos, pos) for pos in predicted_positions)
            features['predicted_invader_distance'] = predicted_dist
        else:
            features['predicted_invader_distance'] = 0

        # Protect capsules
        capsules = self.get_capsules_you_are_defending(successor)
        if capsules:
            min_capsule_dist = min(self.get_manhattan_distance(my_pos, cap) for cap in capsules)
            features['distance_to_capsule'] = min_capsule_dist
        else:
            features['distance_to_capsule'] = 0

        # Distance to food being eaten
        food_list = self.get_food_you_are_defending(successor).as_list()
        if food_list:
            min_food_dist = min(self.get_manhattan_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_food_dist
        else:
            features['distance_to_food'] = 0

        # Penalize stopping and reversing
        if action == Directions.STOP:
            features['stop'] = 1
        else:
            features['stop'] = 0
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1
        else:
            features['reverse'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'predicted_invader_distance': -5,
            'distance_to_capsule': -5,
            'distance_to_food': -1,
            'stop': -100,
            'reverse': -2
        }

    def predict_invader_next_position(self, invader, game_state):
        """
        Predicts the next position of an invader based on its current direction.
        """
        pos = invader.get_position()
        direction = invader.configuration.direction
        dx, dy = Directions.VECTOR[direction]
        predicted_pos = (pos[0] + dx, pos[1] + dy)
        if self.is_position_valid(predicted_pos, game_state):
            return predicted_pos
        return pos

    def is_position_valid(self, pos, game_state):
        """
        Verifies if a position is within the valid game grid.
        """
        x, y = pos
        return 0 <= x < game_state.data.layout.width and 0 <= y < game_state.data.layout.height

    def get_manhattan_distance(self, pos1, pos2):
        """
        Calculates Manhattan distance between two points.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def choose_action(self, game_state):
        """
        Chooses an action using a priority queue based on evaluated features.
        """
        actions = game_state.get_legal_actions(self.index)
        pq = PriorityQueue()

        for action in actions:
            features = self.get_features(game_state, action)
            score = sum(features[key] * self.get_weights(game_state, action)[key] for key in features)
            pq.put((-score, action))  # Higher scores have higher priority

        return pq.get()[1]  # Return action with the highest score