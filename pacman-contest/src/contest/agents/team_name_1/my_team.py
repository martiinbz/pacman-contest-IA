import random
import contest.util as util
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveAgent', second='DefensiveAgent', num_training=0):
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


class OffensiveAgent(ReflexCaptureAgent):
    """
    An offensive agent that seeks food and avoids ghosts.
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.last_action = None

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        # Si lleva al menos dos comidas, evaluar si es mejor ir a por otra o regresar
        if my_state.num_carrying >= 1:
            if food:
                food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
                closest_food = min(food_distances, key=lambda x: x[0])[1]

                defender_distances = [
                    (self.get_maze_distance(closest_food, defender.get_position()), defender.get_position())
                    for defender in defenders
                    if defender.get_position()
                ]
                if defender_distances:
                    closest_defender_dist = min(defender_distances, key=lambda x: x[0])[0]
                    my_dist_to_food = self.get_maze_distance(my_pos, closest_food)

                    # Si la distancia al fantasma es menor o igual que la del agente, regresa a su campo
                    if closest_defender_dist <= my_dist_to_food:
                        return self.get_action_towards(game_state, self.start)

                    # Evaluar si ir a por otra comida o regresar
                    next_food = self.get_next_closest_food(game_state, closest_food)
                    if next_food:
                        next_food_dist = self.get_maze_distance(closest_food, next_food)
                        if next_food_dist <= 1:  # Consider next food only if very close
                            return self.get_action_towards(game_state, next_food)
                    return self.get_action_towards(game_state, self.start)

        # Si no hay comida o está en fase de recolección, comportarse normalmente
        if food:
            food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
            closest_food = min(food_distances, key=lambda x: x[0])[1]
            return self.get_action_towards(game_state, closest_food)

        # Si no hay comida, regresar al inicio
        return self.get_action_towards(game_state, self.start)

    def get_next_closest_food(self, game_state, current_food):
        """
        Returns the next closest food to the current food position.
        """
        my_pos = game_state.get_agent_state(self.index).get_position()
        food_list = self.get_food(game_state).as_list()
        food_list.remove(current_food)
        if food_list:
            food_distances = [(self.get_maze_distance(current_food, food_pos), food_pos) for food_pos in food_list]
            closest_food = min(food_distances, key=lambda x: x[0])[1]
            if self.get_maze_distance(my_pos, closest_food) <= 2:
                return closest_food
        return None

    def get_action_towards(self, game_state, target_pos):
        """
        Returns the action that moves the agent closer to the target position.
        """
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        min_distance = float('inf')

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(new_pos, target_pos)
            if distance < min_distance:
                min_distance = distance
                best_action = action

        return best_action

    def choose_non_repetitive_action(self, actions):
        """
        Chooses an action that is not the same as the last action.
        """
        if self.last_action is None:
            self.last_action = random.choice(actions)
        else:
            actions = [action for action in actions if action != self.last_action]
            if actions:
                self.last_action = random.choice(actions)
            else:
                self.last_action = random.choice(actions)
        return self.last_action

    def get_weights(self, game_state, action):
        """
        Returns the weights for each feature.
        """
        return {
            'distance_to_food': -1.0,
            'invader_distance': 1.5
        }


class DefensiveAgent(ReflexCaptureAgent):
    """
    A defensive agent that protects its side and captures invaders.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        # Feature: distance to the nearest food
        food_list = self.get_food(successor).as_list()
        if food_list:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_food': -1,
            'stop': -100,
            'reverse': -2
        }