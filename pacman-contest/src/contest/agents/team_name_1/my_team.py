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
    return [eval(first)(first_index), eval(second)(second_index)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        values = [self.evaluate(game_state, a) for a in actions]
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
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        if food_list:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        features['stop'] = 1 if action == Directions.STOP else 0
        current_direction = game_state.get_agent_state(self.index).configuration.direction
        successor_direction = successor.get_agent_state(self.index).configuration.direction
        features['reverse'] = 1 if successor_direction == Directions.REVERSE[current_direction] else 0

        capsules = self.get_capsules(successor)
        if capsules:
            min_capsule_distance = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_capsule'] = min_capsule_distance
        else:
            features['distance_to_capsule'] = 0

        return features

    def get_weights(self, game_state, action):
        return {
            'distance_to_food': -1.0,
            'num_invaders': -1000.0,
            'stop': -100.0,
            'reverse': -2.0,
            'distance_to_capsule': -1.0
        }

class OffensiveAgent(ReflexCaptureAgent):
    """
    An offensive agent that seeks food and avoids defenders.
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

        # when carrying food, decide if it should return home or keep collecting
        if my_state.num_carrying >= 1:
            if food:
                food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
                closest_food = min(food_distances, key=lambda x: x[0])[1]
                
                defender_distances = [
                    (self.get_maze_distance(closest_food, defender.get_position()), defender.get_position())
                    for defender in defenders
                    if defender.get_position()
                ]
                #
                if defender_distances:
                    closest_defender_dist = min(defender_distances, key=lambda x: x[0])[0]
                    my_dist_to_food = self.get_maze_distance(my_pos, closest_food)

                    # return home if defender is too close and not scared
                    if closest_defender_dist <= my_dist_to_food and not any(defender.scared_timer > 0 for defender in defenders):
                        return self.get_action_towards(game_state, self.start)

                    # eat another food if it's very close
                    next_food = self.get_next_closest_food(game_state, closest_food)
                    if next_food:
                        next_food_dist = self.get_maze_distance(closest_food, next_food)
                        if next_food_dist <= 1:
                            return self.get_action_towards(game_state, next_food)
                    #if no food is close, return home
                    return self.get_action_towards(game_state, self.start)

        # if not carrying food, continue collecting
        if food:
            food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
            closest_food = min(food_distances, key=lambda x: x[0])[1]
            safe_action = self.get_safe_action_towards(game_state, closest_food, defenders)
            if safe_action:
                return safe_action

            # if no safe action, try to find another food
            for food_pos in food:
                if food_pos != closest_food:
                    safe_action = self.get_safe_action_towards(game_state, food_pos, defenders)
                    if safe_action:
                        return safe_action

        # If no food left or no safe action, return home
        return self.get_action_towards(game_state, self.start)

    def get_next_closest_food(self, game_state, current_food):
       
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
        
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        min_distance = 100000000

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(new_pos, target_pos)
            if distance < min_distance:
                min_distance = distance
                best_action = action

        return best_action

    def get_safe_action_towards(self, game_state, target_pos, defenders):
        
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        min_distance = 1000000

        for action in actions:
            successor = self.get_successor(game_state, action)
            new_pos = successor.get_agent_state(self.index).get_position()
            distance = self.get_maze_distance(new_pos, target_pos)
            defender_distances = [self.get_maze_distance(new_pos, defender.get_position()) for defender in defenders if defender.get_position()]
            if distance < min_distance and all(dist > 1 or defender.scared_timer > 0 for dist, defender in zip(defender_distances, defenders)):
                min_distance = distance
                best_action = action

        return best_action if best_action else None

    def get_features(self, game_state, action):
        
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()
        capsules = self.get_capsules(successor)
        #distance to food
        if food_list:
            min_distance = min(self.get_maze_distance(my_pos, food) for food in food_list)
            features['distance_to_food'] = min_distance
        else:
            features['distance_to_food'] = 0
        #distance to ghost (scared or not)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in enemies if not a.is_pacman and a.get_position() is not None]
        if defenders:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            features['defender_distance'] = min(dists)
            features['scared_defender_distance'] = min(self.get_maze_distance(my_pos, a.get_position()) for a in defenders if a.scared_timer > 0)
        #distance to capsule
        if  capsules:
            min_capsule_distance = min(self.get_maze_distance(my_pos, capsule) for capsule in capsules)
            features['distance_to_capsule'] = min_capsule_distance
        else:
            features['distance_to_capsule'] = 0

        features['stop'] = 1 if action == Directions.STOP else 0

        return features

    def get_weights(self, game_state, action):
        
        return {
            'distance_to_food': -1.0,
            'defender_distance': 50,
            'scared_defender_distance': -50,
            'distance_to_capsule': -10,
            'stop': -100.0
        }

class DefensiveAgent(ReflexCaptureAgent):
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

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