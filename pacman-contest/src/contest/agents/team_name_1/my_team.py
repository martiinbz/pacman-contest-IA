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

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        food = self.get_food(game_state).as_list()
        capsules = self.get_capsules(game_state)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [e for e in enemies if not e.is_pacman and e.get_position() is not None]

        # Calculate distances to food
        food_distances = [(self.get_maze_distance(my_pos, food_pos), food_pos) for food_pos in food]
        closest_food = min(food_distances, key=lambda x: x[0])[1] if food_distances else None

        # Calculate distances to defenders
        defender_distances = [(self.get_maze_distance(my_pos, defender.get_position()), defender.get_position()) for defender in defenders if defender.get_position()]
        closest_defender = min(defender_distances, key=lambda x: x[0])[1] if defender_distances else None

        # Avoid defenders if close
        if closest_defender and self.get_maze_distance(my_pos, closest_defender) < 3:
            safe_actions = [action for action in actions if self.get_maze_distance(
                self.get_successor(game_state, action).get_agent_state(self.index).get_position(), closest_defender) > 3]
            if safe_actions:
                return random.choice(safe_actions)

        # Prioritize capsules if defender is nearby
        if capsules:
            capsule_distances = [(self.get_maze_distance(my_pos, cap), cap) for cap in capsules]
            closest_capsule = min(capsule_distances, key=lambda x: x[0])[1]
            if closest_capsule and closest_defender and self.get_maze_distance(my_pos, closest_defender) < 5:
                return self.get_action_towards(game_state, closest_capsule)

        # Collect food or return home if carrying a lot of food
        if my_state.num_carrying > 5 or (my_pos in self.get_our_side(game_state)):
            return self.get_action_towards(game_state, self.start)

        if closest_food:
            return self.get_action_towards(game_state, closest_food)

        # Default to stop if no better action
        return Directions.STOP

    def get_our_side(self, game_state):
        """
        Returns a set of positions that are on our side of the map.
        """
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid_x = width // 2
        if self.red:
            return {(x, y) for x in range(mid_x) for y in range(height)}
        else:
            return {(x, y) for x in range(mid_x, width) for y in range(height)}

    def get_action_towards(self, game_state, target_pos):
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

class HeuristicAgent(ReflexCaptureAgent):
    def choose_action(self, game_state):
        problem = SearchProblem(game_state, self)
        actions = self.a_star_search(problem, self.heuristic)
        if actions:
            return actions[0]
        else:
            return Directions.STOP

    def heuristic(self, state, problem):
        # Define a heuristic function
        return 0  # Placeholder heuristic

    def a_star_search(self, problem, heuristic):
        from queue import PriorityQueue

        start_state = problem.get_start_state()
        frontier = PriorityQueue()
        frontier.put((0, start_state, []))
        explored = set()

        while not frontier.empty():
            _, current_state, actions = frontier.get()

            if problem.is_goal_state(current_state):
                return actions

            if current_state not in explored:
                explored.add(current_state)

                for successor, action, cost in problem.get_successors(current_state):
                    new_actions = actions + [action]
                    cost = problem.get_cost_of_actions(new_actions)
                    heuristic_cost = cost + heuristic(successor, problem)
                    frontier.put((heuristic_cost, successor, new_actions))

        return []

class SearchProblem:
    def __init__(self, game_state, capture_agent):
        self.game_state = game_state
        self.capture_agent = capture_agent
        self.start_state = game_state.get_agent_position(capture_agent.index)
        self.goal_state = None  # Define the goal state as needed

    def get_start_state(self):
        return self.start_state

    def is_goal_state(self, state):
        # Define the goal state condition
        return state == self.goal_state

    def get_successors(self, state):
        successors = []
        actions = self.game_state.get_legal_actions(self.capture_agent.index)
        for action in actions:
            successor = self.game_state.generate_successor(self.capture_agent.index, action)
            successor_state = successor.get_agent_position(self.capture_agent.index)
            cost = 1  # Define the cost of the action
            successors.append((successor_state, action, cost))
        return successors

    def get_cost_of_actions(self, actions):
        return len(actions)  # Define the cost of a sequence of actions

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