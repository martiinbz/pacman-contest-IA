import random
import contest.util as util
import pickle
import os
from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveQLearningAgent', second='DefensiveQLearningAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
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
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


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
        features['successor_score'] = -len(food_list)  # self.get_score(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
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

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
class OffensiveQLearningAgent(CaptureAgent):

    def _init_(self, index, time_for_computing=.1, num_training=0, epsilon=0.05, alpha=0.6, gamma=0.7):
        super()._init_(index, time_for_computing)
        self.q_values = util.Counter()  # Almacena los valores Q
        self.epsilon = epsilon  # Probabilidad de explorar acciones aleatorias
        self.alpha = alpha  # Tasa de aprendizaje
        self.discount = gamma  # Factor de descuento para los futuros valores Q
        self.num_training = num_training  # Número de episodios de entrenamiento
        self.episodes_so_far = 0  # Contador de episodios jugados

        self.q_values_file = "./offensive_q_values.pkl"  # Ruta para guardar los valores Q

    def register_initial_state(self, game_state):
        """
        Inicializa el agente al comienzo del juego.
        """
        super().register_initial_state(game_state)
        self.load_q_values() #load q_values from file
        self.start = game_state.get_agent_position(self.index) #get_position

    def choose_action(self, game_state):
        """
        Choose an action.        
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return None

        # Exploración
        if util.flip_coin(self.epsilon):  # Exploración aleatoria
            chosen_action = random.choice(legal_actions)
        else:
            # Elegir la mejor acción basada en Q-values
            chosen_action = self.compute_action_from_q_values(game_state)

        # Obtener el siguiente estado y calcular la recompensa
        successor_state = self.get_successor(game_state, chosen_action)
        reward = self.get_reward(successor_state)

        # Actualizar los valores Q con la transición observada
        self.update(game_state, chosen_action, successor_state, reward)

        return chosen_action

    def get_successor(self, game_state, action):
        """
        Returns the successor game state after the given action is taken.
        """
        successor = game_state.generate_successor(self.index, action)
        return successor

    def get_features(self, game_state, action):
        """
        Extract features for the given state-action pair.
        """
        features = util.Counter()    # this features are similar to the reflex agents with a change in ghost dynamic
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        food_list = self.get_food(successor).as_list()

        # Feature: Distance to the nearest food
        if len(food_list) > 0:
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        # Feature: Proximity to ghosts
        ghost_indices = [i for i in self.get_opponents(game_state) if not game_state.get_agent_state(i).is_pacman]
        ghost_positions = [game_state.get_agent_state(i).get_position() for i in ghost_indices if game_state.get_agent_state(i).get_position()]
        if ghost_positions:
            ghost_distances = [self.get_maze_distance(my_pos, ghost_pos) for ghost_pos in ghost_positions]
            min_distance = min(ghost_distances)

            if min_distance == 0:  # Fantasma en la misma posición
                features['ghost_proximity'] = -float('inf')  # Penalización máxima
            elif min_distance < 3:
                features['ghost_proximity'] = -4 / min_distance  # Penalización ajustada
            else:
                features['ghost_proximity'] = 0  # Seguro
        else:
            features['ghost_proximity'] = 0  # No hay fantasmas cerca

        # Feature: Distance to home if carrying enough food
        agent_state = game_state.get_agent_state(self.index)
        if agent_state.num_carrying >= 5:  # Activar esta característica si lleva 6 o más comidas
            home_distance = self.get_maze_distance(my_pos, self.start)
            features['distance_to_home'] = home_distance
        else:
            features['distance_to_home'] = 0

        # Feature: Count of nearby food
        features['successor_score'] = -len(food_list)

        return features

    def get_weights(self):
        """
        Define the weights for features in the state-action evaluation.
        """
        return {
            'distance_to_food': -1,        # Incentivar acercarse a la comida
            'ghost_proximity': 500,      # Penalizar fuertemente estar cerca de fantasmas
            'distance_to_home': -10,      # Incentivar regresar a casa si lleva comida suficiente
            'successor_score': 100        # Recompensar áreas con más comida
        }

    def compute_q_value(self, game_state, action):
        features = self.get_features(game_state, action)                 #computar el q_value a partir de las features y los weights
        weights = self.get_weights()
        return sum(features[feature] * weights[feature] for feature in features)

    def compute_action_from_q_values(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return None
                                                        
        max_q_value = float('-inf')
        best_actions = []
        for action in legal_actions:
            q_value = self.compute_q_value(game_state, action)       #elegir la acción con el mejor q-value
            if q_value > max_q_value:
                max_q_value = q_value
                best_actions = [action]
            elif q_value == max_q_value:
                best_actions.append(action)

        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):       
        current_q_value = self.compute_q_value(state, action)       #actualizar los q_values despues de hacer la acción con los rewards dados. Este agente mezcla features y rewards creemos que no es optimo pero en un principio funciona.
        next_q_value = self.compute_value_from_q_values(next_state)
        sample = reward + self.discount * next_q_value
        self.q_values[(state, action)] = current_q_value + self.alpha * (sample - current_q_value)    

    def compute_value_from_q_values(self, game_state):
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:                                                   #calcula el valor maximo de las q_value
            return 0.0
        return max(self.compute_q_value(game_state, action) for action in legal_actions)

    def get_reward(self, game_state):
        reward = 0
        agent_state = game_state.get_agent_state(self.index)
        agent_position = agent_state.get_position()

        # Recompensas y penalizaciones ajustadas
        if agent_state.get_position() is None:
            reward -= 1000  # Penalización alta si es atrapado

        if agent_state.num_carrying >= 6:  # Incentivar regresar a casa si lleva comida suficiente
            home_distance = self.get_maze_distance(agent_position, self.start)
            reward += max(0, 10 - home_distance * 0.1)  # Recompensa por acercarse a casa

        reward -= 0.1  # Penalización por paso (eficiencia)

        return reward
    def observe_transition(self, state, action, next_state, reward):
        # Records the transition and updates Q-values.
        self.update(state, action, next_state, reward)

    def final(self, game_state):
        
        # Called at the end of the game to save Q-values.
        
        self.episodes_so_far += 1
        if self.episodes_so_far == self.num_training:
            self.epsilon = 0.0
            self.alpha = 0.0
       

    
class DefensiveQLearningAgent(CaptureAgent):
    """
    A Defensive Q-Learning agent for the Pacman Capture the Flag contest.
    This version focuses on defense: stopping invaders and protecting food.
    """

    def _init_(self, index, time_for_computing=0.1, num_training=10, epsilon=0.05, alpha=0.2, gamma=0.8):
        super()._init_(index, time_for_computing)
        self.q_values = util.Counter()
        self.epsilon = epsilon
        self.alpha = alpha
        self.discount = gamma
        self.num_training = num_training
        self.episodes_so_far = 0

        # Path to save Q-values
        self.q_values_file = "./defensive_q_values.pkl"

    def register_initial_state(self, game_state):
        """
        Initialize the agent at the beginning of the game.
        """
        super().register_initial_state(game_state)
        self.start = game_state.get_agent_position(self.index)
        self.load_q_values()

    def choose_action(self, game_state):
        """
        Choose an action for the agent using an epsilon-greedy policy.
        """
        legal_actions = game_state.get_legal_actions(self.index)

        if not legal_actions:
            return None

        # Epsilon-greedy action selection
        if util.flip_coin(self.epsilon):
            return random.choice(legal_actions)

        return self.compute_action_from_q_values(game_state)

    def get_q_value(self, state, action):
        """
        Returns the Q-value for the given state-action pair.
        """
        state_rep = self.get_state_representation(state)
        return self.q_values[(state_rep, action)]

    def compute_value_from_q_values(self, game_state):
        """
        Returns the maximum Q-value for all possible actions in a given state.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return 0.0
        return max(self.get_q_value(game_state, action) for action in legal_actions)

    def compute_action_from_q_values(self, game_state):
        """
        Selects the action with the highest Q-value from the state.
        """
        legal_actions = game_state.get_legal_actions(self.index)
        if not legal_actions:
            return None
        max_value = self.compute_value_from_q_values(game_state)
        best_actions = [action for action in legal_actions if self.get_q_value(game_state, action) == max_value]
        return random.choice(best_actions)

    def update(self, state, action, next_state, reward):
        """
        Update the Q-value for a given state-action pair based on the observed reward.
        """
        state_rep = self.get_state_representation(state)
        sample = reward + self.discount * self.compute_value_from_q_values(next_state)
        current_q_value = self.get_q_value(state, action)
        self.q_values[(state_rep, action)] = current_q_value + self.alpha * (sample - current_q_value)

    def get_state_representation(self, game_state):
        """
        Simplified state representation for defensive play.
        The agent prioritizes stopping invaders and protecting food.
        """
        agent_position = game_state.get_agent_position(self.index)
        enemies = self.get_opponents(game_state)
        invaders = [
            game_state.get_agent_position(enemy) or ("noisy", game_state.get_agent_distances()[enemy])
            for enemy in enemies
            if game_state.get_agent_state(enemy).is_pacman
        ]
        food_protected = self.get_food_you_are_defending(game_state).as_list()

        # State representation includes the agent's position, invaders, and food
        state_representation = (
            agent_position,
            tuple(invaders),
            tuple(food_protected),
        )
        return state_representation

    def observe_transition(self, state, action, next_state, delta_reward):
        """
        Called after each action to update Q-values based on the observed reward.
        """
        reward = self.get_reward(next_state)
        self.update(state, action, next_state, reward)

    def get_reward(self, game_state):
        """
        Calculate the reward for the agent based on the current game state.
        """
        reward = 0
        my_state = game_state.get_agent_state(self.index)
        my_position = my_state.get_position()

        # Penalty for letting invaders eat food
        previous_food = len(self.get_food_you_are_defending(game_state).as_list())
        current_food = len(self.get_food_you_are_defending(game_state).as_list())
        if current_food < previous_food:
            reward -= 100  # High penalty for losing food

        # Reward for being close to invaders
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if invaders:
            dists = [self.get_maze_distance(my_position, invader.get_position()) for invader in invaders]
            reward += 10 / (min(dists) + 1)  # Reward inversely proportional to the distance to invaders

        # Penalty for stopping
        if my_state.configuration.direction == Directions.STOP:
            reward -= 1

        # Reward for maintaining defensive positioning
        if not my_state.is_pacman:
            reward += 5

        return reward

    def final(self, game_state):
        """
        Called at the end of the game.
        """
        self.episodes_so_far += 1
        if self.episodes_so_far == self.num_training:
            self.epsilon = 0.0
            self.alpha = 0.0
        

    