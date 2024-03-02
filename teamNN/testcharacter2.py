# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import math
import heapq
from enum import Enum
# import torch
# import torch.nn as nn
import numpy as np
import random
import os



class World_State(Enum):
    WORLD_SAFE = 0      # not used
    WORLD_HAS_BOMB = 1


class TestCharacter(CharacterEntity):
    weights_file = 'default_weights'

    def __init__(self, name, avatar, x, y, variant_num):
        CharacterEntity.__init__(self, name, avatar, x, y)
        self.variant_num = variant_num
        # keep track of bomb explosion time
        self.my_bomb_timer = math.inf

    def do(self, wrld):
        # always WORLD_HAS_BOMB
        if len(self.get_bomb_location(wrld)) == 0 \
                and len(self.get_monster_location(wrld)) == 0 \
                and len(self.get_explosion_location(wrld)) == 0:

            world_state = World_State.WORLD_SAFE
        else:
            world_state = World_State.WORLD_HAS_BOMB
        # print(world_state)

        # reduce timer
        if self.my_bomb_timer == 0:
            self.my_bomb_timer = math.inf
        # math.inf-1 == math.inf, but math.inf-1 is not math.inf. Use == for comparison
        # so nothing will change if timer = inf (no bomb)
        self.my_bomb_timer -= 1

        # check if close to exit
        near_exit = max(abs(wrld.exitcell[0] - self.x), abs(wrld.exitcell[1] - self.y)) <= 1
        # print(f"Pos {(self.x, self.y)}\tExit {wrld.exitcell}\t{near_exit}\t {abs(wrld.exitcell[0] - self.x), abs(wrld.exitcell[1] - self.y)}")
        if near_exit:
            self.move(wrld.exitcell[0] - self.x, wrld.exitcell[1] - self.y)
            self.delete_w()
            return

        # while the character is not at exit yet
        match world_state:
            # perform A* if world is safe
            case World_State.WORLD_SAFE:
                # get path to exit
                path = self.astar(self, wrld)
                # for i in range(0, len(path)):
                move = path[0]
                # find the interaction point between the path and wall
                if not wrld.wall_at(move[0], move[1]):
                    self.move(move[0] - self.x, move[1] - self.y)
                    path = path[1:]
                elif wrld.explosion_at(move[0], move[1]):
                    self.move(0, 0)
                    # print("\n\nnext step in explosion\n\n")
                    # print(move)
                # place a bomb if the next move is a wall
                else:
                    self.place_bomb()
                    self.my_bomb_timer = wrld.bomb_time +1
                    world_state = World_State.WORLD_HAS_BOMB
                    # break

            # Q learning if world is not safe
            case World_State.WORLD_HAS_BOMB:
                weight = self.read_weights()
                new_weight = self.approximate_Q(weight, wrld)
                self.write_weights(new_weight)


    def read_weights(self):
        self.weights_file = 'weights' + str(self.variant_num)
        if os.path.exists(self.weights_file):
            try:
                weights = [float(line.rstrip('\n')) for line in open(self.weights_file, 'r')]
            except ValueError:
                print('ERROR (ValueError): unexpected newline character encountered')
        else:
                weights = [float(line.rstrip('\n')) for line in open('default_weights_' + str(self.variant_num), 'r')]
        return weights

    def write_weights(self, weight):
        with open(self.weights_file, 'w') as f:
            try:
                f.writelines(['%s\n' % w for w in weight])
            except TypeError:
                print('ERROR (TypeError): weights is empty (is None)')
            f.close()

    def delete_w(self):
        os.remove('weights' + str(self.variant_num))

    # perform approximate q algorithm
    def approximate_Q(self, weight, wrld):
        # hyperparams
        gamma = 0.9  # Discount factor
        alpha = 0.01  # Learning rate
        # alpha = 0.000001  # Learning rate
        # features:
        #   distance to bomb, time left for explosion, in corner, distance to exit, distance to closest wall, dist to monst, in explosion range
        # possible actions:
        #   move up, move down, move left, move right, place bomb
        possible_action_list = [(i,j) for i in range(-1, 2) for j in range(-1, 2) ]
        # dictionary of rewards for the next possible actions
        cur_reward_dict = {}
        cur_q_dict = {}
        bomb_info_dict = {} # location: bomb remaining time
        # # initialize weight
        # weight = [random.randint(1, 100) for _ in range(7)]
        # pick action
        for action in possible_action_list:
            next_location = (self.x + action[0], self.y + action[1])
            next_x, next_y = next_location
            # create new world based on character movement
            new_wrld = wrld.from_world(wrld)
            bomb_info_dict[next_location] = math.inf
            # if self.check_inbound(next_x, next_y, new_wrld) and new_wrld.empty_at(next_x, next_y) and not self.in_explosion_range(next_x, next_y, new_wrld):
            if self.check_inbound(next_x, next_y, new_wrld) and not self.in_explosion_range(next_x, next_y, wrld):

                char = list(new_wrld.characters.values())[0][0]

                if new_wrld.empty_at(next_x, next_y):
                    char.move(action[0], action[1])
                else:
                    # not valid way, try to place bomb
                    # only place bomb if no bomb, else will replace real bomb
                    if len(self.get_bomb_location(wrld)) == 0 \
                            and new_wrld.wall_at(next_x, next_y) \
                            and len(self.get_explosion_location(wrld)) == 0:
                        char.place_bomb()
                        bomb_info_dict[next_location] = new_wrld.bomb_time +1
                        print("Imaginary bomb")
                    else:
                        continue


                # the reward for this step
                # make reward always positive to update weight correctly
                reward = new_wrld.scores[char.name] + 5000

                cur_reward_dict[next_location] = reward
                next_feature_vector = self.get_feature_vector(next_x, next_y, new_wrld)
                q_val = np.dot(weight, next_feature_vector)
                cur_q_dict[next_location] = q_val

        # collect s' and reward
        print(cur_q_dict)
        current_move = max(cur_q_dict, key = cur_q_dict.get)
        current_reward = cur_reward_dict[current_move]
        current_q = cur_q_dict[current_move]
        print(f"Current Pos: {(self.x, self.y)}\tDecided Move: {current_move}")
        # copy bomb
        if bomb_info_dict[current_move] != math.inf:
            self.place_bomb()
            self.my_bomb_timer = bomb_info_dict[current_move]
        else:
            # make the move
            self.move(current_move[0]-self.x, current_move[1]-self.y)

        current_feature_vector = self.get_feature_vector(self.x, self.y, wrld)

        next_q_dict = {}
        # find maxQ(s', a')
        for action in possible_action_list:
            next_location = (current_move[0] + action[0], current_move[1] + action[1])
            # create new world based on character movement
            new_wrld = wrld.from_world(wrld)
            char = next(iter(new_wrld.characters.values()))[0]
            char.move(action[0], action[1])
            next_x = next_location[0]
            next_y = next_location[1]
            if self.check_inbound(next_x, next_y, new_wrld) and new_wrld.empty_at(next_x, next_y)  and not self.in_explosion_range(next_x, next_y, new_wrld): 
                
                next_feature_vector = self.get_feature_vector(next_x, next_y, new_wrld)
                q_val = np.dot(weight, next_feature_vector)
                next_q_dict[next_location] = q_val
        next_q = max(next_q_dict.values())

        # calculate delta
        delta = (current_reward + gamma * next_q) - current_q

        # recalculate weight
        for i in range(len(weight)):
            f = current_feature_vector[i]
            weight[i] = weight[i] + alpha*delta*f

        return weight


    def get_feature_vector(self, x, y, wrld):
        exit_dist = self.dist_to_exit(x, y, wrld)
        monst_dist = self.dist_to_closest_monst(x, y, wrld)
        feature_vector = [exit_dist, monst_dist]
        return feature_vector


    # estimate distance to the exit cell
    def dist_to_exit(self, x, y, wrld):
        return 1 / (1+math.sqrt((wrld.exitcell[0]-x)**2 + 10*(wrld.exitcell[1]-y)**2))
        # return 1 / (1+math.sqrt((wrld.exitcell[0]-x)**2 + (wrld.exitcell[1]-y)**2))

    # get distance to the closest monster
    def dist_to_closest_monst(self, x, y, wrld):
        monst_loc_list = self.get_monster_location(wrld)
        dist_to_closest_monst = math.inf
        for monst_loc in monst_loc_list:
            dist_to_monst = math.sqrt((monst_loc[0]-x)**2 + (monst_loc[1]-y)**2)
            dist_to_closest_monst = min(dist_to_closest_monst, dist_to_monst)

        d = dist_to_closest_monst
        # engineer a sigmoid like function to avoid close contact to monster, 
        # while being less sensitive to far away monster
        return 1 / (1 + (d/1.5)**4)


    
    # check whether character is in the explosion range
    def in_explosion_range(self, x, y, wrld):

        bombLoc = self.get_bomb_location(wrld)
        if len(bombLoc) == 0:
            return bool(wrld.explosion_at(x,y))
        bombLoc = self.get_bomb_location(wrld)[0]
        print(f"Bomb time: {self.my_bomb_timer}")

        # run if about to blow up
        if self.my_bomb_timer <= 1:
            print("RUNNNNNNNNN")
            if x < (bombLoc[0] + wrld.expl_range + 1) and x > (bombLoc[0] - wrld.expl_range - 1) and y == bombLoc[1]:
                return 1
            if y < (bombLoc[1] + wrld.expl_range + 1) and y > (bombLoc[1] - wrld.expl_range - 1) and x == bombLoc[0]:
                return 1
            return 0

        return bool(wrld.explosion_at(x,y))




    # get location of bomb in the world
    def get_bomb_location(self, wrld):
        bombs = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.bomb_at(i, j):
                    bombs.append((i, j))
        return bombs

    # get location of monsters in the world
    def get_monster_location(self, wrld):
        monsters = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.monsters_at(i, j):
                    monsters.append((i, j))
        return monsters

    def get_explosion_location(self, wrld):
        explo_list = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.explosion_at(i, j):
                    explo_list.append((i, j))
        return explo_list

    # check whether the next movement is still inbound
    def check_inbound(self, x, y, wrld):
        if x >= 0 and x < wrld.width() and y >=0 and y < wrld.height():
            return True
        return False



    # calculate A* between character and exit
    def astar(self, char, wrld):
        start = (char.x, char.y)
        end = wrld.exitcell

        frontier = CustomPQ()
        frontier.put(start, 0)        
        came_from = {}
        came_from[start] = None
        cost = {}
        cost[start] = 0
        visited = {}
        visited[start] = True
        
        path = []

        while not frontier.is_empty():
            cur_node = frontier.get()

            if end == cur_node:
                break

            next_node_list = []

            cur_x = cur_node[0]
            cur_y = cur_node[1]
            neighbors = [(cur_x+1, cur_y),
                        (cur_x+1, cur_y+1),
                        (cur_x+1, cur_y-1),
                        (cur_x, cur_y+1),
                        (cur_x, cur_y-1),
                        (cur_x-1, cur_y),
                        (cur_x-1, cur_y+1),
                        (cur_x-1, cur_y-1),]
            for n in neighbors:
                n_x = n[0]
                n_y = n[1]
                if n_x >= 0 and n_y >= 0 and n_x < wrld.width() and n_y < wrld.height():
                    # if wrld.wall_at(n_x, n_y) != True:
                    next_node_list.append(n)
            
            for nn in next_node_list:
                new_cost = cost[cur_node] + wrld.wall_at(nn[0], nn[1])*10
                if (nn not in cost or new_cost < cost[nn]) and nn not in visited:
                    nn_x = nn[0]
                    nn_y = nn[1]
                    cost[nn] = new_cost
                    p = new_cost + math.sqrt((nn_x - end[0])**2 + (nn_y - end[1])**2)
                    frontier.put(nn, p)
                    came_from[nn] = cur_node
                    visited[nn] = True
        
        while cur_node != start:
            path.insert(0, cur_node)
            cur_node = came_from[cur_node]
        
        return path
    
class CustomPQ:
    def __init__(self):
        self.items = []

    def is_empty(self):
        return len(self.items) == 0

    def put(self, element, priority):
        found = False
        for i in range(len(self.items)):
            current_priority, current_element = self.items[i]
            if current_element == element:
                if current_priority > priority:
                    self.items[i] = (priority, element)
                    heapq.heapify(self.items)  # Re-heapify after updating priority
                found = True
                break
        if not found:
            heapq.heappush(self.items, (priority, element))

    def get(self):
        if self.items:
            return heapq.heappop(self.items)[1]
        else:
            raise IndexError("Custom priority queue is empty")
