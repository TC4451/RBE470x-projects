# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from Bomberman.entity import CharacterEntity
from colorama import Fore, Back
import math
import heapq

class TestCharacter(CharacterEntity):

    def do(self, wrld):

        # variant 1 - no monster
        if self.get_monster_count(wrld) == 0:
            path = self.astar(self, wrld)

            for i in range(0, len(path)):
                move = path[-1]
                self.move(move[0] - self.x, move[1] - self.y)
                path = path[:-1]

        else:
            # Check monster location
            cur_m = next(iter(wrld.monsters.values()))[0]
            monster_near = False
            if math.sqrt((self.x - cur_m.x)**2 + (self.y - cur_m.y)**2) < 5:
                monster_near = True

            path = self.astar(self, wrld)
            
            if monster_near != True:
                move = path.pop(0)
                self.move(move[0] - self.x, move[1] - self.y)
        
            else:
                move = self.minimax(wrld, 3)
                self.move(move[0], move[1])

    def get_monster_count(self, wrld):
        monsters = []
        for i in range(wrld.width()):
            for j in range(wrld.height()):
                if wrld.monsters_at(i, j):
                    monsters.append((i, j))
        return len(monsters)

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
                    if wrld.wall_at(n_x, n_y) != True:
                        next_node_list.append(n)
            
            for nn in next_node_list:
                new_cost = cost[cur_node] + 1
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

    def minimax(self, wrld, depth):
        path_list = []
        d = 0
        alpha = -5000
        beta = 5000
        v = -5000

        n_wrld = wrld.from_world(wrld)        
        c = next(iter(n_wrld.characters.values()))[0]

        for dx in [-1, 0, 1]:
            if c.x + dx >= 0 and c.x + dx < wrld.width():
                for dy in [-1, 0, 1]:
                    if (dx != 0 or dy != 0) and c.y + dy >= 0 and c.y + dy < wrld.height():
                        if not wrld.wall_at(c.x + dx, c.y + dy):
                            c.move(dx, dy)
                            events = n_wrld.next()[1]

                            if len(events) != 0 and events[0].tpe == events[0].CHARACTER_KILLED_BY_MONSTER:
                                events = []
                                v = -100
                                path_list.append((v, (dx, dy)))
                                continue

                            elif len(events) != 0 and events[0].tpe == events[0].CHARACTER_FOUND_EXIT:
                                events = []
                                return (dx, dy)

                            else:
                                try:
                                    char = next(iter(n_wrld.characters.values()))[0]
                                    monst = next(iter(n_wrld.monsters.values()))[0]
                                except:
                                    continue

                                if math.sqrt((char.x - monst.x)**2 + (char.y - monst.y)**2) <= math.sqrt(8):
                                    v = -90
                                else:
                                    v, alpha, beta = TestCharacter.min_val(n_wrld, depth, d, char, monst, alpha, beta)

                                path_list.append((v, (dx, dy)))
        
        return max(path_list)[1]

    @staticmethod
    def max_val(wrld, depth, d, c, a, b):
        d += 1
        v = -100
        
        for dx in [-1, 0, 1]:
            if c.x + dx >= 0 and c.x + dx < wrld.width():
                for dy in [-1, 0, 1]:
                    if (dx != 0 or dy != 0) and c.y + dy >= 0 and c.y + dy < wrld.height():
                        if not wrld.wall_at(c.x + dx, c.y + dy):
                            c.move(dx, dy)
                            (newwrld, events) = wrld.next()
                            
                            if len(events) == 0:
                                try:
                                    char = next(iter(newwrld.characters.values()))[0]
                                    monst = next(iter(newwrld.monsters.values()))[0]
                                except:
                                    continue
                                
                                if d == depth:
                                    v = max(v, TestCharacter.get_utility(char, monst, newwrld, True))

                                    if v >= b:
                                        return v, a, b

                                    a = max(a, v)

                                else:
                                    v_new, a, b = TestCharacter.min_val(newwrld, depth, d, char, monst, a, b)
                                    v = max(v, v_new)                
                                    
                            
                            elif events[0].tpe == events[0].CHARACTER_KILLED_BY_MONSTER:
                                v = max(v, -100)
                                continue

                            elif events[0].tpe == events[0].CHARACTER_FOUND_EXIT:
                                v = 100
                                return v, a, b
        return v, a, b

    @staticmethod
    def min_val(wrld, depth, d, char, m, a, b):
        d += 1
        v_new = 100
        v = 100
        
        for dx in [-1, 0, 1]:
            if m.x + dx >= 0 and m.x + dx < wrld.width():
                for dy in [-1, 0, 1]:
                    if (dx != 0 or dy != 0) and m.y + dy >= 0 and m.y + dy < wrld.height():
                        if not wrld.wall_at(m.x + dx, m.y + dy):
                            m.move(dx, dy)
                            (newwrld, events) = wrld.next()

                            if len(events) == 0:
                                try:
                                    char = next(iter(newwrld.characters.values()))[0]
                                    monst = next(iter(newwrld.monsters.values()))[0]
                                except:
                                    continue

                                if d == depth:
                                    v = min(v, TestCharacter.get_utility(char, monst, newwrld))
                                    
                                    if v <= a:
                                        return v, a, b

                                    b = min(b, v)

                                else:
                                    v_new, a, b = TestCharacter.max_val(newwrld, depth, d, char, a, b)
                                    v = min(v, v_new)

                            elif events[0].tpe == events[0].CHARACTER_KILLED_BY_MONSTER:
                                v = -100
                                return v, a, b
        return v, a, b


    @staticmethod
    def get_utility(char, monst, wrld, p = False):
        utility = 0

        a_path = TestCharacter.astar(char, char, wrld)
        utility -= 1/2*len(a_path)
        
        dist = math.sqrt((char.x - monst.x)**2 + (char.y - monst.y)**2)

        if dist < 1.5 and p == False:
            utility -= 10
        elif dist < 1.5 and p:
            utility -= -90
        elif dist < 3:
            utility -= 5
        elif dist < 4.5:
            utility -= 3
        elif dist < 5:
            utility -= 1
        
        return utility

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
                








