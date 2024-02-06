# This is necessary to find the main code
import sys
sys.path.insert(0, '../bomberman')
# Import necessary stuff
from entity import CharacterEntity
from colorama import Fore, Back
import heapq
import math

class TestCharacter(CharacterEntity):

    def do(self, wrld):
        start = (self.x, self.y)
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

        for i in range(0, len(path)):
            move = path[-1]
            self.move(move[0] - self.x, move[1] - self.y)
            path = path[:-1]

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

                    
                








