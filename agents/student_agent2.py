# Student agent: Add your own agent here
from copy import deepcopy
import random
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np


@register_agent("student_agent2")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent2"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        self.autoplay = True

    
    # from world.py
    def check_valid_step(chess_board, start_pos, end_pos, barrier_dir, adv_pos, max_step):
        """
        from world.py
        Check if the step the agent takes is valid (reachable and within max steps).

        Parameters
        ----------
        start_pos : tuple
            The start position of the agent.
        end_pos : np.ndarray
            The end position of the agent.
        barrier_dir : int
            The direction of the barrier.
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Endpoint already has barrier or is boarder
        r, c = end_pos
        if chess_board[r, c, barrier_dir]:
            return False
        if np.array_equal(start_pos, end_pos):
            return True

        # BFS
        state_queue = [(start_pos, 0)]
        visited = {tuple(start_pos)}
        is_reached = False
        while state_queue and not is_reached:
            cur_pos, cur_step = state_queue.pop(0)
            r, c = cur_pos
            if cur_step == max_step:
                break
            for dir, move in enumerate(moves):
                if chess_board[r, c, dir]:
                    continue

                next_pos = cur_pos + move
                if np.array_equal(next_pos, adv_pos) or tuple(next_pos) in visited:
                    continue
                if np.array_equal(next_pos, end_pos):
                    is_reached = True
                    break

                visited.add(tuple(next_pos))
                state_queue.append((next_pos, cur_step + 1))

        return is_reached

    # from world.py
    def check_endgame(chess_board, p0_pos, p1_pos):
        """
        Check if the game ends and compute the current score of the agents.

        Returns
        -------
        is_endgame : bool
            Whether the game ends.
        player_1_score : int
            The score of player 1.
        player_2_score : int
            The score of player 2.
        """
        board_size = chess_board.shape[0]
        # Moves (Up, Right, Down, Left)
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Union-Find
        father = dict()
        for r in range(board_size):
            for c in range(board_size):
                father[(r, c)] = (r, c)

        def find(pos):
            if father[pos] != pos:
                father[pos] = find(father[pos])
            return father[pos]

        def union(pos1, pos2):
            father[pos1] = pos2

        for r in range(board_size):
            for c in range(board_size):
                for dir, move in enumerate(
                    moves[1:3]
                ):  # Only check down and right
                    if chess_board[r, c, dir + 1]:
                        continue
                    pos_a = find((r, c))
                    pos_b = find((r + move[0], c + move[1]))
                    if pos_a != pos_b:
                        union(pos_a, pos_b)

        for r in range(board_size):
            for c in range(board_size):
                find((r, c))
        p0_r = find(tuple(p0_pos))
        p1_r = find(tuple(p1_pos))
        p0_score = list(father.values()).count(p0_r)
        p1_score = list(father.values()).count(p1_r)
        if p0_r == p1_r:
            return False, p0_score, p1_score

        return True, p0_score, p1_score

    # Saagar
    def all_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        Get all possible moves for the agent. 
        """
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        possible_moves = []
        ori_pos = deepcopy(my_pos)

        # use BFS to find all possible moves
        queue_of_states = [(ori_pos, 0)]
        visited = {ori_pos}
        while len(queue_of_states) > 0:
            current_pos, cur_step_count = queue_of_states.pop(0)
            row, col = current_pos
            if cur_step_count <= max_step:
                for direction, move in enumerate(moves):
                    # if this row, col, direction has a wall, skip
                    if chess_board[row, col, direction]:
                        continue
                    possible_moves.append((current_pos, direction))

                    # next chosen position is the current_pos + move
                    next_pos = tuple(map(sum, zip(current_pos, move)))
                    # if the next position is already visited, skip
                    if next_pos in visited:
                        continue
                    # if the next position is the adversary's position, skip
                    if (next_pos == adv_pos):
                        continue 

                    visited.add(tuple(next_pos))
                    queue_of_states.append((next_pos, cur_step_count + 1))
                    
            else:
                break
        #this returns a list of tuples of (position, direction) where position is a tuple of (row, col) and direction is an int
        return possible_moves

    # from world.py
    def set_barrier(chess_board, r, c, dir):
        
        moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
        # Opposite Directions
        opposites = {0: 2, 1: 3, 2: 0, 3: 1}
        # Set the barrier to True
        chess_board[r, c, dir] = True
        # Set the opposite barrier to True
        move = moves[dir]
        chess_board[r + move[0], c + move[1], opposites[dir]] = True

    # Catherine
    def get_best_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        Get best moves at current state node to maximize score.
        Ideally increase points or draw

        Returns: 
        list_moves = contains the best moves to secure a victory, increase points or draws
        """
        #return (list of moves)
        possible_moves = self.all_moves(chess_board,my_pos, adv_pos, max_step)
        win_move = []
        draw_move = []
        lose_move = []

        for move in possible_moves:
            simulated_board = deepcopy(chess_board)
            #put barrier down 
            StudentAgent.set_barrier(simulated_board,move[0][0], move[0][1], move[1])
            #check if end of game using the simulated board
            is_endgame, score1, score2 = StudentAgent.check_endgame(simulated_board, move[0], adv_pos)

            #check if its the end of the game
            if is_endgame:
                if score1 > score2: return (is_endgame, simulated_board,[move] )# return winning move to end the game 
                elif score1 == score2: draw_move.append(move) #add move to potentially draw points 
            else:
                if score1 > score2 : win_move.append(move)
                elif score1 == score2 : draw_move.append(move)
                else : lose_move.append(move)
        
        if not win_move: #if we have no winning moves, we would want to draw
            if not draw_move: #if there is no drawing move, we return a list of losing moves 
                return (is_endgame, simulated_board,lose_move[0] )
            else:
                return (is_endgame, simulated_board, draw_move )
        else:
            return (is_endgame, simulated_board, win_move )

    def get_next_second_best_move(self, chess_board,first_best_moves, my_pos, adv_pos, max_step):
        second_win_move = []
        draw_move = []
        lose_move = []

        for move in first_best_moves:
            simulated_chess_board = deepcopy(chess_board)
            StudentAgent.set_barrier(simulated_chess_board,move[0][0], move[0][1], move[1])

            #check if end of game using the simulated board
            is_end, score1, score2 = StudentAgent.check_endgame(simulated_chess_board, move[0], adv_pos)

              #check if its the end of the game
            if is_end:
                if score1 > score2: return (is_end, 1, simulated_chess_board,[move] )# return winning move to end the game 
                elif score1 == score2: draw_move.append(move) #add move to potentially draw points 
            else:
                if score1 > score2 : second_win_move.append(move)
                elif score1 == score2 : draw_move.append(move)
                else : lose_move.append(move)
        
        if not second_win_move: #if we have no winning moves, we would want to draw
            if not draw_move: #if there is no drawing move, we return a list of losing moves 
                return (is_end, 0, lose_move[0] )
            else:
                return (is_end, 2, draw_move )
        else:
            return (is_end, 1, second_win_move )

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
      
        # 0 -> lost 
        # 1 -> win 
        # 2 -> draw
        is_endgame, simulated_chess_board, next_best_move = StudentAgent.get_best_moves(self,chess_board, my_pos, adv_pos, max_step)
        if(len(next_best_move) == 1 or is_endgame): return next_best_move[0]
        else:
            is_endgame, result, next_move =  StudentAgent.get_next_second_best_move(self,simulated_chess_board, next_best_move, my_pos, adv_pos, max_step)

            if(len(next_best_move) == 1): return next_move[0]

            if(result == 1 or result == 2): return random.choice(next_move)

            return next_best_move[0]

            
        # if is_endgame:
        #     return next_best_move[0]
        # else:
        #    is_endgame, result, next_second_move =  StudentAgent.get_next_second_best_move(self,simulated_chess_board, next_best_move, my_pos, adv_pos, max_step)
        #    if(result == 1 or result == 2 or (result == 1 and is_endgame == 1)): return next_second_move[0]
        #    else: return next_best_move[0]




    
        