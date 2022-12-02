# Student agent: Add your own agent here
from copy import deepcopy
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
    # this returns a boolean as to if the move is valid
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
    # This returns a boolean as to if the game is over, and the scores of the two players
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
        
    # This returns a list of all possible moves from the current position
    # [((new row, new column), direction of wall), ...]
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

    # This returns the best move from the current position based on the current state of the board
    def get_best_moves(self, chess_board, my_pos, adv_pos, max_step):
        possible_moves = self.all_moves(chess_board,my_pos, adv_pos, max_step)

        # These arrays will be used to save the positions that are reachable from the current position.
        # They are separated by whether they will cause the agent to have a higher score than the advarsay, the same score, or a lower score.
        higher_scores = []
        same_scores = []
        lower_scores = []

        for move in possible_moves:
            simulated_board = deepcopy(chess_board)
            #put barrier down on the simulated board
            StudentAgent.set_barrier(simulated_board,move[0][0], move[0][1], move[1])
            # find the score of the simulated board
            is_endgame, score1, score2 = StudentAgent.check_endgame(simulated_board, move[0], adv_pos)
            
            #check if its the end of the game
            if is_endgame:
                if score1 > score2: return move # return winning move to end the game 
            else:
                h = StudentAgent.heuristic(simulated_board, move[0], adv_pos)
                # save the move in the appropriate array along with the heuristic value
                if score1 > score2:
                    higher_scores.append((move, is_endgame, h))
                elif score1 == score2:
                    same_scores.append((move, is_endgame, h))
                else:
                    lower_scores.append((move, is_endgame))

        # a "score" has the form (move, is_endgame, heuristic value)

        lowest_h = 500
        lowest_h_move = -1
        # if there is no winning move, but there is a move that results in a higher score for the agent, return that move
        # in the case where there is more than one move that fits this criteria, return the move with the lowest heuristic value
        for score in higher_scores:
            if score[2] < lowest_h:
                lowest_h = score[2]
                lowest_h_move = score[0]
        if lowest_h_move != -1:
            return lowest_h_move
        # if there is no move where the agent has a higher score, but there is a move that results in the same score and is not the end of the game, return it
        # in the case where there is more than one move that fits this criteria, return the move with the lowest heuristic value
        for score in same_scores:
            if score[1] == False:
                if score[2] < lowest_h:
                    lowest_h = score[2]
                    lowest_h_move = score[0]
        if lowest_h_move != -1:
            return lowest_h_move
        # if there is no move where the scores are the same where the game is not over, return the move that causes the game to draw
        for score in same_scores:
            return score[0]
        # if there is no draw, return the move that results in a lower score where the game does not end
        for score in lower_scores:
            if score[1] == False: 
                return score[0]
        # if there is no move that results in a lower score where the game does not end, return the move that results in a lower score where the game ends
        for score in lower_scores:  
            return score[0]

    # This returns an int. The lower it is, the better that move is
    def heuristic(chess_board, pos, adv_pos):
        # This is the heuristic function that will be used to determine the best move to make.
        # This first part counts the amount of walls in the area around the agent
        x_pos, y_pos = tuple(pos)
        num_walls = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    if chess_board[x_pos + i - 1,y_pos + j - 1,k]:
                        num_walls += 1

        # This part checks how close the agent is to the center
        center = chess_board.shape[0] // 2
        dist = abs(x_pos - center) + abs(y_pos - center)

        # This part checks how many walls are around the adversary
        adv_x, adv_y = tuple(adv_pos)
        num_adv_walls = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    if chess_board[adv_x + i - 1,adv_y + j - 1,k]:
                        num_adv_walls += 1

        # We would like to minimize the walls around us, as well as our distance to the center, and would like to maximize
        # the walls around the adversary. So we return the value below and try to mimimize it.
        return num_walls * 3 + dist - num_adv_walls
    

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
    
        return StudentAgent.get_best_moves(self,chess_board, my_pos, adv_pos, max_step)