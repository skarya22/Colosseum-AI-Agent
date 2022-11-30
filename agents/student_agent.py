# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys


@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

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
        
        # dummy return
        return my_pos, self.dir_map["u"]
    
    # Saagar
    def check_step_validity(self, chess_board, start_pos, end_pos, adv_pos, max_step, barrier_dir):
        """
        Returns a boolean value indicating whether the step is valid or not.
        """
        # Check if new barrier is taken
        if chess_board[end_pos[0], end_pos[1], barrier_dir]:
            return False
        # If not moving, then the move is valid regardless
        if (start_pos[0] == end_pos[0] and start_pos[1] == end_pos[1]):
            return True
        # Check to see if adv_pos is in the way
        if (end_pos[0] == adv_pos[0] and end_pos[1] == adv_pos[1]):
            return False
            
        # Check to see if it is possible to make a path to get to end_pos without running out of steps
        # If it is possible, then the move is valid
        # If it is not possible, then the move is invalid

        # Breadth first search through all possible paths
        queue = [(start_pos, 0)]
        visited = {start_pos}
        reached = False

        moves = ((0, 1), (1, 0), (0, -1), (-1, 0))
        while queue and not reached:
            pos, steps = queue.pop(0)
            if steps >= max_step:
                break
            for move in moves:
                new_pos = (pos[0] + move[0], pos[1] + move[1])
                if chess_board[pos[0], pos[1], self.dir_map["r"]] and move == moves[1]:
                    continue
                if chess_board[pos[0], pos[1], self.dir_map["d"]] and move == moves[0]:
                    continue
                if chess_board[pos[0], pos[1], self.dir_map["l"]] and move == moves[2]:
                    continue
                if chess_board[pos[0], pos[1], self.dir_map["u"]] and move == moves[3]:
                    continue
                if new_pos not in visited:
                    visited.add(new_pos)
                    queue.append((new_pos, steps + 1))
                    if new_pos == end_pos:
                        reached = True
                        break
        return reached

    # Catherine
    def all_moves(self, chess_board, my_pos, adv_pos, max_step):
        """
        Returns a list of all possible moves
        """
        moves = []
        for i in range(4):
            new_pos = (my_pos[0] + (i % 2) * (i - 2), my_pos[1] + (i % 2) * (2 - i)) # check this logic
            if self.check_step_validity(chess_board, my_pos, new_pos, adv_pos, max_step, i):
                moves.append(((new_pos[0], new_pos[1]), i))
        return moves

    # Saagar
    def check_result(self, chess_board, my_pos, adv_pos, max_step):
        """
        Returns a boolean value indicating whether the game is over or not.
        """
        
        return False

    # Saagar
    def create_barrier(self, chess_board, row, column, direction):
       # returns nothing
       pass

    # Catherine
    def get_viable_moves(self, chess_board, my_pos, adv_pos, max_step):
        #return (list of moves)
        return


    
        