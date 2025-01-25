'''
This file is a high-level plan for the project.
'''
class Planner:
    def get_possible_moves_list(self, whole_dataset):
        """Return a list of all possible moves for the given dataset."""
        ## TODO: implement this
        return all_possible_moves_list

    def move_str2idx(self, moves_str) -> int:
        """Convert a move string to an index in the all_possible_moves_list."""
        ## TODO: implement this
        return idx

    def move_idx2str(self, idx) -> str:
        """Convert an index in the all_possible_moves_list to a move string."""
        ## TODO: implement this
        return moves_str

    def get_moves(self, fen):
        """Return a list of all possible moves for the given position."""
        ## TODO: implement this
        return board_possible_moves_list

    def get_moves_arr(self, board_possible_moves_list=None):
        """['e2e4','e7e5'] -> [1.0,1.0,...,0.0]"""
        ## TODO: implement this
        return board_possible_moves_arr


plan = Planner()

def fen2board(fen):
    """Convert a FEN string to a board_array."""
    ## TODO: implement this
    return board_array 

def policy(board_array, board_possible_moves_arr):
    """Return a policy array for the given board_array."""
    ## TODO: implement this
    return board_policy_probs_array

def train_policy_network(whole_dataset, plan: Planner):
    """Train a policy network on the whole dataset."""
    ## TODO: implement this
    return policy_network

def sample_policy(board_policy_probs_array, plan: Planner):
    """Return a random move from the board_policy_probs_array."""
    ## TODO: implement this
    return policy_move_str


