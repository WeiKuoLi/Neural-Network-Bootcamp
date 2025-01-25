import chess

# Create a new chess board
board = chess.Board()

# Print the initial board
print("Initial Board:")
print(board)
print("FEN:", board.fen())

# Make a move (e.g., e2 to e4)
move = chess.Move.from_uci("e2e4")
if move in board.legal_moves:
    board.push(move)
    print("\nAfter move e2e4:")
    print(board)
    print("FEN:", board.fen())
else:
    print(f"Move {move} is not valid!")

# Make another move (e.g., e7 to e5)
move = chess.Move.from_uci("e7e5")
if move in board.legal_moves:
    board.push(move)
    print("\nAfter move e7e5:")
    print(board)
    print("FEN:", board.fen())
else:
    print(f"Move {move} is not valid!")

# Check if the game is over
if board.is_checkmate():
    print("\nCheckmate!")
elif board.is_stalemate():
    print("\nStalemate!")
elif board.is_game_over():
    print("\nGame over!")
else:
    print("\nThe game is still ongoing.")

