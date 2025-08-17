import chess

# This module evaluates the material balance of a chess board.
piece_values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000  # not really used
}

pawn_table = [
    [ 0,   0,   0,   0,   0,   0,   0,   0],
    [50,  50,  50,  50,  50,  50,  50,  50],
    [10,  10,  20,  30,  30,  20,  10,  10],
    [ 5,   5,  10,  25,  25,  10,   5,   5],
    [ 0,   0,   0,  20,  20,   0,   0,   0],
    [ 5,  -5, -10,   0,   0, -10,  -5,   5],
    [ 5,  10,  10, -20, -20,  10,  10,   5],
    [ 0,   0,   0,   0,   0,   0,   0,   0]
]

knight_table = [
    [-50, -40, -30, -30, -30, -30, -40, -50],
    [-40, -20,   0,   0,   0,   0, -20, -40],
    [-30,   0,  10,  15,  15,  10,   0, -30],
    [-30,   5,  15,  20,  20,  15,   5, -30],
    [-30,   0,  15,  20,  20,  15,   0, -30],
    [-30,   5,  10,  15,  15,  10,   5, -30],
    [-40, -20,   0,   5,   5,   0, -20, -40],
    [-50, -40, -30, -30, -30, -30, -40, -50]
]

bishop_table = [
    [-20, -10, -10, -10, -10, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,  10,  10,   5,   0, -10],
    [-10,   5,   5,  10,  10,   5,   5, -10],
    [-10,   0,  10,  10,  10,  10,   0, -10],
    [-10,  10,  10,  10,  10,  10,  10, -10],
    [-10,   5,   0,   0,   0,   0,   5, -10],
    [-20, -10, -10, -10, -10, -10, -10, -20]
]

rook_table = [
    [ 0,   0,   0,   0,   0,   0,   0,   0],
    [ 5,  10,  10,  10,  10,  10,  10,   5],
    [-5,   0,   0,   0,   0,   0,   0,  -5],
    [-5,   0,   0,   0,   0,   0,   0,  -5],
    [-5,   0,   0,   0,   0,   0,   0,  -5],
    [-5,   0,   0,   0,   0,   0,   0,  -5],
    [-5,   0,   0,   0,   0,   0,   0,  -5],
    [ 0,   0,   0,   5,   5,   0,   0,   0]
]

queen_table = [
    [-20, -10, -10,  -5,  -5, -10, -10, -20],
    [-10,   0,   0,   0,   0,   0,   0, -10],
    [-10,   0,   5,   5,   5,   5,   0, -10],
    [ -5,   0,   5,   5,   5,   5,   0,  -5],
    [  0,   0,   5,   5,   5,   5,   0,  -5],
    [-10,   5,   5,   5,   5,   5,   0, -10],
    [-10,   0,   5,   0,   0,   0,   0, -10],
    [-20, -10, -10,  -5,  -5, -10, -10, -20]
]

king_table_mid = [
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-30, -40, -40, -50, -50, -40, -40, -30],
    [-20, -30, -30, -40, -40, -30, -30, -20],
    [-10, -20, -20, -20, -20, -20, -20, -10],
    [ 20,  20,   0,   0,   0,   0,  20,  20],
    [ 20,  30,  10,   0,   0,  10,  30,  20]
]

king_table_end = [
    [-50, -40, -30, -20, -20, -30, -40, -50],
    [-30, -20, -10,   0,   0, -10, -20, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  30,  40,  40,  30, -10, -30],
    [-30, -10,  20,  30,  30,  20, -10, -30],
    [-30, -30,   0,   0,   0,   0, -30, -30],
    [-50, -30, -30, -30, -30, -30, -30, -50]
]

# Reverse the tables for black pieces
pawn_table.reverse()
knight_table.reverse()
bishop_table.reverse()
rook_table.reverse()
queen_table.reverse()
king_table_mid.reverse()
king_table_end.reverse()

piece_square_scores = {
    chess.PAWN: pawn_table,
    chess.KNIGHT: knight_table,
    chess.BISHOP: bishop_table,
    chess.ROOK: rook_table,
    chess.QUEEN: queen_table,
    chess.KING: king_table_mid  # will switch to endgame table later
}

def is_endgame(board: chess.Board):
    # A simple endgame check: if there are no queens or rooks left, it's likely an endgame
    white_pieces = len(board.pieces(chess.KNIGHT, chess.WHITE)) + \
                   len(board.pieces(chess.BISHOP, chess.WHITE)) + \
                   len(board.pieces(chess.ROOK, chess.WHITE)) + \
                   len(board.pieces(chess.QUEEN, chess.WHITE))
    
    black_pieces = len(board.pieces(chess.KNIGHT, chess.BLACK)) + \
                   len(board.pieces(chess.BISHOP, chess.BLACK)) + \
                   len(board.pieces(chess.ROOK, chess.BLACK)) + \
                   len(board.pieces(chess.QUEEN, chess.BLACK))

    return True if (white_pieces + black_pieces <= 4) else False

def material_score(board: chess.Board):
    score = 0
    for piece_type in piece_values:
        score += len(board.pieces(piece_type, chess.WHITE)) * piece_values[piece_type]
        score -= len(board.pieces(piece_type, chess.BLACK)) * piece_values[piece_type]
    return score

def piece_square_score(board: chess.Board):
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)

        if piece:
            table = piece_square_scores[piece.piece_type]
            if piece.piece_type == chess.KING:
                if is_endgame(board):
                    table = king_table_end
                else:
                    pass  # use midgame table

            if piece.color == chess.WHITE:
                score += table[chess.square_rank(square)][chess.square_file(square)]
            else:
                square = chess.square_mirror(square)
                score -= table[chess.square_rank(square)][chess.square_file(square)]
    
    return score

def evaluate(board: chess.Board):
    if board.is_checkmate():
        return -99999 if board.turn else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    score += material_score(board)
    score += piece_square_score(board)

    return score

if __name__ == "__main__":
    # Example usage
    board = chess.Board()
    print(board)
    print("Initial board evaluation:", evaluate(board))

    # Make a move and evaluate again
    board.push_san("e4")
    print(board)
    print("After 1.e4 evaluation:", evaluate(board))

    # Make another move and evaluate
    board.push_san("d5")
    print(board)
    print("After 1.e4 e5 evaluation:", evaluate(board))

    board.push_san("e4")
    print(board)
    print("After 1.e4 e5 evaluation:", evaluate(board))