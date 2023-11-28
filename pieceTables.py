import chess

# Piece tables for each piece type
piece_tables = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        50, 50, 50, 50, 50, 50, 50, 50,
        20, 10, 20, 30, 30, 20, 10, 20,
        10, 5, 15, 25, 25, 10, 5, 10,
        10, 0, 10, 20, 20, 5, 0, 10,
        5, 0, -5, 0, 0, -10, 0, 5,
        5, 10, 5, -20, -20, 10, 10, 5,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KNIGHT: [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20, 0, 0, 0, 0, -20, -40,
        -30, 0, 10, 15, 15, 10, 0, -30,
        -30, 5, 15, 15, 15, 15, 5, -30,
        -15, 0, 15, 15, 15, 15, 0, -15,
        -10, 5, 10, 10, 10, 10, 5, -10,
        -40, -20, 0, 5, 5, 0, -20, -40,
        -50, -20, -30, -30, -30, -30, -20, -50
    ],
    chess.BISHOP: [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10, 0, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 10, 10, 5, 0, -10,
        -10, 10, 5, 10, 10, 5, 10, -10,
        0, 0, 10, 10, 10, 10, 0, 0,
        -5, 15, 10, 5, 5, 10, 15, -5,
        -5, 10, 0, 5, 5, 0, 10, -5,
        -20, -10, -10, -10, -10, -10, -10, -20
    ],
    chess.ROOK: [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 3, 5, 8, 8, 5, 3, 0
    ],
    chess.QUEEN: [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10, 5, 0, 0, 0, 0, 0, -10,
        -10, 0, 5, 5, 5, 5, 0, -10,
        -5, 0, 5, 5, 5, 5, 0, -5,
        0, 0, 5, 5, 5, 5, 0, -5,
        -10, 5, 5, 5, 5, 5, 0, -10,
        -10, 0, 5, 0, 0, 0, 0, -10,
        -20, -10, -10, 0, 0, -10, -10, -20
    ],
    chess.KING: [
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        20, 15, 0, 0, 0, 0, 15, 20,
        20, 35, 5, 0, 0, 5, 35, 20,
    ]
}

endgame_piece_tables = {
    chess.PAWN: [
        0, 0, 0, 0, 0, 0, 0, 0,
        85, 85, 85, 85, 85, 85, 85, 85,
        50, 50, 50, 50, 50, 50, 50, 50,
        40, 40, 40, 40, 40, 40, 40, 40,
        30, 30, 30, 30, 30, 30, 30, 30,
        10, 10, 10, 10, 10, 10, 10, 10,
        10, 10, 10, 10, 10, 10, 10, 10,
        0, 0, 0, 0, 0, 0, 0, 0
    ],
    chess.KING: [ # Good to go to the center
        -50, -30, -30, -30, -30, -30, -30, -50,
        -30, -10, 0, 0, 0, 0, -10, -30,
        -30, 0, 20, 30, 30, 20, 0, -30,
        -30, 0, 30, 40, 40, 30, 0, -30,
        -30, 0, 30, 40, 40, 30, 0, -30,
        -30, 0, 20, 30, 30, 20, 0, -30,
        -30, -15, -10, 0, 0, -10, -15, -30,
        -50, -40, -30, -20, -20, -30, -40, -50
    ]
}

def convert_piece_table(square_centric_table):
    index_centric_table = [0] * 64

    for square in chess.SQUARES:
        rank, file = chess.square_rank(square), chess.square_file(square)
        index = (7 - rank) * 8 + file
        index_centric_table[index] = square_centric_table[square]

    return index_centric_table

def reverse_table(table):
    return list(reversed(table))