"""
Testing bot (see if changes to bot.py are good)
"""

import random
import chess
import time
from collections.abc import Iterator
from bot import Bot, TranspositionTable
from pieceTables import piece_tables, convert_piece_table, reverse_table
from contextlib import contextmanager

@contextmanager
def game_manager() -> Iterator[None]:
    """Creates context for game."""

    print("===== GAME STARTED =====")
    ping: float = time.perf_counter()
    try:
        # DO NOT EDIT. This will be replaced w/ judging context manager.
        yield
    finally:
        pong: float = time.perf_counter()
        total = pong - ping
        print(f"Total game time = {total:.3f} seconds")
    print("===== GAME ENDED =====")

class Bot2:
    def __init__(self, fen=None):
        self.board = chess.Board(fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        self.transposition_table = TranspositionTable()
        self.transposition_table.zobrist_hash(self.board)
        self.piece_tables = {piece: convert_piece_table(piece_table) for piece, piece_table in piece_tables.items()}
        self.best_move = None
        self.best_eval = None

    def check_move_is_legal(self, initial_position, new_position) -> bool:

        """
            To check if, from an initial position, the new position is valid.

            Args:
                initial_position (str): The starting position given chess notation.
                new_position (str): The new position given chess notation.

            Returns:
                bool: If this move is legal
        """

        return chess.Move.from_uci(initial_position + new_position) in self.board.legal_moves
    
    def next_move(self, maximizing_player, depth=5) -> str:
        """
            The main call and response loop for playing a game of chess.
            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        return self.minimax_root(self.board, maximizing_player, depth)

    def evaluate(self, board) -> float:
        
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.15,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 9999 # King is worth "infinite" points
        }

        piece_endgame_weights = {
            chess.KNIGHT: 0.4,
            chess.BISHOP: 0.4,
            chess.ROOK: 1.0,
            chess.QUEEN: 2.5,
        }

        endgame_weight = 0
        starting_endgame_weight = 6.1

        def get_positional_bonus(piece, square):
            # Simple positional bonuses for pieces
            bonus = 0
            endgame_weight = 0
            if piece.color == chess.WHITE:
                bonus += self.piece_tables[piece.piece_type][square] * 0.01
            else:
                bonus += reverse_table(self.piece_tables[piece.piece_type])[square] * 0.01
            if piece.piece_type == chess.KING:
                bonus += get_king_safety_bonus(piece, square)
            elif piece.piece_type is not chess.PAWN: # Minor pieces
                controlled_squares = board.attacks(square)
                if piece.piece_type == chess.KNIGHT:
                    bonus += .02 * len(controlled_squares)  # Bonus for controlling squares
                elif piece.piece_type == chess.BISHOP:
                    for attacked_square in controlled_squares:
                        attacked_piece = board.piece_at(attacked_square)
                        if attacked_piece is not None and attacked_piece.color != piece.color:
                            bonus += 0.03 # Bonus for attacking pieces
                        else:
                            bonus += 0.015 # Bonus for controlling squares
                elif piece.piece_type == chess.ROOK:
                    bonus += get_rook_open_file_bonus(piece, square) + get_connected_rooks_bonus(piece, square)
            return bonus
        
        def get_king_safety_bonus(piece, square):
            bonus = 0
            for attacked_square in board.attacks(square):
                adjacent_piece = board.piece_at(attacked_square)
                if adjacent_piece is not None:
                    if adjacent_piece.piece_type == chess.PAWN and adjacent_piece.color == piece.color:
                        bonus += 0.05 # Bonus for king being protected by pawn
                    elif adjacent_piece.color == piece.color:
                        bonus += 0.025 # Bonus for king being protected by minor piece
            return bonus
        
        def get_rook_open_file_bonus(piece, square):
            rook_rank = chess.square_rank(square)
            if (piece.color == chess.WHITE and rook_rank == 0) or (piece.color == chess.BLACK and rook_rank == 7): # Check if rook is on the 1st or 8th rank
                rook_file = chess.square_file(square)
                for pawn_square in board.pieces(chess.PAWN, piece.color):
                    if chess.square_file(pawn_square) == rook_file:
                        return 0
                return 0.15 # Give bonus for rook being on an open file - not blocked by its own pawns
            return 0
        
        def get_connected_rooks_bonus(piece, square):
            bonus = 0
            rooks = board.pieces(chess.ROOK, piece.color)
            if len(rooks) > 1:
                for rook_square in rooks:
                    if rook_square != square and rook_square in board.attacks(square):
                        bonus += 0.12 # Give bonus for rooks being connected
            return bonus

        if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
        elif board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return 0
        else:
            score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    value = piece_values[piece.piece_type] + get_positional_bonus(piece, square)
                    if piece.color == chess.WHITE:
                        score += value
                    else:
                        score -= value
                    #print(f"Piece: {piece}, Score: {value}")
                
            return score

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        hash = self.transposition_table.probe()
        if hash is not None and hash[0] >= depth:
            if board.fen().split(" ")[0] != hash[3]:
                print("!")
                print(hash[3])
                print(board.fen().split(" ")[0])
            if hash[2] == 0: # Exact
                return hash[1]
            elif hash[2] == 1: # Lower bound
                alpha = max(alpha, hash[1])
            elif hash[2] == 2: # Upper bound
                beta = min(beta, hash[1])
            if alpha >= beta: # If the bounds overlap, return the eval
                return hash[1]

        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        legal_moves = list(board.legal_moves)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(board, move), reverse=True)

        if maximizing_player: # White
            for move in ordered_moves:
                eval = self.push_and_pop_move(board, depth, not maximizing_player, alpha, beta, move)
                if eval > alpha:
                    alpha = eval
                    self.transposition_table.store(depth, eval, 0, board.fen())
                if beta <= alpha:
                    self.transposition_table.store(depth, alpha, 1, board.fen())
                    return alpha
            self.transposition_table.store(depth, alpha, 2, board.fen())
            return alpha
        else:
            for move in ordered_moves:
                eval = self.push_and_pop_move(board, depth, not maximizing_player, alpha, beta, move)
                if eval < beta:
                    beta = eval
                    self.transposition_table.store(depth, eval, 0, board.fen())
                if beta <= alpha:
                    self.transposition_table.store(depth, beta, 1, board.fen()  )
                    return beta
            self.transposition_table.store(depth, beta, 2, board.fen())
            return beta
        
    def minimax_root(self, board, maximizing_player, depth):
        legal_moves = list(board.legal_moves)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(board, move), reverse=True)
        self.best_eval = float('-inf')

        for move in ordered_moves:
            eval = self.push_and_pop_move(board, depth - 1, not maximizing_player, float('-inf'), float('inf'), move)
            if eval > self.best_eval:
                self.best_eval = eval
                self.best_move = move  
        return str(self.best_move)
    
    def push_and_pop_move(self, board, depth, maximizing_player, alpha, beta, move):
        prev_hash = self.transposition_table.get_hash()
        board.push(move)
        self.transposition_table.zobrist_hash(board)
        eval = self.minimax(board, depth - 1, maximizing_player, alpha, beta)
        board.pop()
        self.transposition_table.set_hash(prev_hash)
        return eval
        
    def move_priority(self, board, move):
        # Define a priority for each move type: checks > captures > major piece moves  > other moves
        if board.gives_check(move):
            return 4
        elif board.is_capture(move):
            return 3
        elif not board.is_zeroing(move):
            return 2
        else:
            return 1
        
    def get_eval(self):
        return (f"Best move: {self.best_move}, Eval: {self.best_eval}")


if __name__ == "__main__":

    chess_bot = Bot2()  # you can enter a FEN here, like Bot("...")
    original_bot = Bot()
    with game_manager():

        playing = True
        player_playing = False
        depth = 5

        while playing:
            if chess_bot.board.turn:
                print("Bot2 move")
                move = chess_bot.next_move(depth)
                chess_bot.board.push_san(move)
                original_bot.board.push_san(move)
                print(chess_bot.get_eval())
            else:
                print("Original bot move")
                move = original_bot.next_move(depth)
                original_bot.board.push_san(move)
                chess_bot.board.push_san(move)
                print(original_bot.get_eval())

            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")

                # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                print(chess_bot.board.outcome())

                playing = False

