"""
The Brandeis Quant Club ML/AI Competition (November 2023)

Author: @Ephraim Zimmerman
Email: quants@brandeis.edu
Website: brandeisquantclub.com; quants.devpost.com

Description:

For any technical issues or questions please feel free to reach out to
the "on-call" hackathon support member via email at quants@brandeis.edu

Website/GitHub Repository:
You can find the latest updates, documentation, and additional resources for this project on the
official website or GitHub repository: https://github.com/EphraimJZimmerman/chess_hackathon_23

License:
This code is open-source and released under the MIT License. See the LICENSE file for details.
"""

import random
import chess
import time
from collections.abc import Iterator
from contextlib import contextmanager
import test_bot



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


class Bot:
    def __init__(self, fen=None):
        self.board = chess.Board(fen if fen else "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")

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
    
    def next_move(self) -> str:
        depth = 4 # change depth here
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        return self.minimax_root(self.board, depth, False)

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
            chess.KNIGHT: 4,
            chess.BISHOP: 4,
            chess.ROOK: 10,
            chess.QUEEN: 25,
        }

        white_material = 0
        black_material = 0

        def get_positional_bonus(piece, square):
            # Simple positional bonuses for pieces
            bonus = 0
            if piece.piece_type == chess.PAWN:
                bonus += get_pawn_push_bonus(piece, square)
            elif piece.piece_type == chess.KING:
                bonus += get_king_safety_bonus(piece, square)
            else: # Minor pieces
                controlled_squares = board.attacks(square)
                if piece.piece_type == chess.KNIGHT:
                    for square in controlled_squares:
                        if board.piece_at(square) is not None:
                            bonus += 0.03
                        bonus += 0.015 * (4 - abs(3 - chess.square_file(square)))  # Bonus for controlling the center
                    #bonus += 0.025 * controlling_squares  # Bonus for controlling squares
                    #bonus += 0.2 * (4 - abs(3 - chess.file_index(square)))  # Bonus for controlling the center
                    bonus -= get_undevelopment_penalty_knight(piece, square)
                elif piece.piece_type == chess.BISHOP:
                    bonus += 0.02 * len(controlled_squares)  # Bonus for controlling long diagonals
                    for attacked_square in controlled_squares:
                        if board.piece_at(attacked_square) is not None and board.is_pinned(not piece.color, attacked_square):
                            bonus += 0.2 # Bonus for pinning pieces
                    bonus -= get_undevelopment_penalty_bishop(piece, square)
                elif piece.piece_type == chess.ROOK:
                    bonus += get_rook_open_file_bonus(piece, square) + get_connected_rooks_bonus(piece, square) + (0.02 * len(controlled_squares))  # Bonus for controlling squares
                elif piece.piece_type == chess.QUEEN:
                    bonus += 0.015 * len(controlled_squares)   # Give queen a smaller bonus since queens have more mobility 
            return bonus
        
        def get_pawn_push_bonus(piece, square):
            if piece.color == chess.WHITE:
                distance_to_promotion = 7 - chess.square_rank(square)
                promotion_square = chess.square(chess.square_file(square), 7)
            else:
                distance_to_promotion = chess.square_rank(square)
                promotion_square = chess.square(chess.square_file(square), 0)
            bonus = 2 * (3 ** (1 - distance_to_promotion)) # Exponential bonus for pawns closer to promotion - I used 2 when the pawn is 1 square away from promotion because people typically equate 2 pawns on the 7th rank to a rook
            """
            pawn_file = chess.between(promotion_square, square)
            for front_square in pawn_file:
                if board.piece_at(front_square) is not None and board.piece_at(front_square).piece_type == chess.PAWN and board.piece_at(front_square).color is not piece.color:
                    return bonus + 0.5 # If pawn is blocked, return half the bonus
            """
            return bonus
        
        def get_king_safety_bonus(piece, square):
            bonus = 0
            for attacked_square in board.attacks(square):
                adjacent_piece = board.piece_at(attacked_square)
                if adjacent_piece is not None:
                    if adjacent_piece.piece_type == chess.PAWN and adjacent_piece.color == piece.color:
                        bonus += 0.05 # Bonus for king being protected by pawn
                    bonus += 0.02 # Bonus for king being protected by other pieces
            return bonus
        
        def get_king_danger_penalty(piece, square):
            if piece.color == chess.WHITE:
                king_square = board.king(chess.WHITE)
        
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

        def get_undevelopment_penalty_knight(piece, square):
            penalty = 0
            if chess.square_rank(square) == 0 or chess.square_rank(square) == 7:
                penalty += 0.25 # knights on the first or last rank of the board are doing nothin
            if chess.square_file(square) == 0 or chess.square_file(square) == 7: 
                penalty += 0.15 # knights on the edge of the board are less valuable
            return penalty
        
        def get_undevelopment_penalty_bishop(piece, square):
            if chess.square_rank(square) == 0 or chess.square_rank(square) == 7:
                return 0.2 # Incentivize bishops to move off the first/last rank
            return 0

        if board.is_checkmate():
            if board.turn:
                return -9999
            else:
                return 9999
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        else:
            score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    value = piece_values[piece.piece_type] + get_positional_bonus(piece, square)
                    if piece.color == chess.WHITE:
                        score += value
                        white_material += value
                    else:
                        score -= value
                        black_material += value
                    # print(f"Piece: {piece}, Score: {value}")
            return score

    def minimax(self, board, depth, maximizing_player, alpha, beta):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        legal_moves = list(board.legal_moves)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(board, move), reverse=True)

        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                board.push(move) # Make the move
                eval = self.minimax(board, depth - 1, False, alpha, beta)
                board.pop() # Undo the move
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, True, alpha, beta)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def minimax_root(self, board, depth, maximizing_player):
        legal_moves = list(board.legal_moves)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(board, move), reverse=True)

        best_move = None
        best_eval = float('inf')

        for move in ordered_moves:
            board.push(move)
            eval = self.minimax(board, depth - 1, not maximizing_player, float('-inf'), float('inf'))
            board.pop()
            if eval < best_eval:
                best_eval = eval
                best_move = move
                
                print(f"Best move: {best_move}, Best eval: {best_eval}")
        
        return str(best_move)
        
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


# Add promotion stuff

if __name__ == "__main__":

    chess_bot = Bot()  # you can enter a FEN here, like Bot("...")
    with game_manager():

        """
        
        Feel free to make any adjustments as you see fit. The desired outcome 
        is to generate the next best move, regardless of whether the bot 
        is controlling the white or black pieces. The code snippet below 
        serves as a useful testing framework from which you can begin 
        developing your strategy.

        """

        playing = True

        while playing:
            if chess_bot.board.turn:
                chess_bot.board.push_san(test_bot.get_move(chess_bot.board))
            else:
                chess_bot.board.push_san(chess_bot.next_move())
            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")

                # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                print(chess_bot.board.outcome())

                playing = False

