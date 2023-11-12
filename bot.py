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
        depth = 5 # change depth here
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        legal_moves = list(self.board.legal_moves)

        # Order moves based on the priority: captures > pawn moves > castling > other moves (for ab pruning)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(self.board, move), reverse=True)

        best_move = None
        best_eval = float('-inf')

        for move in ordered_moves:
            self.board.push(move)
            eval = self.minimax(self.board, depth - 1, False, float('-inf'), float('inf')) # We play as black
            self.board.pop()

            if eval > best_eval:
                best_eval = eval
                best_move = move

        return str(best_move)

    def evaluate(self, board) -> float:

        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3.15, # Rank bishops slightly higher than knights
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 9999 # King is worth "infinite" points
        }

        def get_positional_bonus(piece, square):
            # Simple positional bonuses for pieces
            bonus = 0
            if piece.piece_type == chess.PAWN:
                bonus += 0.1 * (7 - chess.rank_index(square))
            elif piece.piece_type == chess.KNIGHT:
                bonus += 0.2 * (4 - abs(3 - chess.file_index(square)))  # Bonus for controlling the center
            elif piece.piece_type == chess.BISHOP:
                bonus += 0.1 * (7 - chess.rank_index(square))  # Bonus for controlling long diagonals
            elif piece.piece_type == chess.ROOK:
                bonus += 0.1 * (7 - chess.rank_index(square))  # Bonus for controlling open files
            elif piece.piece_type == chess.QUEEN:
                bonus += 0.1 * (7 - chess.rank_index(square))  # Bonus for queen mobility
            return bonus
        
        def king_safety(board, color):
            # Evaluate king safety based on pawn structure and piece positions
            king_square = board.king(color)

            # Penalty for exposed king
            safety_score = 0
            if board.is_checkmate():
                safety_score -= 1000  # Strong penalty for checkmate

            # Bonus for pawn shield in front of the king
            pawn_shield_squares = chess.SquareSet(chess.pawn_push(color, king_square)) & board.pieces(chess.PAWN, color)
            safety_score += len(pawn_shield_squares) * 0.1

            # Penalty for open lines towards the king
            enemy_rooks = board.pieces(chess.ROOK, not color)
            for enemy_rook_square in enemy_rooks:
                if chess.square_file(enemy_rook_square) == chess.square_file(king_square):
                    safety_score -= 0.2  # Penalty for open file towards the king

            return safety_score

        if board.is_checkmate():
            if board.turn:
                return float('-inf')
            else:
                return float('inf')
        elif board.is_stalemate() or board.is_insufficient_material():
            return 0
        else:
            score = 0
            for square in chess.SQUARES:
                piece = board.piece_at(square)
                if piece is not None:
                    if piece.color == chess.WHITE:
                        score += piece_values[piece.piece_type]
                    else:
                        score -= piece_values[piece.piece_type]
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
        
    def move_priority(self, board, move):
        # Define a priority for each move type: captures > pawn moves > castling > other moves
        if board.is_capture(move):
            return 3
        elif board.is_zeroing(move):
            return 2
        elif board.is_castling(move):
            return 1
        else:
            return 0


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

