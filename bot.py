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
        self.pawn_value = 1
        self.knight_value = 3
        self.bishop_value = 3
        self.rook_value = 5
        self.queen_value = 9

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
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        return self.best_move(5).uci()
    
    def evaluate(self) -> float:
        material = 0
        material += len(self.board.pieces(chess.PAWN, chess.WHITE)) * self.pawn_value
        material += len(self.board.pieces(chess.KNIGHT, chess.WHITE)) * self.knight_value
        material += len(self.board.pieces(chess.BISHOP, chess.WHITE)) * self.bishop_value
        material += len(self.board.pieces(chess.ROOK, chess.WHITE)) * self.rook_value
        material += len(self.board.pieces(chess.QUEEN, chess.WHITE)) * self.queen_value
        material -= len(self.board.pieces(chess.PAWN, chess.BLACK)) * self.pawn_value
        material -= len(self.board.pieces(chess.KNIGHT, chess.BLACK)) * self.knight_value
        material -= len(self.board.pieces(chess.BISHOP, chess.BLACK)) * self.bishop_value
        material -= len(self.board.pieces(chess.ROOK, chess.BLACK)) * self.rook_value
        material -= len(self.board.pieces(chess.QUEEN, chess.BLACK)) * self.queen_value
        return material

    def minimax(self, depth, alpha, beta, is_maximizing):
        if depth == 0:
            return -self.evaluate()
        if is_maximizing:
            best_move = -9999
            for move in self.board.legal_moves:
                self.board.push(move)
                best_move = max(best_move, self.minimax(depth - 1, alpha, beta, not is_maximizing))
                self.board.pop()
                alpha = max(alpha, best_move)
                if beta <= alpha:
                    return best_move
            return best_move
        else:
            best_move = 9999
            for move in self.board.legal_moves:
                self.board.push(move)
                best_move = min(best_move, self.minimax(depth - 1, alpha, beta, not is_maximizing))
                self.board.pop()
                beta = min(beta, best_move)
                if beta <= alpha:
                    return best_move
            return best_move
        
    def best_move(self, depth):
        best_move = None
        best_value = -9999
        alpha = -10000
        beta = 10000
        for move in self.board.legal_moves:
            self.board.push(move)
            board_value = self.minimax(depth - 1, alpha, beta, False)
            self.board.pop()
            if board_value > best_value:
                best_value = board_value
                best_move = move
        return best_move


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
