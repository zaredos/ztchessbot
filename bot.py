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
from pieceTables import piece_tables, endgame_piece_tables, convert_piece_table, reverse_table
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
        self.transposition_table = TranspositionTable()
        self.piece_tables = {piece: convert_piece_table(piece_table) for piece, piece_table in piece_tables.items()}
        self.endgame_piece_tables = {piece: convert_piece_table(piece_table) for piece, piece_table in endgame_piece_tables.items()}
        self.reverse_piece_tables = {piece: reverse_table(piece_table) for piece, piece_table in self.piece_tables.items()}
        self.reverse_endgame_piece_tables = {piece: reverse_table(piece_table) for piece, piece_table in self.endgame_piece_tables.items()}
        self.starting_squares = [None] * 64
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is not None:
                self.starting_squares[square] = self.board.piece_at(square)

        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 9999 # King is worth "infinite" points
        }

        self.piece_endgame_weights = {
            chess.KNIGHT: 0.4,
            chess.BISHOP: 0.4,
            chess.ROOK: 1.0,
            chess.QUEEN: 2.5,
        }

        self.middle_squares = [chess.D4, chess.D5, chess.E4, chess.E5]


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
    
    def next_move(self, color, max_think_time=None) -> str:
        """
            The main call and response loop for playing a game of chess.

            Returns:
                str: The current location and the next move.
        """

        # Assume that you are playing an arbitrary game. This function, which is
        # the core "brain" of the bot, should return the next move in any circumstance.

        start_time = time.time()
        same_best_move_count = 0
        prev_best_move = None
        depth = 1
        while same_best_move_count < 2 or prev_best_move == None:
            best_move = self.negamax_root(self.board, depth, color)
            if best_move is not None:   
                if best_move == prev_best_move:
                    same_best_move_count += 1
                else:
                    same_best_move_count = 0
                prev_best_move = best_move
                if max_think_time is not None and time.time() - start_time > max_think_time:
                    break
            depth += 1
        print(f"Depth: {depth - 1}")
        return prev_best_move
    
    def get_endgame_weight(self, board):
        starting_endgame_weight = 12.2
        endgame_weight = 0
        endgame_weight += self.piece_endgame_weights[chess.QUEEN] * (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK)))
        endgame_weight += self.piece_endgame_weights[chess.ROOK] * (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK)))
        endgame_weight += self.piece_endgame_weights[chess.BISHOP] * (len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK)))
        endgame_weight += self.piece_endgame_weights[chess.KNIGHT] * (len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK)))
        return 1 - endgame_weight / starting_endgame_weight # 0 = start, 1 = end
    
    def calculate_piece_tables(self, piece, square, endgame_weight=0):
        bonus = 0
        if piece.color == chess.WHITE:
            if piece.piece_type == chess.PAWN or piece.piece_type == chess.KING:
                bonus += self.piece_tables[piece.piece_type][square] * 0.01 * (1 - endgame_weight)
                bonus += self.endgame_piece_tables[piece.piece_type][square] * 0.01 * (endgame_weight)
            else:
                bonus += self.piece_tables[piece.piece_type][square] * 0.01
        else:
            if piece.piece_type == chess.PAWN or piece.piece_type == chess.KING:
                bonus += self.reverse_piece_tables[piece.piece_type][square] * 0.01 * (1 - endgame_weight)
                bonus += self.reverse_endgame_piece_tables[piece.piece_type][square] * 0.01 * (endgame_weight)
            else:
                bonus += self.reverse_piece_tables[piece.piece_type][square] * 0.01
        return bonus

    def evaluate(self, board) -> float:
        endgame_weight = self.get_endgame_weight(board)

        def get_positional_bonus(piece, square): # Simple positional bonuses for pieces
            bonus = self.calculate_piece_tables(piece, square, endgame_weight)
            if piece.piece_type == chess.KING:
                bonus += get_king_safety_bonus(piece, square)
            elif piece.piece_type == chess.PAWN:
                bonus += get_pawn_score(piece, square) # To be implemented
            else:
                controlled_squares = board.attacks(square)
                if piece.piece_type == chess.KNIGHT:
                    for attacked_square in controlled_squares:
                        if attacked_square in self.middle_squares and board.piece_at(attacked_square) is not None: # Bonus for attacking/defending pieces in the middle
                            bonus += 0.055
                    bonus += get_knight_outpost_bonus(piece, square) - get_minor_piece_blocking_middle_pawns_penalty(piece, square) # Bonus for controlling squares
                elif piece.piece_type == chess.BISHOP:
                    bonus += (0.025 * len(controlled_squares)) + get_two_bishops_bonus(piece, square) - get_minor_piece_blocking_middle_pawns_penalty(piece, square)
                elif piece.piece_type == chess.ROOK:
                    bonus += get_rook_open_file_bonus(piece, square) + get_connected_rooks_bonus(piece, square)
                elif piece.piece_type == chess.QUEEN:
                    bonus -= get_early_queen_penalty(piece, square)
            return bonus
        
        def get_pawn_score(piece, square):
            score = 0
            pawn_file = chess.square_file(square)
            friendly_pawns = board.pieces(chess.PAWN, piece.color)
            enemy_pawns = board.pieces(chess.PAWN, not piece.color)
            adjacent_files = max(pawn_file - 1, 0), min(pawn_file + 1, 7)
            has_adjacent_pawns = False
            has_adjacent_enemy_pawns = False
            if len(friendly_pawns) > 1:
                for pawn_square in friendly_pawns:
                    if pawn_square != square:
                        if pawn_square == pawn_file:
                            score -= 0.11 # Doubled pawn penalty
                        if chess.square_file(pawn_square) in adjacent_files:
                            has_adjacent_pawns = True
                            rank = chess.square_rank(pawn_square)
                            if rank == chess.square_rank(square): # Bonus for having adjacent pawns on the same rank
                                if piece.color == chess.WHITE:
                                    score += ((rank) ** 2) * 0.01
                                else:
                                    score += ((7 - rank) ** 2) * 0.01
            if not has_adjacent_pawns:
                score -= 0.15 # Isolated pawn penalty
            if len(enemy_pawns) > 0:
                for pawn_square in enemy_pawns:
                    if chess.square_file(pawn_square) in adjacent_files or pawn_square == pawn_file:
                        has_adjacent_enemy_pawns = True
                        break
            if not has_adjacent_enemy_pawns:
                score += 0.475 * endgame_weight # Passed pawn bonus
            for attacked_square in board.attacks(square):
                if attacked_square in self.middle_squares:
                    score += 0.055 # Bonus for controlling the center 
            return score
        
        def get_two_bishops_bonus(piece, square):
            if len(board.pieces(chess.BISHOP, piece.color)) > 1:
                return 0.11 # Bonus for having two bishops
            return 0
            
        def get_knight_outpost_bonus(piece, square):
            friendly_pawns = board.pieces(chess.PAWN, piece.color)
            for pawn_square in friendly_pawns:
                if square in board.attacks(pawn_square):
                    rank = chess.square_rank(square)
                    if piece.color == chess.WHITE:
                        if rank == 4 or rank == 5:
                            return 0.15 * (rank - 3)
                    else:
                        if rank == 2 or rank == 3:
                            return 0.15 * (4 - rank)
            return 0

        def get_minor_piece_blocking_middle_pawns_penalty(piece, square):
            if endgame_weight < 0.3 and (piece.color == chess.WHITE and (square == chess.D3 and board.piece_at(chess.D2) is not None and board.piece_at(chess.D2).piece_type == chess.PAWN) or (square == chess.E3 and board.piece_at(chess.E2) is not None and board.piece_at(chess.E2).piece_type == chess.PAWN)) or (piece.color == chess.BLACK and (square == chess.D6 and board.piece_at(chess.D7) is not None and board.piece_at(chess.D7).piece_type == chess.PAWN) or (square == chess.E6 and board.piece_at(chess.E7) is not None and board.piece_at(chess.E7).piece_type == chess.PAWN)):
                return 0.25 # Penalty for minor piece blocking middle pawn
            return 0
         
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
        
        def get_king_open_penalty(piece, square):
            penalty = 0
            for adjacent_square in board.attacks(square):
                if board.piece_at(adjacent_square) is None:
                    sorted_ray = sorted(chess.SquareSet.ray(square, adjacent_square), key=lambda x: chess.square_distance(x, square))[1:]
                    num_open_squares = 0
                    for ray_square in sorted_ray:
                        square_piece = board.piece_at(ray_square)
                        if square_piece is None or square_piece.color != piece.color:
                            num_open_squares += 1
                        else:
                            break
                    penalty += 0.035 * num_open_squares * (1 - endgame_weight) # Penalty for king being on an open file
            return penalty
        
        def get_king_near_opponent_pieces_penalty(piece, square):
            enemy_major_pieces = board.pieces(chess.ROOK, not piece.color) | board.pieces(chess.QUEEN, not piece.color) | board.pieces(chess.BISHOP, not piece.color) | board.pieces(chess.KNIGHT, not piece.color)
            penalty = 0
            for enemy_piece_square in enemy_major_pieces:
                if chess.square_distance(square, enemy_piece_square) < 3:
                    penalty += 0.05 * (3 - chess.square_distance(square, enemy_piece_square)) * (1 - endgame_weight) * (1 - self.piece_values[piece.piece_type] / 9) # Penalty for king being near opponent pieces
            return penalty
        
        def get_early_queen_penalty(piece, square):
            rank = chess.square_rank(square)
            if endgame_weight < 0.1:
                if piece.color == chess.WHITE:
                    return 0.12 * min(0, rank - 2) # Penalty for queen being out early
                else:
                    return 0.12 * min(0, 5 - rank)
            return 0
        
        def get_rook_open_file_bonus(piece, square):
            rook_rank = chess.square_rank(square)
            if (piece.color == chess.WHITE and rook_rank == 0) or (piece.color == chess.BLACK and rook_rank == 7): # Check if rook is on the 1st or 8th rank
                rook_file = chess.square_file(square)
                for pawn_square in board.pieces(chess.PAWN, piece.color):
                    if chess.square_file(pawn_square) == rook_file:
                        return 0
                return 0.15 * (1 - endgame_weight) # Give bonus for rook being on an open file - not blocked by its own pawns
            return 0
        
        def get_rook_behind_pawn_bonus(piece, square):
            bonus = 0
            controlled_squares = board.attacks(square)
            for controlled_square in controlled_squares:
                controlled_piece = board.piece_at(controlled_square)
                if controlled_piece is not None and controlled_piece.color == piece.color and controlled_piece.piece_type == chess.PAWN:
                    if controlled_piece.piece_type == chess.PAWN and controlled_piece.color == piece.color:
                        bonus += 0.05
        
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
                    value = self.piece_values[piece.piece_type] + get_positional_bonus(piece, square)
                    if piece.color == chess.WHITE:
                        score += value
                    else:
                        score -= value
            return score

    def negamax(self, board, depth, color, alpha, beta):
        hash = self.transposition_table.probe()
        alpha_original = alpha
        if hash is not None and hash[0] >= depth:
            if hash[2] == 0: # Exact
                return hash[1], hash[3]
            elif hash[2] == 1: # Lower bound
                alpha = max(alpha, hash[1])
            elif hash[2] == 2: # Upper bound
                beta = min(beta, hash[1])
            if alpha >= beta: # If the bounds overlap, return the eval
                return hash[1], hash[3]

        if depth == 0 or board.is_game_over(): 
            return color * self.evaluate(board), None # Assume minimax is always called with depth > 0
        
        ordered_moves = self.get_ordered_moves(board)
        best_eval = float('-inf')
        best_move = None
        for move in ordered_moves:
            eval = -self.push_and_pop_move(board, depth - 1, -color, -beta, -alpha, move)
            if eval > best_eval:
                best_eval = eval
                alpha = max(alpha, eval)
                best_move = move
            if beta <= alpha:
                break

        if eval <= alpha_original:
            self.transposition_table.store(depth, eval, 2, best_move) # Upper bound
        elif eval >= beta:
            self.transposition_table.store(depth, eval, 1, best_move)
        else:
            self.transposition_table.store(depth, eval, 0, best_move)
        return best_eval, best_move
        
    def negamax_root(self, board, depth, color):
        self.transposition_table.zobrist_hash(board)
        eval, best_move = self.negamax(board, depth, color, float('-inf'), float('inf'))
        print(f"Best move: {best_move}, Eval: {eval}")
        return str(best_move)
    
    def push_and_pop_move(self, board, depth, color, alpha, beta, move):
        prev_hash = self.transposition_table.get_hash()
        self.transposition_table.move_hash(board, move)
        board.push(move)
        if board.is_castling(move) or board.is_en_passant(move) or move.promotion is not None:
            self.transposition_table.zobrist_hash(board)
        eval, _ = self.negamax(board, depth, color, alpha, beta)
        board.pop()
        self.transposition_table.set_hash(prev_hash)
        return eval
    
    def get_ordered_moves(self, board):
        legal_moves = list(board.legal_moves)
        ordered_moves = sorted(legal_moves, key=lambda move: self.move_priority(board, move), reverse=True)
        hash = self.transposition_table.probe()
        if hash is not None and hash[3] is not None:
            next_move = hash[3]
            ordered_moves.remove(next_move)
            ordered_moves.insert(0, next_move)
        return ordered_moves
        
    def move_priority(self, board, move):
        # Define a priority for each move type: checks > captures > major piece moves  > other moves
        piece = board.piece_at(move.from_square)
        captured_piece = board.piece_at(move.to_square)
        endgame_weight = self.get_endgame_weight(board)
        priority = self.calculate_piece_tables(piece, move.to_square, endgame_weight) - self.calculate_piece_tables(piece, move.from_square, endgame_weight)
        if board.gives_check(move):
            return 5 + priority # Check
        elif board.is_capture(move) and not board.is_en_passant(move) and self.piece_values[captured_piece.piece_type] > self.piece_values[piece.piece_type]:
            return 4 + ((self.piece_values[captured_piece.piece_type] - self.piece_values[piece.piece_type]) * 0.25) + priority # Capture higher value piece
        else:
            if board.is_capture(move) and len(board.attackers(piece.color, move.to_square)) > len(board.attackers(not piece.color, move.to_square)): # attackers > defenders
                    return 3.75 + priority # Capture defended piece
            attackers = board.attackers(not piece.color, move.from_square)
            is_safe_square = self.is_safe_square(board, move.to_square, piece)
            for attacker in attackers:
                attacker_piece = board.piece_at(attacker)
                if self.piece_values[attacker_piece.piece_type] < self.piece_values[piece.piece_type] and is_safe_square: # If the attacker is worth less than the piece being attacked
                    return 3.25 + priority # Escape move
            if is_safe_square:
                new_board = board.copy()
                new_board.push(move) # look into the future
                attacked_squares = new_board.attacks(move.to_square)
                for attacked_square in attacked_squares:
                    attacked_piece = new_board.piece_at(attacked_square)
                    if attacked_piece is not None:
                        if attacked_piece.color == piece.color and not self.is_safe_square(board, attacked_square, attacked_piece):
                            return priority + 3.25 + (1 - self.piece_values[piece.piece_type] / 9) * .1 # Defend unprotected piece
                        elif attacked_piece.color != piece.color and (self.piece_values[attacked_piece.piece_type] > self.piece_values[piece.piece_type] or len(new_board.attackers(not piece.color, attacked_square)) > len(new_board.attackers(piece.color, attacked_square))):
                            return priority + 3.5 + (1 - self.piece_values[piece.piece_type] / 9) * .1 # Attack piece with higher value/undefended pieces
                        elif self.piece_values[piece.piece_type] < 4 and self.starting_squares[move.from_square] == piece and len(attacked_squares.intersection(self.middle_squares)) > 0 and endgame_weight < 0.25: # Develop pieces off starting square
                            return 2 + priority
            if board.is_capture(move):
                return 2.5 + priority
            elif board.is_castling(move):
                return 1.5 + priority
            elif not board.is_zeroing(move):
                return 1 + priority
        return priority
            
    def is_safe_square(self, board, square, piece):
        attackers = board.attackers(not piece.color, square)
        for attacker in attackers:
            attacker_piece = board.piece_at(attacker)
            if self.piece_values[attacker_piece.piece_type] < self.piece_values[piece.piece_type] and not board.is_pinned(not piece.color, attacker): # If the attacker is worth less than the piece being attacked
                return False
        defenders = board.attackers(piece.color, square)
        if len(attackers) > len(defenders):
            return False
        return True
            
    def get_eval(self):
        return (f"Best move: {self.best_move}, Eval: {self.best_eval}")

class TranspositionTable:
    def __init__(self):
        self.table = {}
        self.hash = 0
        self.random_keys = [random.randint(0, 2 ** 64 - 1) for _ in range(773)] # 12 * 64 + 5
        self.piece_keys = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }

    def probe(self):
        return self.table.get(self.hash, None)
    
    def store(self, depth, eval, flag, best_move=None):
        self.table[self.hash] = (depth, eval, flag, best_move)

    def zobrist_hash(self, board):
        self.hash = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                self.hash ^= self.random_keys[64 * self.get_piece_key(piece) + square]  
        if board.turn == chess.BLACK:
            self.hash ^= self.random_keys[64 * 12]
        if board.has_kingside_castling_rights(chess.WHITE):
            self.hash ^= self.random_keys[64 * 12 + 1]
        if board.has_queenside_castling_rights(chess.WHITE):
            self.hash ^= self.random_keys[64 * 12 + 2]
        if board.has_kingside_castling_rights(chess.BLACK):
            self.hash ^= self.random_keys[64 * 12 + 3]
        if board.has_queenside_castling_rights(chess.BLACK):
            self.hash ^= self.random_keys[64 * 12 + 4]
        return self.hash

    def move_hash(self, board, move):
        from_square = move.from_square
        to_square = move.to_square
        piece = board.piece_at(from_square) # Don't need to check if piece is None because move is legal
        captured_piece = board.piece_at(to_square)
        if captured_piece is not None: # If a piece is captured, remove it from the hash
            self.hash ^= self.random_keys[64 * self.get_piece_key(captured_piece) + to_square]
        piece_value = self.get_piece_key(piece)
        self.hash ^= self.random_keys[64 * piece_value + from_square] # Remove piece from old square
        self.hash ^= self.random_keys[64 * piece_value + to_square] # Add piece to new square
        self.hash ^= self.random_keys[64 * 12] # Switch turn
        """
        if board.is_castling(move):
            if piece.color == chess.WHITE:
                self.hash ^= self.random_keys[64 * 12 + 1]
                self.hash ^= self.random_keys[64 * 12 + 2]
                if to_square == chess.G1:
                    self.hash ^= self.random_keys[64 * 3 + chess.F1]
                    self.hash ^= self.random_keys[64 * 3 + chess.H1]
                else:
                    self.hash ^= self.random_keys[64 * 3 + chess.D1]
                    self.hash ^= self.random_keys[64 * 3 + chess.A1]
            else:
                self.hash ^= self.random_keys[64 * 12 + 3]
                self.hash ^= self.random_keys[64 * 12 + 4]
                if to_square == chess.G8:
                    self.hash ^= self.random_keys[64 * 9 + chess.F8]
                    self.hash ^= self.random_keys[64 * 9 + chess.H8]
                else:
                    self.hash ^= self.random_keys[64 * 9 + chess.D8]
                    self.hash ^= self.random_keys[64 * 9 + chess.A8]
        elif board.is_en_passant(move):
            captured_square = chess.square(chess.square_file(to_square), chess.square_rank(from_square))
            captured_piece = board.piece_at(to_square)
            self.hash ^= self.random_keys[64 * self.get_piece_key(captured_piece) + captured_square]
        """
        return self.hash
    
    def get_piece_key(self, piece):
        piece_value = self.piece_keys[piece.piece_type]
        if piece.color == chess.BLACK:
            piece_value += 6
        return piece_value
    
    def set_hash(self, hash):
        self.hash = hash

    def get_hash(self):
        return self.hash

    def clear(self):
        self.table = {}

    def get_items(self):
        return self.table.items()

    

# Add promotion stuff

if __name__ == "__main__":

    chess_bot = Bot()  # you can enter a FEN here, like Bot("...")
    with game_manager():

        playing = True
        player_playing = True
        color = -1

        while playing:
            if chess_bot.board.turn:
                if player_playing:
                    print("Your move:")
                    move = input()
                    while not chess.Move.from_uci(move) in chess_bot.board.legal_moves:
                        print("Invalid move. Try again:")
                        move = input()
                    chess_bot.board.push_san(move)
                else:
                    chess_bot.board.push_san(test_bot.get_move(chess_bot.board))
            else:
                chess_bot.board.push_san(chess_bot.next_move(color, 5))

            print(chess_bot.board, end="\n\n")

            if chess_bot.board.is_game_over():
                if chess_bot.board.is_stalemate():
                    print("Is stalemate")
                elif chess_bot.board.is_insufficient_material():
                    print("Is insufficient material")

                # EX: Outcome(termination=<Termination.CHECKMATE: 1>, winner=True)
                print(chess_bot.board.outcome())

                playing = False

