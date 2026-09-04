import abc
from typing import Iterable

import chess


class Representation(abc.ABC):
    """
    Abstract interface for translating a chess game into a sequence of discrete events.
    """

    @abc.abstractmethod
    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        """
        Convert an iterable of chess moves into a list of string tokens.
        """
        pass


class UCIRepresentation(Representation):
    """
    Translates a sequence of moves into Universal Chess Interface (UCI) strings.
    """

    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        return [move.uci() for move in moves]


class SANRepresentation(Representation):
    """
    Translates a sequence of moves into Standard Algebraic Notation (SAN) strings.
    """

    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        board = chess.Board()
        san_moves = []
        for move in moves:
            san_moves.append(board.san(move))
            board.push(move)
        return san_moves
