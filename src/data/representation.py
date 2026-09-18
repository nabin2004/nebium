"""
Chess move sequence representation strategies for Nebium.

Provides abstractions for converting game trajectories into Universal Chess
Interface (UCI) coordinate strings or Standard Algebraic Notation (SAN).
"""

import abc
from typing import Iterable

import chess


class Representation(abc.ABC):
    """
    Abstract base interface for translating a chess game into discrete token strings.
    """

    @abc.abstractmethod
    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        """
        Converts an iterable of chess moves into a list of string tokens.

        Args:
            moves: Iterable of python-chess Move objects.

        Returns:
            List of string move tokens.
        """
        pass


class UCIRepresentation(Representation):
    """
    Translates a sequence of moves into Universal Chess Interface (UCI) strings (e.g. 'e2e4').
    """

    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        """
        Encodes moves as coordinate UCI notation.

        Args:
            moves: Sequence of chess moves.

        Returns:
            List of 4-5 character UCI strings.
        """
        return [move.uci() for move in moves]


class SANRepresentation(Representation):
    """
    Translates a sequence of moves into Standard Algebraic Notation (SAN) strings (e.g. 'Nf3').
    """

    def encode_game(self, moves: Iterable[chess.Move]) -> list[str]:
        """
        Encodes moves as SAN strings by statefully replaying on a board.

        Args:
            moves: Sequence of chess moves.

        Returns:
            List of context-dependent SAN move strings.
        """
        board = chess.Board()
        san_moves = []
        for move in moves:
            san_moves.append(board.san(move))
            board.push(move)
        return san_moves

