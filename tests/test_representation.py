import chess

from src.data.representation import SANRepresentation, UCIRepresentation


def test_uci_representation():
    board = chess.Board()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    rep = UCIRepresentation()
    encoded = rep.encode_game(moves)
    assert encoded == ["e2e4", "e7e5", "g1f3"]


def test_san_representation():
    board = chess.Board()
    moves = [
        chess.Move.from_uci("e2e4"),
        chess.Move.from_uci("e7e5"),
        chess.Move.from_uci("g1f3"),
    ]
    rep = SANRepresentation()
    encoded = rep.encode_game(moves)
    assert encoded == ["e4", "e5", "Nf3"]
