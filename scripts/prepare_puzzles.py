import csv
import io
import json
import random
import time
from pathlib import Path
import urllib.request
import zstandard as zstd
import requests
import chess.pgn

def fetch_puzzle_chunk(num_lines=2000):
    url = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    req = urllib.request.Request(url, headers={'User-Agent': 'Nebium-Eval-Bot'})
    with urllib.request.urlopen(req) as response:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(response) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            lines = []
            for i in range(num_lines):
                line = text_stream.readline()
                if not line:
                    break
                lines.append(line)
            return lines

def prepare_puzzles(output_path, num_per_bin=50):
    print("Fetching puzzle chunk from Lichess...")
    lines = fetch_puzzle_chunk(2000)
    
    # Bins: <1500, 1500-2000, 2000+
    bins = {'<1500': [], '1500-2000': [], '2000+': []}
    
    reader = csv.reader(lines)
    for row in reader:
        if len(row) < 9:
            continue
        puzzle_id, fen, moves_str, rating, rd, pop, nb, themes, game_url, tags = row[:10]
        try:
            rating = int(rating)
        except ValueError:
            continue
            
        if rating < 1500:
            b = '<1500'
        elif rating < 2000:
            b = '1500-2000'
        else:
            b = '2000+'
            
        bins[b].append({
            'id': puzzle_id,
            'fen': fen,
            'moves': moves_str.split(),
            'rating': rating,
            'game_url': game_url
        })
        
    # Sample from bins
    selected = []
    for b, items in bins.items():
        if len(items) > num_per_bin:
            selected.extend(random.sample(items, num_per_bin))
        else:
            selected.extend(items)
            
    print(f"Selected {len(selected)} puzzles. Fetching game PGNs...")
    
    results = []
    session = requests.Session()
    session.headers.update({'User-Agent': 'Nebium-Eval-Bot'})
    
    for i, puz in enumerate(selected):
        game_id = puz['game_url'].split('/')[-1]
        # Puzzles are often from variations or blitz games. 
        # The game id in the url usually has 8 chars.
        if len(game_id) > 8:
            game_id = game_id[:8]
            
        pgn_url = f"https://lichess.org/game/export/{game_id}?evals=false&clocks=false&literate=false"
        try:
            resp = session.get(pgn_url)
            if resp.status_code == 429:
                print("Rate limited by Lichess. Waiting 60 seconds...")
                time.sleep(60)
                resp = session.get(pgn_url)
            
            if resp.status_code != 200:
                print(f"Failed to fetch {pgn_url}: HTTP {resp.status_code}")
                continue
                
            pgn_text = resp.text
            pgn_io = io.StringIO(pgn_text)
            game = chess.pgn.read_game(pgn_io)
            
            if game is None:
                continue
                
            board = game.board()
            history = []
            
            # Reconstruct the game and find the FEN
            found = False
            for move in game.mainline_moves():
                history.append(move.uci())
                board.push(move)
                
                # Check if FEN matches
                if board.fen() == puz['fen'] or board.board_fen() == puz['fen'].split()[0]:
                    found = True
                    break
                    
            if not found:
                # Some puzzles start from a position in the middle that isn't exact due to castling rights etc,
                # but board_fen() usually matches.
                continue
                
            # The puzzle 'moves' contains the opponent's blunder first, then the solution
            opponent_move = puz['moves'][0]
            solution_move = puz['moves'][1]
            
            # The prompt is the history up to the FEN + the opponent's move
            prompt_moves = " ".join(history + [opponent_move])
            
            results.append({
                'id': puz['id'],
                'rating': puz['rating'],
                'prompt': prompt_moves,
                'solution': solution_move,
                'full_solution': puz['moves'][1:]
            })
            
            print(f"Processed {i+1}/{len(selected)}: {puz['id']} (Rating {puz['rating']})")
            time.sleep(0.5) # Be nice to Lichess API
            
        except Exception as e:
            print(f"Error on {puz['id']}: {e}")
            
    # Save to jsonl
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    print(f"Saved {len(results)} puzzles to {out_path}")

if __name__ == "__main__":
    prepare_puzzles("data/fixtures/puzzles.jsonl", num_per_bin=30)
