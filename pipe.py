import torch
import chess
import architecture as arch
import evaluate_board

tokenizer = arch.tokenizer
label_to_id = arch.label_to_id
id_to_label = arch.id_to_label
config = arch.Config()
max_length = config.max_position_embeddings
counter = 0

def prepare_input(board):
    pos = '[CLS]'
    if board.turn:
        pos = pos + ' ' + 'WHITE'

        if chess.Move.from_uci("e1g1") in board.legal_moves:
            pos = pos + ' ' + 'KINGSIDE_CASTLE'
        else:
            pos = pos + ' ' + 'NO_KINGSIDE_CASTLE'
        if chess.Move.from_uci("e1c1") in board.legal_moves:
            pos = pos + ' ' + 'QUEENSIDE_CASTLE'
        else:
            pos = pos + ' ' + 'NO_QUEENSIDE_CASTLE'

    else:
        pos = pos + ' ' + 'BLACK'

        if chess.Move.from_uci("e8g8") in board.legal_moves:
            pos = pos + ' ' + 'KINGSIDE_CASTLE'
        else:
            pos = pos + ' ' + 'NO_KINGSIDE_CASTLE'
        if chess.Move.from_uci("e8c8") in board.legal_moves:
            pos = pos + ' ' + 'QUEENSIDE_CASTLE'
        else:
            pos = pos + ' ' + 'NO_QUEENSIDE_CASTLE'
        
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            pos = pos + ' ' + piece.symbol()
        else:
            pos = pos + ' ' + 'EMPTY'
    
    input_ids = [tokenizer[word] for word in pos.split()]
    input_ids = input_ids[:max_length]
    
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

def topk_moves(board, output, k=1):
    start_probs = torch.softmax(output[0], dim=-1)
    end_probs = torch.softmax(output[1], dim=-1)
    
    start_topk = torch.topk(start_probs, k=start_probs.size(1), dim=-1)
    end_topk = torch.topk(end_probs, k=end_probs.size(1), dim=-1)

    moves = {}
    for i in range(start_topk.indices.size(1)):
        start_idx = start_topk.indices[0][i].item()

        for j in range(end_topk.indices.size(1)):
            end_idx = end_topk.indices[0][j].item()

            if start_idx >= 64 or end_idx >= 64:
                if start_idx == end_idx:
                    if start_idx == 64:
                        move = chess.Move.from_uci("e1g1" if board.turn else "e8g8")
                        if move in board.legal_moves:
                            moves[move] = (start_topk.values[0][i].item() + end_topk.values[0][i].item()) / 2
                        else:
                            continue
                    else:
                        move = chess.Move.from_uci("e1c1" if board.turn else "e8c8")
                        if move in board.legal_moves:
                            moves[move] = (start_topk.values[0][i].item() + end_topk.values[0][i].item()) / 2
                        else:
                            continue
                else:
                    continue
            
            else:
                if start_idx == end_idx:
                    continue
                else:
                    move = chess.Move.from_uci(f"{id_to_label[start_idx]}{id_to_label[end_idx]}")
                    if move in board.legal_moves:
                        moves[move] = (start_topk.values[0][i].item() + end_topk.values[0][i].item()) / 2
                    else:
                        continue
    top_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)[:k]
            
    return top_moves

def search(board, model, device, is_maximizing, depth, alpha, beta, k):
    global counter

    if depth == 0 or board.is_game_over():
        return evaluate_board.evaluate(board)

    input = prepare_input(board).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_k_moves = topk_moves(board, output, k)

    if is_maximizing: # maximizing player
        value = -float("inf")
        for move in top_k_moves:
            counter += 1

            board.push(move[0])
            value = max(value, search(board, model, device, False, depth - 1, alpha, beta, k))
            board.pop()

            alpha = max(alpha, value)
            if beta <= alpha:  # prune
                break
        return value

    else:  # minimizing player
        value = float("inf")
        for move in top_k_moves:
            counter += 1
            
            board.push(move[0])
            value = min(value, search(board, model, device, True, depth - 1, alpha, beta, k))
            board.pop()

            beta = min(beta, value)
            if beta <= alpha:  # prune
                break
        return value

def choose_move(board, model, device, depth=1, k=1):
    global counter

    root_player = board.turn
    best_val = -float("inf") if board.turn == chess.WHITE else float("inf")
    best_move = None

    input = prepare_input(board).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_k_moves = topk_moves(board, output, k)

    alpha, beta = -float("inf"), float("inf")

    for move in top_k_moves:
        counter += 1

        board.push(move[0])
        value = search(board, model, device, board.turn, depth - 1, alpha, beta, k)
        board.pop()

        if root_player:  # we are white at root
            if value > best_val:
                best_val = value
                best_move = move
            alpha = max(alpha, best_val)
        else:  # we are black at root
            if value < best_val:
                best_val = value
                best_move = move
            beta = min(beta, best_val)

    situations = counter
    counter = 0

    if best_move is None:
        best_move = next(iter(board.legal_moves))

    return best_move, best_val, situations

