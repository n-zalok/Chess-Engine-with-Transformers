import torch
import chess
import architecture as arch

tokenizer = arch.tokenizer
label_to_id = arch.label_to_id
id_to_label = arch.id_to_label
config = arch.Config()
max_length = config.max_position_embeddings

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
                    if board.piece_at(chess.parse_square(id_to_label[start_idx])) is not None:
                        if board.piece_at(chess.parse_square(id_to_label[start_idx])).piece_type == chess.PAWN and (chess.square_rank(chess.parse_square(id_to_label[end_idx])) in [0, 7]):
                            move = chess.Move.from_uci(f"{id_to_label[start_idx]}{id_to_label[end_idx]}q")
                        else:
                            move = chess.Move.from_uci(f"{id_to_label[start_idx]}{id_to_label[end_idx]}")

                        if move in board.legal_moves:
                            moves[move] = (start_topk.values[0][i].item() + end_topk.values[0][i].item()) / 2
                        else:
                            continue
                    else:
                        continue
    top_moves = sorted(moves.items(), key=lambda x: x[1], reverse=True)[:k]
            
    return top_moves


def choose_move(board, model, device, k=1):

    input = prepare_input(board).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input)
    top_k_moves = topk_moves(board, output, k)

    return top_k_moves

