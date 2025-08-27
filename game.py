import time
import torch
import pygame
import chess
import pipe

# Load the model
device = "cpu"

model = pipe.arch.ChessMoveClassifier(pipe.config, device)
state_dict = torch.load('./artifact/chess_model.pth', map_location=torch.device("cpu"))
model.load_state_dict(state_dict)

k = 1

# ==== Pygame chess GUI ====
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
GREEN = (0, 255, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont(None, 30, bold=True)
pygame.display.set_caption("Play vs Your Model")

# Initiate the Board
board = chess.Board()

# Store latest evaluation info
latest_move_score = None
latest_latency = None


def draw_board(screen, board, latest_move_score, latest_latency, legal_moves=None):
    colors = [(240, 217, 181), (181, 136, 99)]
    SQ_SIZE = 480 // 8

    # Draw squares
    for rank in range(8):
        for file in range(8):
            color = colors[(rank + file) % 2]
            pygame.draw.rect(
                screen, color, pygame.Rect(file*SQ_SIZE, rank*SQ_SIZE, SQ_SIZE, SQ_SIZE)
            )

    # Draw pieces
    pieces = board.piece_map()
    for square, piece in pieces.items():
        col = chess.square_file(square)
        row = 7 - chess.square_rank(square)
        img = pygame.image.load(f"pieces/{piece.symbol()}.png")
        img = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
        if player_is_white:
            screen.blit(img, (col*SQ_SIZE, row*SQ_SIZE))
        else:
            screen.blit(img, (420 - (col*SQ_SIZE), 420 - (row*SQ_SIZE)))

    # Draw legal move hints
    if legal_moves:
        for move in legal_moves:
            col = chess.square_file(move.to_square)
            row = 7 - chess.square_rank(move.to_square)
            if player_is_white:
                center = (col*SQ_SIZE + SQ_SIZE//2, row*SQ_SIZE + SQ_SIZE//2)
            else:
                center = (480 - (col*SQ_SIZE + SQ_SIZE//2), 480 - (row*SQ_SIZE + SQ_SIZE//2))
            pygame.draw.circle(screen, GREEN, center, 8)

    # Draw evaluation text
    if latest_latency is not None:
        eval_text = f"Score: {latest_move_score:.2f} | Latency: {latest_latency:.0f}ms"
        text_surface = font.render(eval_text, True, (255, 0, 0))  # Red text
        screen.blit(text_surface, (10, 10))  # Top-left corner

def choose_side():
    choosing = True

    pygame.draw.rect(screen, (255, 255, 255), pygame.Rect(0, 0, WIDTH, HEIGHT / 2))
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(0, HEIGHT / 2, WIDTH, HEIGHT / 2))
    pygame.display.flip()

    while choosing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                _,y = event.pos
                choosing = False
                break
    
    return True if y < (HEIGHT / 2) else False
def promotion_menu(player_is_white):
    menu_running = True

    # Define promotion pieces
    options = {"q": chess.QUEEN, "r": chess.ROOK, "b": chess.BISHOP, "n": chess.KNIGHT}
    rects = []


    # Draw menu background
    menu_height = SQ_SIZE
    pygame.draw.rect(screen, (200, 200, 200), (0, 0, WIDTH, menu_height))


    # Draw piece options
    for i, piece_type in enumerate(options.keys()):
        if player_is_white:
            img = pygame.image.load(f"pieces/{piece_type.upper()}.png")
        else:
            img = pygame.image.load(f"pieces/{piece_type}.png")
        img = pygame.transform.scale(img, (SQ_SIZE, SQ_SIZE))
        x = i * SQ_SIZE
        rect = screen.blit(img, (x, 0))
        rects.append((rect, options[piece_type]))


    pygame.display.flip()

    while menu_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                for rect, piece_type in rects:
                    if rect.collidepoint(pos):
                        promotion_choice = piece_type
                        menu_running = False
                break
    
    return promotion_choice


# Choosing side
player_is_white = choose_side()


def main():
    global board, latest_move_score, latest_latency

    running = True
    selected_square = None
    legal_moves = None


    # If player chose black, model plays first
    if not player_is_white:
        top_k_moves = pipe.choose_move(board, model, device, k=k)
        best_move = top_k_moves[0][0]
        board.push(best_move)

    while running:
        draw_board(screen, board, latest_move_score, latest_latency, legal_moves)
        pygame.display.flip()

        if board.is_game_over():
            print("Game Over:", board.result())
            pygame.time.wait(3000)
            running = False
            continue

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse click
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if player_is_white:
                    pass
                else:
                    x = abs(x - 480)
                    y = abs(y - 480)
                
                file = x // SQ_SIZE
                rank = 7 - (y // SQ_SIZE)
                square = chess.square(file, rank)

                # If it is the first click get the clicked square
                if selected_square is None:
                    piece = board.piece_at(square)
                    if piece and ((piece.color == chess.WHITE and player_is_white) or (piece.color == chess.BLACK and not player_is_white)):
                        selected_square = square
                        legal_moves = [m for m in board.legal_moves if m.from_square == square]
                # If it is the second click 
                else:
                    # If the click is on a non-valid move square get the clicked square
                    piece = board.piece_at(square)
                    if piece and ((piece.color == chess.WHITE and player_is_white) or (piece.color == chess.BLACK and not player_is_white)):
                        selected_square = square
                        legal_moves = [m for m in board.legal_moves if m.from_square == square]
                    # Else make the move
                    else:
                        # Handle pawn promotion
                        if board.piece_at(selected_square).piece_type == chess.PAWN and (chess.square_rank(square) in [0, 7]):
                            promotion_piece = promotion_menu(player_is_white)
                            move = chess.Move(selected_square, square, promotion=promotion_piece)
                        else:
                            move = chess.Move(selected_square, square)


                        if move in board.legal_moves:
                            board.push(move)
                            legal_moves = None

                            # Model plays after you
                            if not board.is_game_over():
                                start_time = time.perf_counter()
                                top_k_moves = pipe.choose_move(board, model, device, k=k)
                                end_time = time.perf_counter()
                                
                                best_move = top_k_moves[0][0]
                                score = top_k_moves[0][1]

                                board.push(best_move)

                                latest_move_score = score
                                latest_latency = (end_time - start_time) * 1000
                        
                        else:
                            selected_square = None
                            legal_moves = None
                        
                        selected_square = None

    pygame.quit()


if __name__ == "__main__":
    main()
