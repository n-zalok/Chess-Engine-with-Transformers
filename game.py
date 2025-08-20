import time
import torch
import pygame
import chess
import pipe

device = "cpu"

model = pipe.arch.ChessMoveClassifier(pipe.config, device)
state_dict = torch.load('./chess_model.pth')
model.load_state_dict(state_dict)

depth = 4
k = 4

# ==== Pygame chess GUI ====
WIDTH, HEIGHT = 480, 480
SQ_SIZE = WIDTH // 8
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
font = pygame.font.SysFont("Arial", 20, bold=True)
pygame.display.set_caption("Play vs Your Model")

board = chess.Board()


# Store latest evaluation info
latest_move_score = None
latest_eval = None
latest_situations = None
latest_latency = None

def draw_board(screen, board, latest_move_score, latest_eval, latest_situations, latest_latency):
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
        screen.blit(img, (col*SQ_SIZE, row*SQ_SIZE))

    # Draw evaluation text
    if latest_eval is not None:
        eval_text = f"""Score: {latest_move_score:.2f} | Eval: {latest_eval:.2%} |
                        Situations: {latest_situations} | Latency: {latest_latency:.0f}ms"""
        text_surface = font.render(eval_text, True, (255, 0, 0))  # Black text
        screen.blit(text_surface, (10, 10))  # Top-left corner

def main():
    global board, latest_move_score, latest_eval, latest_situations, latest_latency

    running = True
    selected_square = None

    while running:
        draw_board(screen, board, latest_move_score, latest_eval, latest_situations, latest_latency)
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
                file = x // SQ_SIZE
                rank = 7 - (y // SQ_SIZE)
                square = chess.square(file, rank)

                if selected_square is None:
                    selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                        # Model plays after you
                        if not board.is_game_over():
                            start_time = time.perf_counter()
                            m_move, evaluation, situations = pipe.choose_move(board, model, device, depth=depth, k=k)
                            end_time = time.perf_counter()

                            current_value = pipe.evaluate_board.evaluate(board)
                            board.push(m_move[0])

                            latest_move_score = m_move[1]
                            latest_eval = -(evaluation - current_value) / abs(current_value)
                            latest_situations = situations
                            latest_latency = (end_time - start_time) * 1000
                    selected_square = None

    pygame.quit()

if __name__ == "__main__":
    main()