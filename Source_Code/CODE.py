import pygame
import numpy as np
import random

pygame.init()

ROWS = 4
COLS = 4
CELL_SIZE = 120

WIDTH = COLS * CELL_SIZE
HEIGHT = ROWS * CELL_SIZE + 100

START = (0, 0)
GOAL = (3, 3)
OBSTACLES = [(1, 1), (2, 1)]

ACTIONS = [0, 1, 2, 3]

MOVES = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}


def get_state(position):
    return position[0] * COLS + position[1]


def move_agent(position, action):
    row, col = position
    dr, dc = MOVES[action]
    new_position = (row + dr, col + dc)

    if (new_position[0] < 0 or new_position[0] >= ROWS or
            new_position[1] < 0 or new_position[1] >= COLS):
        return position, -10

    if new_position in OBSTACLES:
        return position, -10

    if new_position == GOAL:
        return new_position, 100

    return new_position, -1


# Q-LEARNING TRAINING
q_table = np.zeros((ROWS * COLS, 4))

learning_rate = 0.1
discount_factor = 0.9
epsilon = 1.0
episodes = 3000

print("Training AI Agent...")

for episode in range(episodes):
    position = START

    for step in range(100):
        state = get_state(position)

        if random.random() < epsilon:
            action = random.choice(ACTIONS)
        else:
            action = np.argmax(q_table[state])

        next_position, reward = move_agent(position, action)
        next_state = get_state(next_position)

        if next_position == GOAL:
            target = reward
        else:
            target = reward + discount_factor * np.max(q_table[next_state])

        q_table[state, action] += learning_rate * (
            target - q_table[state, action]
        )

        position = next_position

        if position == GOAL:
            break

    epsilon = max(0.01, epsilon * 0.995)

print("Training Completed!")

# GENERATE THE PATH LEARNED BY THE AI
position = START
learned_path = [position]
visited = {position}

for step in range(30):
    state = get_state(position)
    action = np.argmax(q_table[state])
    next_position, reward = move_agent(position, action)

    if next_position == position or next_position in visited:
        break

    position = next_position
    learned_path.append(position)
    visited.add(position)

    if position == GOAL:
        break

print("Learned Path:", learned_path)
print("Goal Reached:", position == GOAL)

# OBSTACLE COLLISION DEMONSTRATION
demo_position = (0, 1)
demo_result, demo_reward = move_agent(demo_position, 1)

print("Obstacle Collision Demonstration")
print("Initial Position:", demo_position)
print("Action: DOWN")
print("New Position:", demo_result)
print("Reward:", demo_reward)

# PYGAME WINDOW
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning - Grid World")

WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GREEN = (40, 190, 90)
YELLOW = (255, 210, 50)
GRAY = (190, 190, 190)
BLUE = (60, 120, 220)
RED = (220, 50, 50)

font = pygame.font.SysFont(None, 28)
small_font = pygame.font.SysFont(None, 23)

agent_position = START
manual_path = [START]

# Modes: manual, ai_wait, ai_run
mode = "manual"
ai_index = 0
last_ai_move = 0
ai_move_delay = 600

status = "Arrow keys: Manual | A: AI mode"
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:

            # Press A to select autonomous AI mode
            if event.key == pygame.K_a:
                mode = "ai_wait"
                agent_position = START
                manual_path = [START]
                ai_index = 0
                status = "AI mode selected. Press R to start the AI."

            # Press R to start the AI after selecting A
            elif event.key == pygame.K_r and mode == "ai_wait":
                mode = "ai_run"
                agent_position = START
                manual_path = [START]
                ai_index = 0
                last_ai_move = pygame.time.get_ticks()
                status = "AI is solving the grid..."
            # Press R after reaching the goal to reset
            elif event.key == pygame.K_r and mode == "goal_reached":
                mode = "manual"
                agent_position = START
                manual_path = [START]
                ai_index = 0
                status = "Reset to START. Use arrow keys or press A."

            # Manual arrow-key movement
            elif mode == "manual":
                key_to_action = {
                    pygame.K_UP: 0,
                    pygame.K_DOWN: 1,
                    pygame.K_LEFT: 2,
                    pygame.K_RIGHT: 3
                }

                if event.key in key_to_action:
                    new_position, reward = move_agent(
                        agent_position, key_to_action[event.key]
                    )

                    if new_position == agent_position:
                        status = "COLLISION! Reward: -10"
                        print(status)
                    elif new_position == GOAL:
                        agent_position = new_position
                        manual_path.append(agent_position)
                        mode = "goal_reached"
                        status = "GOAL REACHED! Press R to reset."
                        print("GOAL REACHED! Reward: 100")
                        print("Press R to reset and play again.")
                    else:
                        agent_position = new_position
                        manual_path.append(agent_position)
                        status = f"Moved to {agent_position}. Reward: {reward}"
                        print(status)

            # Backspace returns to manual mode and resets the agent
            if event.key == pygame.K_BACKSPACE:
                mode = "manual"
                agent_position = START
                manual_path = [START]
                ai_index = 0
                status = "Manual mode. Agent reset to START."

    # AUTONOMOUS AI MOVEMENT
    if mode == "ai_run":
        current_time = pygame.time.get_ticks()

        if current_time - last_ai_move >= ai_move_delay:
            if ai_index < len(learned_path) - 1:
                ai_index += 1
                agent_position = learned_path[ai_index]
                manual_path.append(agent_position)
                last_ai_move = current_time
                print("AI moved to:", agent_position)

            if agent_position == GOAL:
                mode = "goal_reached"
                status = "AI SOLVED THE GRID! Press R to reset."
                print(status)
    # DRAW BACKGROUND
    screen.fill(WHITE)

    # DRAW GRID
    for row in range(ROWS):
        for col in range(COLS):
            x = col * CELL_SIZE
            y = row * CELL_SIZE
            cell = (row, col)

            if cell in OBSTACLES:
                color = RED
            elif cell == GOAL:
                color = YELLOW
            else:
                color = WHITE

            pygame.draw.rect(
                screen, color,
                (x, y, CELL_SIZE, CELL_SIZE)
            )

            pygame.draw.rect(
                screen, GRAY,
                (x, y, CELL_SIZE, CELL_SIZE), 2
            )

            if cell == START:
                label = font.render("START", True, BLUE)
                screen.blit(label, (x + 30, y + 10))

            if cell == GOAL:
                label = font.render("GOAL", True, BLACK)
                screen.blit(label, (x + 35, y + 10))

    # DRAW VISITED PATH
    for row, col in manual_path:
        center = (
            col * CELL_SIZE + CELL_SIZE // 2,
            row * CELL_SIZE + CELL_SIZE // 2
        )
        pygame.draw.circle(screen, BLUE, center, 9)

    # DRAW AGENT
    agent_center = (
        agent_position[1] * CELL_SIZE + CELL_SIZE // 2,
        agent_position[0] * CELL_SIZE + CELL_SIZE // 2
    )
    pygame.draw.circle(screen, GREEN, agent_center, 25)

    # DISPLAY STATUS AND CONTROLS
    message = font.render(status, True, BLACK)
    screen.blit(message, (10, ROWS * CELL_SIZE + 10))

    if mode == "ai_wait":
        controls_text = "Press R to let the trained AI solve the grid"
    elif mode == "ai_run":
        controls_text = "AI is moving automatically..."
    elif mode == "goal_reached":
        controls_text = "Goal reached! Press R to reset"
    else:
        controls_text = "Arrows: Manual | A: AI mode | Backspace: Reset"

    controls = small_font.render(controls_text, True, BLACK)
    screen.blit(controls, (10, ROWS * CELL_SIZE + 48))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()
print("Application Closed.")
