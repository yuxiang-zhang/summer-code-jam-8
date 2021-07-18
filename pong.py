def playPong():
    import curses
    import random
    import time

    ESCAPE = 27
    PADDLE_LENGTH = 6
    ROWS = 23
    COLUMNS = 79

    def get_new_ball_coord():
        y = random.randint(4, ROWS - 4)
        x = COLUMNS // 2 + 1
        return y, x

    def draw_paddle(window, y, x, number=0):
        for i in range(PADDLE_LENGTH):
            window.addch(y + i, x, ' ', curses.color_pair(number))

    def has_collided(ball_y, ball_x, paddle_y, paddle_x):
        for i in range(PADDLE_LENGTH):
            if ball_y == paddle_y + i and ball_x == paddle_x + 1:
                return True

    def display_winner(window, player):
        window.addstr(ROWS // 2 - 2, (COLUMNS // 2 + 1) - len('%s wins!' % player) // 2,
                      '%s wins!' % player, curses.color_pair(3) | curses.A_BOLD)
        window.addstr(ROWS // 2 - 1, (COLUMNS // 2 + 1) - len('Game Over') // 2,
                      'Game Over', curses.color_pair(3) | curses.A_BOLD)
        window.refresh()
        time.sleep(1)
        window.nodelay(0)
        return window.getch()

    def randomize_speed():
        return [random.choice((-1, 1)), random.choice((-1, 1))]

    def main(stdscr):
        # Create a new Curses window
        win = curses.newwin(ROWS, COLUMNS)
        curses.noecho()
        curses.cbreak()
        win.keypad(1)
        curses.curs_set(0)

        # Initialize color pairs
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_GREEN)
        curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_WHITE)
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_CYAN)

        # Game variables
        speed = randomize_speed()
        player = computer = 0
        player_y = ROWS // 3
        player_x = 2
        computer_y = ROWS // 3
        computer_x = COLUMNS - 3
        ball_y, ball_x = get_new_ball_coord()

        # Display welcome screen and wait for key press
        win.clear()
        win.border()
        win.addstr(ROWS // 2 - 2, (COLUMNS // 2 + 1) - len('XPONG') // 2,
                   'XPONG', curses.color_pair(3) | curses.A_BOLD)
        win.refresh()
        q = win.getch()
        win.clear()
        win.border()
        win.nodelay(1)

        # Game loop
        while True:
            # Delete old ball and paddles
            draw_paddle(win, player_y, player_x)
            draw_paddle(win, computer_y, computer_x)
            win.addch(ball_y, ball_x, ' ')

            # Get user input and check if the user has pressed 'q' to quit
            # otherwise move the player
            key = win.getch()
            if key == ESCAPE or key == ord('q'):
                break
            elif key == curses.KEY_DOWN \
                    and player_y < ROWS - (PADDLE_LENGTH + 1):
                player_y += 1
            elif key == curses.KEY_UP and player_y > 1:
                player_y -= 1

            # Move the computer
            if computer_y - random.randint(0, 3) < ball_y \
                    and computer_y < ROWS - (PADDLE_LENGTH + 1):
                computer_y += 1
            if computer_y + PADDLE_LENGTH > ball_y \
                    and computer_y > 1:
                computer_y -= 1

            # Move the ball
            ball_y += speed[0]
            ball_x += speed[1]

            # Check ball collision
            if has_collided(ball_y, ball_x, player_y, player_x):
                speed[1] = -speed[1]
            if has_collided(ball_y, ball_x, computer_y, computer_x - 1):
                speed[1] = -speed[1]

            # Checks if the ball is going off win
            if ball_y <= 1 or ball_y >= ROWS - 2:
                speed[0] = -speed[0]
            if ball_x <= 1:
                speed[0] = random.choice((-1, 1))
                ball_y, ball_x = get_new_ball_coord()
                computer += 1
            elif ball_x >= COLUMNS - 2:
                speed[0] = random.choice((-1, 1))
                ball_y, ball_x = get_new_ball_coord()
                player += 1

            # Draw the net, paddles and ball
            for i in range(1, ROWS, 2):
                win.addch(i, 40, '|')
            draw_paddle(win, player_y, player_x, 1)
            draw_paddle(win, computer_y, computer_x, 1)
            win.addstr(1, 10, 'Player: %s' % player, curses.color_pair(3))
            win.addstr(1, COLUMNS
                       - len('Computer: %s' % computer) - 11, 'Computer: %s' % computer,
                       curses.color_pair(3))

            # Check if there is a winner and display it on screen
            if player == 11 or computer == 11:
                if player > computer:
                    key = display_winner(win, 'Player')
                else:
                    key = display_winner(win, 'Computer')
                if key == ESCAPE or key == ord('q'):
                    break
                # Reset game variables
                speed = randomize_speed()
                player = computer = 0
                player_y = ROWS // 3
                player_x = 2
                computer_y = ROWS // 3
                computer_x = COLUMNS - 3
                ball_y, ball_x = get_new_ball_coord()
                win.clear()
                win.border()
                win.nodelay(1)
            win.addch(ball_y, ball_x, 'x', curses.color_pair(2))
            win.refresh()
            time.sleep(0.05)

        # Clean up before exiting
        curses.nocbreak()
        win.keypad(0)
        curses.echo()
        curses.endwin()

    curses.wrapper(main)
