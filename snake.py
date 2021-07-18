# ** Credit where credit due: barebones snake based off https://github.com/TheAILearner/Snake-Game-using-Python-Curses/blob/master/snake_game_using_curses.py

def playSnake():
    import curses
    import curses.ascii
    import random
    import time

    #initialize screen (different from game window)
    sc = curses.initscr()
    #recieve dimensions of allocated screen and set width and height of game window to such dimensions
    winHeight, winWidth = sc.getmaxyx()
    win = curses.newwin(winHeight, winWidth, 0, 0)
    curses.start_color()
    curses.use_default_colors()

    win.keypad(1)
    curses.curs_set(0) # make cursor invisible

    # Initial Snake and food position, y x
    snake_head = [10,15]
    snake_position = [[15,10],[14,10],[13,10]]
    food_position = [10,45]
    score = 0

    # display food
    win.addch(food_position[0], food_position[1], curses.ACS_DIAMOND)

    prev_button_direction = 1
    button_direction = 1
    key = curses.KEY_RIGHT

    speed = 60 # timeout time of window. lower timeout time will mean faster game

    def hitsFood(score):
        food_position = [random.randint(1,winHeight-2),random.randint(1,winWidth-2)]
        score += 1
        return food_position, score

    def hitsWall(snake_head):
        if snake_head[0]>=winHeight-1 or snake_head[0]<=0 or snake_head[1]>=winWidth-1 or snake_head[1]<=0 :
            return True
        else:
            return False

    def hitsSelf(snake_position):
        snake_head = snake_position[0]
        if snake_head in snake_position[1:]:
            return True
        else:
            return False


    while True:
        """
        border params: left side, right side, top side, bottom side, top left, top right, bottom left, bottom right
        pass 0 for default
        """
        win.border(0)
        """
        makes the window wait 100 miliseconds for an input,
        allowing the player to start in a different direction if they wanted to
        """
        win.timeout(speed)

        next_key = win.getch()

        if next_key == -1:
            key = key
        else:
            key = next_key

        # 0-Left, 1-Right, 3-Up, 2-Down
        if key == curses.KEY_LEFT and prev_button_direction != 1:
            button_direction = 0
        elif key == curses.KEY_RIGHT and prev_button_direction != 0:
            button_direction = 1
        elif key == curses.KEY_UP and prev_button_direction != 2:
            button_direction = 3
        elif key == curses.KEY_DOWN and prev_button_direction != 3:
            button_direction = 2
        else:
            print(" ") # the one solution i have found to get rid of the incorrect keys getting appended to the score (they disappear after short delay)

        prev_button_direction = button_direction

        # Change the head position based on the button direction
        if button_direction == 1:
            snake_head[1] += 1
        elif button_direction == 0:
            snake_head[1] -= 1
        elif button_direction == 2:
            snake_head[0] += 1
        elif button_direction == 3:
            snake_head[0] -= 1
        else:
            pass

        # Increase Snake length on eating food
        if snake_head == food_position:
            food_position, score = hitsFood(score)
            snake_position.insert(0, list(snake_head))
            win.addch(food_position[0], food_position[1], curses.ACS_DIAMOND)

        else:
            snake_position.insert(0, list(snake_head))
            last = snake_position.pop()
            win.addch(last[0], last[1], ' ')

        # display snake, arrows that face in current bearing of snake
        if button_direction == 1:
            win.addch(snake_position[0][0], snake_position[0][1], '■')
        elif button_direction == 0:
            win.addch(snake_position[0][0], snake_position[0][1], '■')
        elif button_direction == 3:
            win.addch(snake_position[0][0], snake_position[0][1], '█')
        elif button_direction == 2:
            win.addch(snake_position[0][0], snake_position[0][1], '█')


        win.addstr(1, 1, "Score: {}".format(score), curses.A_UNDERLINE)

        # On collision kill the snake
        if hitsWall(snake_head) or hitsSelf(snake_position):
            break


    sc.addstr(10, 30, f"FINAL SCORE: {score}")
    sc.addstr(12, 30, "shutting down in a couple seconds...")
    sc.refresh()
    time.sleep(2.5)
    curses.endwin()

