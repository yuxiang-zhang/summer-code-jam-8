from itertools import cycle

import numpy as np

from nurses import ScreenManager, Widget, colors
from nurses.keys import DOWN, LEFT, RIGHT, TAB, UP
from nurses.widgets import ArrayWin

# Keybindings
SPACE_KEY, RESET_KEY = ord(' '), ord('r')
FORFEIT_KEY = ord('g')
FLAG_KEY = ord('f')

# Miscs
OFFSET_TOP, OFFSET_LEFT = 5, 25
DELTA = .1

# Symbols
COVERED_SYMBOL = '❑'
EMPTY_SYMBOL = '⯀'
MINE_SYMBOL = '☠'
FLAG_SYMBOL = '☢'
HAPPYFACE_SYMBOL = '☺'
SADFACE_SYMBOL = '☹'
CURSOR_SYMBOL = 'ᐁ'
BOXEDCHECK_SYMBOL = '☑'
BOXEDCROSS_SYMBOL = '☒'

# States
COVERED_STATE = 0
UNCOVERED_STATE = 1
FLAGGED_STATE = 2


class Cursor(Widget):
    """Movable cursor to point to a land location."""

    move_up = UP
    move_down = DOWN
    move_left = LEFT
    move_right = RIGHT

    lr_step = 1
    ud_step = 1

    wrap_height = None
    wrap_width = None

    offset_top = OFFSET_TOP
    offset_left = OFFSET_LEFT

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_wrapping(self, height: int, width: int) -> None:
        """Set Cursor movement wrapping height and width."""
        self.wrap_height, self.wrap_width = height, width

    def on_press(self, key: int) -> bool:
        """Handle keys for Cursor movement."""
        top, left = self.top, self.left
        height, width = self.height, self.width

        if key == self.move_up:
            if top > 0:
                self.top -= self.ud_step
        elif key == self.move_down:
            if top + height < self.parent.height:
                self.top += self.ud_step
        elif key == self.move_left:
            if left > 0:
                self.left -= self.lr_step
        elif key == self.move_right:
            if left + width < self.parent.width:
                self.left += self.lr_step
        else:
            return super().on_press(key)

        if self.wrap_height:
            self.top = (self.top - self.offset_top) % self.wrap_height + self.offset_top
        if self.wrap_width:
            self.left = (self.left - self.offset_left) % self.wrap_width + self.offset_left
        return True


class Lawn(ArrayWin):
    """MineSweeper Game Board."""

    def __init__(self, rows: int, cols: int, num_mines: int, *args, **kwargs) -> None:

        # Initialize display from ArrayWin
        super().__init__(OFFSET_TOP, OFFSET_LEFT, rows, cols, *args, **kwargs)

        self._gsm = None
        self._marching_task = None

        self.difficulties = cycle([(9, 11, 10), (16, 16, 40), (16, 30, 99)])
        self.resize(rows, cols, num_mines)

    def schedule_marching(self, delay: int = .3) -> None:
        """Use ScreenManager handle to animate scoreboard text."""

        def marching_scoreboard() -> None:
            """Animate scoreboard with marching texts."""
            head = self._scoreboard[1, 0]
            tail = self._scoreboard[1, 1:]
            self._scoreboard[1, :-1] = tail
            self._scoreboard[1, -1] = head

        if self._gsm:
            self._marching_task = self._gsm.schedule(marching_scoreboard, delay=delay, n=120)

    def init_lawn(self) -> None:
        """Initialize game board for the next game."""
        # Reset display
        self[:, :] = COVERED_SYMBOL

        # Reset colors
        colors_copy = np.full(self._shape, self.color)
        self.colors = colors_copy

        # Display the game board
        self.revealed = False

        # Clear the game board
        self._state_map[:, :] = COVERED_STATE

        # Randomize mine locations
        self._mine_map = self._mine_map.reshape(-1)
        np.random.shuffle(self._mine_map)
        self._mine_map = self._mine_map.reshape(self._shape)

        self._solution_map = self.build_solution()

        self.unload_screen()
        self.load_screen()

        # Unset timer
        self.timer = None

    def load_screen(self, gsm: ScreenManager = None) -> None:
        """Add stuff to the screen."""
        # Game ScreenManager, useful for scheduling animation task
        if gsm:
            self._gsm = gsm
        if self._gsm:
            # Scoreboard display scores and shoutout banner
            self._scoreboard = self._gsm.root.new_widget(OFFSET_TOP + self._shape[0] + 2, OFFSET_LEFT,
                                                         height=2, width=self._shape[1],
                                                         color=colors.RED_ON_BLACK, create_with=ArrayWin)

            instructions = [
                '↹: change difficulty',
                'r: reset game',
                'g: give up game',
                '␣: uncover location',
                'f: flag mine',
                'arrows: move pointer',
                'esc: leave game'
            ]

            # List instructions on the side
            self._instructions = self._gsm.root.new_widget(OFFSET_TOP, OFFSET_LEFT + self._shape[1] + 2,
                                                           height=len(instructions), width=20,
                                                           color=colors.YELLOW_ON_BLACK, create_with=ArrayWin)

            # Erase shoutout text
            self._scoreboard[:, :] = ' '
            self._scoreboard[1, :8] = "Welcome!"

            # Set instructions
            for i, line in enumerate(instructions):
                self._instructions[i, :] = line.ljust(text_len, ' ')

            self.schedule_marching()

    def unload_screen(self) -> None:
        """Remove stuff from the screen."""
        if self._gsm and self._scoreboard:
            self._gsm.root.remove_widget(self._scoreboard)
            self._scoreboard = None

        if self._gsm and self._instructions:
            self._gsm.root.remove_widget(self._instructions)
            self._instructions = None

    def build_solution(self) -> np.ndarray:
        """Count mines near each land in adjacent lands."""
        rows, cols = self._shape
        solution_map = np.zeros((rows, cols)).astype(int)

        for r in range(rows):
            for c in range(cols):
                solution_map[r, c] = self._mine_map[
                    max(0, r - 1):min(r + 2, rows),
                    max(0, c - 1):min(c + 2, cols)
                ].sum()

        solution_map = solution_map.astype(str)
        solution_map[solution_map == '0'] = EMPTY_SYMBOL
        return solution_map

    def reveal_mines(self) -> None:
        """Reveal all mine locations."""
        if not self.revealed:
            self.revealed = True
            self[:, :] = np.where(self._mine_map, MINE_SYMBOL, self[:, :])

    def resize(self, rows: int, cols: int, num_mines: int) -> None:
        """Resize game board."""
        self._shape = rows, cols
        self.height, self.width = rows, cols
        self._num_mines = num_mines

        # Game board records the state of each cell
        self._state_map = np.zeros((rows, cols)).astype(int)

        # Initialize with given mine amount
        self._mine_map = np.r_[np.full(rows * cols - num_mines, False), np.full(num_mines, True)]
        self._solution_map = np.zeros((rows, cols)).astype(int)

        return super()._resize()

    def refresh(self) -> None:
        """Handle terminal display refresh."""
        if self.timer and not self.revealed:
            # Timer on the right of scoreboard
            self._scoreboard[0, -3:] = str(int(self.timer)).rjust(3, '0')
            self.timer += DELTA

            # Flagging count on the left
            self._scoreboard[0, :3] = str(self._num_mines - (self._state_map == FLAGGED_STATE).sum()).rjust(3, '0')

        return super().refresh()

    def on_press(self, key: int) -> bool:
        """Handle key press events."""
        # Initialize timer
        if not self.timer:
            self.timer = DELTA
            self._scoreboard[1, :] = ' '
            self._marching_task.cancel()

        if key == FORFEIT_KEY:
            self.reveal_mines()

        elif key == TAB:
            params = next(self.difficulties)
            self.resize(*params)
            self.init_lawn()
            self._scoreboard[1, :] = '{}x{}/{}x{}'.format(*self._shape,
                                                          MINE_SYMBOL, self._num_mines).ljust(self._shape[1], ' ')

            cursor.set_wrapping(*self._shape)

        elif key == RESET_KEY:
            self.init_lawn()

        elif not self.revealed:
            if key == SPACE_KEY:
                self.poke()

            if key == FLAG_KEY:
                self.flag()

        else:
            return super().on_press(key)
        return True

    def flag(self) -> None:
        """Flag the location for potential mine."""
        row, col = cursor.top - OFFSET_TOP, cursor.left - OFFSET_LEFT
        if self._state_map[row, col] == COVERED_STATE:
            self._state_map[row, col] = FLAGGED_STATE
            self[row, col] = FLAG_SYMBOL
        elif self._state_map[row, col] == FLAGGED_STATE:
            self._state_map[row, col] = COVERED_STATE
            self[row, col] = COVERED_SYMBOL

    def poke(self) -> None:
        """Uncover the pointed location."""
        row, col = cursor.top - OFFSET_TOP, cursor.left - OFFSET_LEFT
        rows, cols = self._shape

        def uncover_land(r: int, c: int) -> None:
            """Uncover adjacent locations when the adjacent mine count is 0."""
            if not (0 <= r < rows and 0 <= c < cols) or state_map[r, c] == UNCOVERED_STATE:
                return
            state_map[r, c] = UNCOVERED_STATE

            # Uncover the 8 adjacent locations
            if self._solution_map[r, c] == EMPTY_SYMBOL:
                for rr in range(r - 1, r + 2):
                    for cc in range(c - 1, c + 2):
                        if not (rr == r and cc == c):
                            uncover_land(rr, cc)

        if self._state_map[row, col] != COVERED_STATE:
            return

        if self._mine_map[row, col]:
            self.lose(row, col)
        else:
            state_map = self._state_map.copy()
            uncover_land(row, col)
            self._state_map = state_map
            self[:, :] = np.where(self._state_map == UNCOVERED_STATE, self._solution_map, self[:, :])
            self.evaluate()

    def win(self) -> None:
        """Handle winning."""
        # Replace good flags with boxed check marks, and all other locations with solutions
        self[:, :] = np.where(self._mine_map,
                              BOXEDCHECK_SYMBOL, self._solution_map)

        # Put up smiley and winning shoutout
        self._scoreboard[0, len(self._scoreboard[0]) // 2] = HAPPYFACE_SYMBOL
        self._scoreboard[1, :8] = "You win!"
        self.schedule_marching(.1)

        self.revealed = True

    def lose(self, r: int, c: int) -> None:
        """Handle losing."""
        # Replace good flags with boxed check marks, bad flags with boxed crosses,
        #  and all other locations with solutions
        self[:, :] = np.where(self._mine_map,
                              np.where(self._state_map == FLAGGED_STATE, BOXEDCHECK_SYMBOL, MINE_SYMBOL),
                              np.where(self._state_map == FLAGGED_STATE, BOXEDCROSS_SYMBOL, self._solution_map))

        # Color exploded mine
        colors_copy = np.full(self._shape, self.color)
        colors_copy[r, c] = colors.WHITE_ON_RED
        self.colors = colors_copy

        # Put up sad face and losing callout
        self._scoreboard[0, len(self._scoreboard[0]) // 2] = SADFACE_SYMBOL
        self._scoreboard[1, :8] = "You die!"
        self.schedule_marching(.8)

        self.revealed = True

    def evaluate(self) -> None:
        """Evaluate winning or losing."""
        # If all and only all mines are covered, win
        if np.all(self._mine_map == (self._state_map != UNCOVERED_STATE)):
            self.win()
        # Otherwise, if the number of covered cells matches the number of mines, lose
        elif (self._state_map != UNCOVERED_STATE).sum() == self._num_mines:
            row, col = cursor.top - OFFSET_TOP, cursor.left - OFFSET_LEFT
            self.lose(row, col)


with ScreenManager() as gsm:
    num_mines = 10
    rows, cols = 8, 8

    text_len = 20

    # Draw board
    lawn = gsm.root.new_widget(rows=rows, cols=cols, num_mines=num_mines, create_with=Lawn)
    lawn.init_lawn()
    lawn.load_screen(gsm)
    lawn.resize(rows * 2, cols * 2, num_mines)
    lawn.init_lawn()

    # Draw Cursor
    cursor = gsm.root.new_widget(OFFSET_TOP, OFFSET_LEFT, 1, 1, transparent=True, create_with=Cursor)
    cursor.window.addstr(0, 0, CURSOR_SYMBOL)
    cursor.set_wrapping(rows * 2, cols * 2)

    # Schedule refreshing task
    gsm.schedule(gsm.root.refresh, delay=DELTA)
    gsm.run()
