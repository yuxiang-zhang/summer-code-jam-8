import curses
import curses.ascii
import time

from mine import playMinesweeper
from nurses.keys import BACKSPACE, ESCAPE
from pong import playPong
from snake import playSnake


def init():
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_RED)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_GREEN)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_YELLOW)
    curses.init_pair(4, curses.COLOR_BLUE, curses.COLOR_BLUE)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_MAGENTA)
    curses.init_pair(6, curses.COLOR_CYAN, curses.COLOR_CYAN)
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_WHITE)
    curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_BLACK)
    curses.init_pair(9, curses.COLOR_BLACK, curses.COLOR_GREEN)


def create_box(stdscr, x, y, x2, y2, color):
    for y3 in range(y, y2):
        for x3 in range(x, x2):
            stdscr.addch(y3, x3, curses.ACS_BLOCK, curses.color_pair(color))


def createGameboy(stdscr):
    h, w = stdscr.getmaxyx()
    w1 = w//2-25
    h1 = h//2-20
    w2 = w//2+25
    h2 = h//2+20

    create_box(stdscr, w1, h1, w2, h2, 7)  # main gray box

    create_box(stdscr, w1+2, h1+1, w2-2, h1+15, 8)  # outline around green play area

    # create_box(stdscr, w1+4, h1+3, w2-4, h1+13 , 2) # green play area

    create_box(stdscr, w1+6, h1+19, w1+9, h1+25, 8)  # vertical of d-pad

    create_box(stdscr, w1+3, h1+21, w1+12, h1+23, 8)  # horizontal of d-pad

    create_box(stdscr, w2-15, h1+23, w2-11, h1+25, 1)  # bottom left button

    create_box(stdscr, w2-10, h1+19, w2-6, h1+21, 1)  # top right button


def writeMenu(stdscr):
    h, w = stdscr.getmaxyx()
    w1 = w//2-25
    h1 = h//2-20
    w2 = w//2+25
    h2 = h//2+20
    for x in range(w1+4, w2-4):
        for y in range(h1+3, h1+13):
            stdscr.addch(y, x, curses.ACS_BLOCK, curses.color_pair(2))

    # draw box surrounding snake game
    stdscr.addstr(h1+4, w1+7, "┌", curses.color_pair(9))
    stdscr.addstr(h1+9, w1+7, "└", curses.color_pair(9))
    stdscr.addstr(h1+4, w1+17, "┐", curses.color_pair(9))
    stdscr.addstr(h1+9, w1+17, "┘", curses.color_pair(9))

    for i in range(5, 9):
        stdscr.addstr(h1+i, w1+7, "│", curses.color_pair(9))
        stdscr.addstr(h1+i, w1+17, "│", curses.color_pair(9))

    for i in range(8, 17):
        stdscr.addstr(h1+4, w1+i, "─", curses.color_pair(9))
        stdscr.addstr(h1+9, w1+i, "─", curses.color_pair(9))

    stdscr.addstr(h1+5, w1+10, "Snake", curses.color_pair(9))
    """for i in range(10, 15): # underline snake (bc underline attribute on str causes error idk why)
        stdscr.addstr(h1+6, w1+i, "¯", curses.color_pair(9))"""

    stdscr.addstr(h1+8, w1+9, "─", curses.color_pair(9))  # draw mini snake
    stdscr.addstr(h1+8, w1+10, "─", curses.color_pair(9))
    stdscr.addstr(h1+8, w1+11, "┘", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+11, "┌", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+12, "─", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+13, "─", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+15, "·", curses.color_pair(9))

    # box for pong
    stdscr.addstr(h1+4, w1+18, "┌", curses.color_pair(9))
    stdscr.addstr(h1+9, w1+18, "└", curses.color_pair(9))
    stdscr.addstr(h1+4, w1+28, "┐", curses.color_pair(9))
    stdscr.addstr(h1+9, w1+28, "┘", curses.color_pair(9))

    for i in range(5, 9):
        stdscr.addstr(h1+i, w1+18, "│", curses.color_pair(9))
        stdscr.addstr(h1+i, w1+28, "│", curses.color_pair(9))

    for i in range(19, 28):
        stdscr.addstr(h1+4, w1+i, "─", curses.color_pair(9))
        stdscr.addstr(h1+9, w1+i, "─", curses.color_pair(9))

    stdscr.addstr(h1+5, w1+21, "Pong", curses.color_pair(9))

    stdscr.addstr(h1+6, w1+19, "│", curses.color_pair(9))  # draw mini pong
    stdscr.addstr(h1+8, w1+26, "│", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+23, "·", curses.color_pair(9))

    # box for minesweeper
    stdscr.addstr(h1+4, w1+29, "┌", curses.color_pair(9))
    stdscr.addstr(h1+10, w1+29, "└", curses.color_pair(9))
    stdscr.addstr(h1+4, w1+41, "┐", curses.color_pair(9))
    stdscr.addstr(h1+10, w1+41, "┘", curses.color_pair(9))

    for i in range(5, 10):
        stdscr.addstr(h1+i, w1+29, "│", curses.color_pair(9))
        stdscr.addstr(h1+i, w1+41, "│", curses.color_pair(9))

    for i in range(30, 41):
        stdscr.addstr(h1+4, w1+i, "─", curses.color_pair(9))
        stdscr.addstr(h1+10, w1+i, "─", curses.color_pair(9))

    stdscr.addstr(h1+5, w1+30, "MineSweeper", curses.color_pair(9))

    stdscr.addstr(h1+7, w1+32, "❑", curses.color_pair(9))  # mini minesweeper
    stdscr.addstr(h1+7, w1+34, "☢", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+36, "❑", curses.color_pair(9))
    stdscr.addstr(h1+7, w1+38, "☠", curses.color_pair(9))
    stdscr.addstr(h1+8, w1+32, "❑", curses.color_pair(9))
    stdscr.addstr(h1+8, w1+34, "❑", curses.color_pair(9))
    stdscr.addstr(h1+8, w1+36, "❑", curses.color_pair(9))
    stdscr.addstr(h1+8, w1+38, "☢", curses.color_pair(9))

    stdscr.addstr(h1+11, w1+8, "Thinking inside boxes®", curses.color_pair(9))


def main(stdscr):
    key = None

    curses.initscr()
    stdscr.keypad(True)
    curses.curs_set(0)
    curses.start_color()

    curses.curs_set(0)
    init()
    createGameboy(stdscr)
    writeMenu(stdscr)
    key = stdscr.getch()

    if key == ord('1'):
        stdscr.clear()
        playSnake()
    elif key == ord('2'):
        stdscr.clear()
        playPong()
    elif key == ord('3'):
        stdscr.clear()
        playMinesweeper()
    elif key == ESCAPE:
        pass


try:
    curses.wrapper(main)
except Exception as e:
    print(e)
    print("Your screen resolution might be too small to render all game elements. Please try running in fullscreen.")
