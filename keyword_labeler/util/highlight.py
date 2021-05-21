# Functions to change the color of text in the console


def red_highlight(text: str) -> str:
    red_start = "\u001b[1m\u001b[40m\u001b[31m"
    color_end = "\u001b[0m"

    return red_start + text + color_end


def blue_highlight(text: str) -> str:
    blue_start = "\u001b[1m\u001b[40m\u001b[34m"
    color_end = "\u001b[0m"

    return blue_start + text + color_end


def green_highlight(text: str) -> str:
    green_start = "\u001b[1m\u001b[40m\u001b[32m"
    color_end = "\u001b[0m"

    return green_start + text + color_end


def yellow_highlight(text: str) -> str:
    yellow_start = "\u001b[1m\u001b[40m\u001b[33m"
    color_end = "\u001b[0m"

    return yellow_start + text + color_end
