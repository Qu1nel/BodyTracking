from typing import Union, List, Tuple, Optional

import cv2
from numpy import ndarray

from tracking.drawing.colors import *
from tracking.bodyparts.base_solution import Landmark

__doc__ = """Just redefined funcs)"""
__all__ = (
    'draw_line',
    'draw_circle',
    'draw_fill_circle',
    'draw_rect',
    'draw_fill_rect',
    'draw_text'
)


def draw_line(
        image: ndarray,
        point1: Union[Landmark, Union[List[int], Tuple[int]]],
        point2: Union[Landmark, Union[List[int], Tuple[int]]],
        color: Union[Tuple[int], List[int]] = RED,
        thickness: Optional[int] = None
) -> None:
    """
    The function line draws the line segment between pt1 and pt2 points in the image. The line is
    clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected
    or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased
    lines are drawn using Gaussian filtering.

    :param image: Image.
    :param point1: Landmark object or sequence with two coord statement.
    :param point2: Landmark object or sequence with two coord statement.
    :param color: The sequence contains a three values. RGB.
    :param thickness: Int. Line thickness value.
    :return: None
    """
    cv2.line(image, point1, point2, color, thickness)


def draw_circle(
        image: ndarray,
        center: Union[Landmark, Union[List[int], Tuple[int]]],
        radius: int,
        color: Union[Tuple[int], List[int]] = RED,
        thickness: Optional[int] = None
) -> None:
    """
    Draws a circle.

    :param image: Image.
    :param center: Center of the circle.
    :param radius: Radius of the circle.
    :param color: The sequence contains a three values. RGB.
    :param thickness: Int. Circle thickness value.
    :return: None
    """
    cv2.circle(image, center, radius, color, thickness)


def draw_fill_circle(
        image: ndarray,
        center: Union[Landmark, Union[List[int], Tuple[int]]],
        radius: int,
        color: Union[Tuple[int], List[int]] = RED
) -> None:
    """
    Draws a circle.

    :param image: Image.
    :param center: Center of the circle.
    :param radius: Radius of the circle.
    :param color: The sequence contains a three values. RGB.
    :return: None
    """
    cv2.circle(image, center, radius, color, -1)


def draw_rect(
        image: ndarray,
        point1: Union[Landmark, Union[List[int], Tuple[int]]],
        point2: Union[Landmark, Union[List[int], Tuple[int]]],
        color: Union[Tuple[int], List[int]] = RED,
        thickness: Optional[int] = None
) -> None:
    """
    Draws a rectangle.

    The function cv::rectangle draws a rectangle outline whose two opposite corners
    are point1 and point2.

    :param image: Image.
    :param point1: Landmark object or sequence with two coord statement.
    :param point2: Landmark object or sequence with two coord statement.
    :param color: The sequence contains a three values. RGB.
    :param thickness: Int. Rectangle thickness value.
    :return: None
    """
    cv2.rectangle(image, point1, point2, color, thickness)


def draw_fill_rect(
        image: ndarray,
        point1: Union[Landmark, Union[List[int], Tuple[int]]],
        point2: Union[Landmark, Union[List[int], Tuple[int]]],
        color: Union[Tuple[int], List[int]] = RED
) -> None:
    """
    Draws a filled rectangle.

    The function cv::rectangle draws a filled rectangle outline whose two opposite corners
    are point1 and point2.

    :param image: Image.
    :param point1: Landmark object or sequence with two coord statement.
    :param point2: Landmark object or sequence with two coord statement.
    :param color: The sequence contains a three values. RGB.
    :return: None
    """
    cv2.rectangle(image, point1, point2, color, -1)


def draw_text(
        image: ndarray,
        text: str,
        org: Union[List[int], Tuple[int]],
        fontFace: int, fontScale: int,
        color: Union[Tuple[int], List[int]] = PURPLE,
        thickness: Optional[int] = None
) -> None:
    """
    Draws a text string.

    The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered
    using the specified font are replaced by question marks.

    :param image: Image.
    :param text: Text string to be drawn.
    :param org: It is the coordinates of the bottom-left corner of the text string in the image.
            The coordinates are represented as tuples of two values i.e. (X coordinate value, Y coordinate value).
    :param fontFace: It denotes the font type. Some of font types are FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN.
    :param fontScale: Font scale factor that is multiplied by the font-specific base size.
    :param color: The sequence contains a three values. RGB.
    :param thickness: Int. Text thickness value.
    :return: None
    """
    cv2.putText(image, text, org, fontFace, fontScale, color, thickness)
