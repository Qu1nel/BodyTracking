import math
from typing import Union, List, Tuple

from tracking.bodyparts.base_solution import Landmark


def find_distance(
        point_one: Union[Landmark, Union[List[int], Tuple[int]]],
        point_two: Union[Landmark, Union[List[int], Tuple[int]]]
) -> float:
    """
    Returns distance between point_one and point_two

    :param point_one: Landmark object or sequence with two coord statement
    :param point_two: Landmark object or sequence with two coord statement
    :return: float of num, I.e. distance between point_one and point_two
    """
    if isinstance(point_one, Landmark) and isinstance(point_two, Landmark):
        return math.sqrt((point_one.x - point_two.x) ** 2 + (point_one.y - point_two.y) ** 2)

    return math.sqrt((point_one[0] - point_two[0]) ** 2 + (point_one[1] - point_two[1]) ** 2)


def find_coord_center_point_distance(
        point_one: Union[Landmark, Union[List[int], Tuple[int]]],
        point_two: Union[Landmark, Union[List[int], Tuple[int]]]
) -> Tuple[int, int]:
    """
    Returns the coords of the center point of the distance between point_one and point_two

    :param point_one: Landmark object or sequence with two coord statement
    :param point_two: Landmark object or sequence with two coord statement
    :return: Int of num
    """
    if isinstance(point_one, Landmark) and isinstance(point_two, Landmark):
        return int((point_one.x + point_two.x) / 2), int((point_one.y + point_two.y) / 2)

    return int((point_one[0] + point_two[0]) / 2), int((point_one[1] + point_two[1]) / 2)
