
class Examine:
    def __init__(self):
        pass

    #not yet finished
    @classmethod
    def examine(cls, map_data, items_data):
        if cls.is_goal(map_data, items_data):
            pass
        if cls.is_deadlock(map_data, items_data):
            pass
        pass

    @classmethod
    def is_goal(cls, map_data, items_data):
        is_goal = True
        for boxes in items_data.boxes:
            x, y = boxes.get_coord()
            if not map_data[x][y] == '.' :
                is_goal = False
        return is_goal

    #not yet finished
    @classmethod
    def is_deadlock(cls, map_data, items_data):
        return True

    """
    The helper functions that will help the other functions.
    """
    @classmethod
    def item_in_map(cls, map_data, x, y):
        return map_data[x][y]

    @classmethod
    def items_in_all_directions(cls, map_data, x, y):
        left = cls.item_in_map(map_data, x - 1, y)
        right = cls.item_in_map(map_data, x + 1, y)
        up = cls.item_in_map(map_data, x, y - 1)
        down = cls.item_in_map(map_data, x, y + 1)
        return left, right, up, down

    #not yet finished
    @classmethod
    def clockwise(cls, index):
        pass

    #not yet finished
    @classmethod
    def count_three(cls, items_data):
        pass
