from examine.examine import Examine


class Deadlock:
    def __init__(self):
        pass

    @classmethod
    def is_deadlock(cls, mapData, itemsData):
        if

    """
    Visualization:
    
    ###
    #$
    #
    """
    @classmethod
    def is_in_corner(cls, mapData, box):
        x, y = box.get_coord()
        left, right, up, down = Examine.items_in_all_directions(mapData, x, y)
        if (left == '#' or right == '#') and (up == '#' or down == '#'):
            return True
        else:
            return False
    """
    hindi ko pa inimplement deadlocks with air. Ex.
    ###
    # $
    #$
    
    or 
    ###
    $ #
    $ #
     $#
    """
    @classmethod
    def is_in_deadlocks_with_air(cls, map_data, item_data, box):
        pass

    @classmethod
    def is_four_closed(cls, map_data, items_data, box):
        x, y = box.get_coord()
        for i in range(8):
            if i % 2:


