

"""
this method examines a state. This method is mean to be used for ALL algorithms to abstract away from the goal.

Args:
    map_data - the immutable map
    items_data - the mutable items

Returns:
    Boolean Value:
        True - "go ahead"
        False - "stay behind"

Notes: This function calls methods from the examine folder.
"""
#not yet finished
def examine(map_data, items_data):
    if is_goal(map_data, items_data):
        pass
    if is_deadlock(map_data, items_data):
        pass
    pass

def is_goal(map_data, items_data):
    is_goal = True
    for boxes in items_data.boxes:
        x, y = boxes.get_coord()
        if not map_data[x][y] == '.' :
            is_goal = False
    return is_goal

#not yet finished
def is_deadlock(map_data, items_data):
    return True

"""
The helper functions that will help the other functions.
"""
def item_in_map(map_data, x, y):
    return map_data[x][y]

def items_in_all_directions(map_data, x, y):
    left = item_in_map(map_data, x - 1, y)
    right = item_in_map(map_data, x + 1, y)
    up = item_in_map(map_data, x, y - 1)
    down = item_in_map(map_data, x, y + 1)
    return left, right, up, down

#not yet finished
def clockwise(index):
    pass

#not yet finished
def count_three(items_data):
    pass
