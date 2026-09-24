from examine.examine import examine

"""
This is a class method which checks if the state is in deadlock. 

Args:
    map_data: the immutable items of the map;
    items_data: the movable items.

Returns:
    True if deadlock
    False otherwise.

Note: 
    Calls the other functions in this module.
"""
def is_deadlock(map_data, items_data):
    pass

"""
Visualization:

###
#$
#

This method checks if the box is in the corner of a wall. 

Args:
    map_data: the immutable map
    box: the box to be examined
    
Returns:
    True if in corner
    False otherwise
"""
def _is_in_corner(map_data, box):
    x, y = box.get_coord()
    left, right, up, down = Examine.items_in_all_directions(map_data, x, y)
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
 
 This method checks if there are "air deadlocks" -- as seen in the examples provided above. 
 
 Args:
    map_data: the data of the map
    item_data: the items in the map that are movable
    box: the box to be examined.
"""
def _is_in_deadlocks_with_air(map_data, item_data, box):
    pass

"""
this method scouts the board if there are any clusters of four scattered given a location.

Args:
    map_data: the data of the map
    items_data: the items of the map (must be a snapshot)
    box: the box to be examined

Returns:
    True if there is a block of four
    False otherwise. 
"""
def _is_four_closed(map_data, items_data, box):
    x, y = box.get_coord()
    for i in range(8):
        if i % 2:
            pass
        pass


