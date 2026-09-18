"""
This class represents the state. This will be one object that will 
be put in the dfs, bfs servers
"""
class State:
    def __init__(self, boxes_data, person_data, prev_state):
        self.boxes = boxes_data
        self.person = person_data
        self.prev_state = prev_state

    """
        This pseudo-constructor takes the statistics from the previous
        state and creates a state object from it

        Args:
            cls: the class itself
            items_data: the data of the items

        Return: 
            A new state object 

        Note: DO NOT USE THIS ON OTHER STATES
    """
    @classmethod
    def from_prev_state(cls, state, prev_state):
        return cls(state.boxes_data, state.person_data, prev_state)

    """
    This pseudo-constructor takes the initial original items
    creates a state object from it
    
    Args:
        cls: the class itself
        items_data: the data of the items
        
    Return: 
        A new state object 
        
    Note: DO NOT USE THIS ON OTHER STATES
    """
    @classmethod
    def from_initial_state(cls, items_data):
        return cls(items_data.boxes_data, items_data.person_data, null)

    """
    This function gets a box in this state using a number
    
    Args:
        id_tag: an integer that identifies the box that we wish to retrieve
    
    Return: 
        The laman of the box requested
    
    ASSUMPTION: THIS USES get() AS A WAY TO IDENTIFY THE BOX NUMBER
    THIS WILL CHANGE DEPENDING ON THE DATA STRUCTURE OF THE ARRAY OF BOXES
    """
    def get_box (self, id_tag):
        return self.boxes.get(id_tag)

    """
    This method gets the statistics of the person.
    
    Returns:
        The person object
    """
    def get_person(self):
        return self.person
