
class Action:
    def __init__(self, user_input):
        self.direction = user_input

    """
    This method changes the direction set in this object 
    
    Args:
        user_input: a character that gives the direction
    """
    @classmethod
    def change_direction(cls, user_input):
        cls.direction = user_input

    """
    This method gets the direction that has been set in this action
    
    Return:
        the direction set in this class
    """
    @classmethod
    def get_direction(cls):
        return cls.direction