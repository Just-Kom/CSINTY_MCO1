
class Trigger:
    @staticmethod
    def trigger(cls, state, action):

        person_data = cls.change_coord(state.person_data, action)

        boxes_data = state.boxes_data
        # hindi pa toh naayos, the adjustment of boxes_data
        new_state = State(boxes_data, person_data, state)
        # there is an error here
        return new_state

    @staticmethod
    def change_coord(coordinate, action):
        if action == 'l':
            coordinate.x += 1
        elif action == 'r':
            coordinate.x -= 1
        elif action == 'u':
            coordinate.y += 1
        elif action == 'd':
            coordinate.y -= 1

        return coordinate