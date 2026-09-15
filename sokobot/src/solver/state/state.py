import box

class State:
    def __init__(self, boxes_data, person_data):
        self.boxes = boxes_data
        self.person = person_data

    def add_box(self, the_box):
        if self.boxes < 8:
            self.boxes.add(the_box)

    def remove_box(self, box_tag):
        self.boxes.remove(box_tag)

    def set_coordinates(self, box_tag, x, y):
        self.boxes