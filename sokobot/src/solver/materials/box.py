
class Box:
    def __init__(self, id_tag, x, y):
        self.id_tag = id_tag
        self.x = x
        self.y = y

    def get_coord(self):
        return self.x, self.y

    def get_id_tag(self):
        return self.id_tag