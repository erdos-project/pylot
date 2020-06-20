class Lane(object):
    def __init__(self, left_markings, right_markings):
        self.left_markings = left_markings
        self.right_markings = right_markings

    def draw_on_world(self, world):
        for marking in self.left_markings:
            world.debug.draw_point(marking.as_carla_location(), size=0.1)
        for marking in self.right_markings:
            world.debug.draw_point(marking.as_carla_location(), size=0.1)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Lane({})'.format(zip(self.left_markings, self.right_markings))
