class Lane(object):
    def __init__(self, left_markings, right_markings):
        self.left_markings = left_markings
        self.right_markings = right_markings

    def draw_on_frame(self, frame, inverse_transform=None):
        """Draw lane markings on a frame.

        Args:
            bgr_frame: Frame on which to draw the waypoints.
            inverse_transform (optional): To be used to transform the waypoints
                to relative to the ego vehicle.
        """
        extrinsic_matrix = frame.camera_setup.get_extrinsic_matrix()
        intrinsic_matrix = frame.camera_setup.get_intrinsic_matrix()
        for marking in self.left_markings:
            if inverse_transform:
                marking = inverse_transform * marking
            pixel_location = marking.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_point(pixel_location, [0, 0, 0])
        for marking in self.right_markings:
            if inverse_transform:
                marking = inverse_transform * marking
            pixel_location = marking.location.to_camera_view(
                extrinsic_matrix, intrinsic_matrix)
            frame.draw_point(pixel_location, [0, 0, 0])

    def draw_on_world(self, world):
        for marking in self.left_markings:
            world.debug.draw_point(marking.as_carla_location(), size=0.1)
        for marking in self.right_markings:
            world.debug.draw_point(marking.as_carla_location(), size=0.1)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return 'Lane({})'.format(zip(self.left_markings, self.right_markings))
