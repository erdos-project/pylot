import cv2
import numpy as np
import time

from erdos.message import Message
from erdos.op import Op
from erdos.utils import setup_csv_logging, setup_logging, time_epoch_ms

from pylot_utils import add_timestamp, bgra_to_bgr, create_detected_lane_stream, is_camera_stream


class LaneDetectionOperator(Op):
    def __init__(self,
                 name,
                 output_stream_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(LaneDetectionOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._output_stream_name = output_stream_name

    @staticmethod
    def setup_streams(input_streams, output_stream_name):
        # Register a callback on the camera input stream.
        input_streams.filter(is_camera_stream).add_callback(
            LaneDetectionOperator.on_msg_camera_stream)
        return [create_detected_lane_stream(output_stream_name)]

    def on_msg_camera_stream(self, msg):
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image_np = msg.frame

        # TODO(ionel): Implement lane detection.
        edges = self.apply_canny(image_np)
        lines_edges = self.apply_hough(image_np, edges)

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(
            time_epoch_ms(), self.name, msg.timestamp, runtime))

        if self._flags.visualize_lane_detection:
            add_timestamp(msg.timestamp, lines_edges)
            cv2.imshow(self.name, lines_edges)
            cv2.waitKey(1)

        output_msg = Message(image_np, msg.timestamp)
        self.get_output_stream(self._output_stream_name).send(output_msg)

    def apply_canny(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gausian smoothing.
        kernel_size = 5
        blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # TODO(ionel): Explore low and high threshold values.

        # If pixel diffs is larger than this threshold then the pixels
        # are kept.
        high_threshold = 60
        # If pixel diff of neighboring pixels is larger than this threshold
        # then pixels are kept.
        low_threshold = 30
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
        return edges

    def apply_hough(self, image, edges):
        # Create a masked edges image using cv2.fillPoly()
        mask = np.zeros_like(edges)
        ignore_mask_color = 255
        imshape = image.shape
        # TODO(ionel): Do not hard code values! Figure out which region to
        # select.
        vertices = np.array([[(0,imshape[0]),
                              (375, 275),
                              (425, 275),
                              (imshape[1],imshape[0])]],
                            dtype=np.int32)

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2 # distance resolution in pixels of the Hough grid
        theta = np.pi / 180 # angular resolution in radians of the Hough grid
        threshold = 15 # min number of votes (intersections in Hough grid cell)
        min_line_length = 40 # min number of pixels making up a line
        max_line_gap = 20 # max gap in pixels between connectable line segments
        line_image = np.copy(image)*0 # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line
        # segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                                np.array([]), min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges))

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

        return lines_edges

    def execute(self):
        self.spin()
