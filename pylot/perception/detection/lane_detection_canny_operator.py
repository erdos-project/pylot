"""Implements an operator that detects lanes."""
import math
from collections import namedtuple

import cv2

import erdos
from erdos.operator import OneInOneOut
from erdos.context import OneInOneOutContext

import numpy as np

Line = namedtuple("Line", "x1, y1, x2, y2, slope")

from pylot.perception.camera_frame import CameraFrame


class CannyEdgeLaneDetectionOperator(OneInOneOut):
    """Detects driving lanes using a camera.

    The operator uses standard vision techniques (Canny edge).

    Args:
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, flags):
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._kernel_size = 7

    def on_data(self, context: OneInOneOutContext, data: CameraFrame):
        """Invoked whenever a frame message is received on the stream."""
        self._logger.debug('@{}: {} received message'.format(
            context.timestamp, self.config.name))
        assert data.encoding == 'BGR', 'Expects BGR frames'
        # Make a copy of the image coming into the operator.
        image = np.copy(data.as_numpy_array())

        # Get the dimensions of the image.
        x_lim, y_lim = image.shape[1], image.shape[0]

        # Convert to grayscale.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur.
        image = cv2.GaussianBlur(image, (self._kernel_size, self._kernel_size),
                                 0)

        # Apply the Canny Edge Detector.
        image = cv2.Canny(image, 30, 60)

        # Define a region of interest.
        points = np.array(
            [[
                (0, y_lim),  # Bottom left corner.
                (0, y_lim - 60),
                (x_lim // 2 - 20, y_lim // 2),
                (x_lim // 2 + 20, y_lim // 2),
                (x_lim, y_lim - 60),
                (x_lim, y_lim),  # Bottom right corner.
            ]],
            dtype=np.int32)
        image = self._region_of_interest(image, points)

        # Hough lines.
        image = self._draw_lines(image)

        cv2.imshow('image', image)
        cv2.waitKey(0)

        context.write_stream.send(erdos.Message(context.timestamp, image))

    def _region_of_interest(self, image, points):
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, points, 255)
        return cv2.bitwise_and(image, mask)

    def _extrapolate_lines(self, image, left_line, right_line):
        top_y = None
        if left_line is not None and right_line is not None:
            top_y = min(
                [left_line.y1, left_line.y2, right_line.y1, right_line.y2])
        base_y = image.shape[0]

        final_lines = []
        if left_line is not None:
            actual_slope = float(left_line.y2 -
                                 left_line.y1) / float(left_line.x2 -
                                                       left_line.x1)
            base_x = int((base_y - left_line.y1) / actual_slope) + left_line.x1
            final_lines.append(
                Line(base_x, base_y, left_line.x1, left_line.y1, actual_slope))

            if top_y is None:
                top_y = min([left_line.y1, left_line.y2])

            top_x = int((top_y - left_line.y2) / actual_slope) + left_line.x2
            final_lines.append(
                Line(top_x, top_y, left_line.x2, left_line.y2, actual_slope))

        if right_line is not None:
            actual_slope = float(right_line.y2 -
                                 right_line.y1) / float(right_line.x2 -
                                                        right_line.x1)
            base_x = int(
                (base_y - right_line.y1) / actual_slope) + right_line.x1
            final_lines.append(
                Line(base_x, base_y, right_line.x1, right_line.y1,
                     actual_slope))

            if top_y is None:
                top_y = min([right_line.y1, right_line.y2])

            top_x = int((top_y - right_line.y2) / actual_slope) + right_line.x2
            final_lines.append(
                Line(top_x, top_y, right_line.x2, right_line.y2, actual_slope))
        return final_lines

    def _draw_lines(self, image):
        lines = cv2.HoughLinesP(image,
                                rho=1,
                                theta=np.pi / 180.0,
                                threshold=40,
                                minLineLength=10,
                                maxLineGap=30)
        line_img = np.zeros((image.shape[0], image.shape[1], 3),
                            dtype=np.uint8)

        if lines is None:
            return line_img

        # Construct the Line tuple collection.
        cmp_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = math.degrees(math.atan2(y2 - y1, x2 - x1))
                cmp_lines.append(Line(x1, y1, x2, y2, slope))

        # Sort the lines by their slopes after filtering lines whose slopes
        # are > 20 or < -20.
        cmp_lines = sorted(filter(
            lambda line: line.slope > 20 or line.slope < -20, cmp_lines),
                           key=lambda line: line.slope)

        if len(cmp_lines) == 0:
            return line_img

        # Filter the lines with a positive and negative slope and choose
        # a single line out of those.
        left_lines = [
            line for line in cmp_lines if line.slope < 0 and line.x1 < 300
        ]
        right_lines = [
            line for line in cmp_lines
            if line.slope > 0 and line.x1 > image.shape[1] - 300
        ]

        final_lines = []
        # Find the longest line from the left and the right lines and
        # extrapolate to the middle of the image.
        left_line = None
        if len(left_lines) != 0:
            left_line = max(left_lines,
                            key=lambda line: abs(line.y2 - line.y1))
            final_lines.append(left_line)

        right_line = None
        if len(right_lines) != 0:
            right_line = max(right_lines,
                             key=lambda line: abs(line.y2 - line.y1))
            final_lines.append(right_line)

        final_lines.extend(
            self._extrapolate_lines(image, left_line, right_line))

        for x1, y1, x2, y2, slope in final_lines:
            cv2.line(line_img, (x1, y1), (x2, y2),
                     color=(255, 0, 0),
                     thickness=2)
            cv2.putText(line_img, "({}, {})".format(x1, y1), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2,
                        cv2.LINE_AA)
        return line_img

    def destroy(self):
        self._logger.warn('destroying {}'.format(self.config.name))
