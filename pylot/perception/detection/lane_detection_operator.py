import cv2
import erdust
import numpy as np
import time

from pylot.utils import add_timestamp, time_epoch_ms


class LaneDetectionOperator(erdust.Operator):
    def __init__(self,
                 camera_stream,
                 detected_lanes_stream,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        camera_stream.add_callback(self.on_msg_camera_stream,
                                   [detected_lanes_stream])
        self._name = name
        self._flags = flags
        self._logger = erdust.utils.setup_logging(name, log_file_name)
        self._csv_logger = erdust.utils.setup_csv_logging(
            name + '-csv', csv_file_name)

    @staticmethod
    def connect(camera_stream):
        detected_lanes_stream = erdust.WriteStream()
        return [detected_lanes_stream]

    def on_msg_camera_stream(self, msg, detected_lanes_stream):
        self._logger.debug('@{}: {} received message'.format(
            msg.timestamp, self._name))
        start_time = time.time()
        assert msg.encoding == 'BGR', 'Expects BGR frames'
        image_np = msg.frame

        # TODO(ionel): Implement lane detection.
        edges = self.apply_canny(image_np)
        lines_edges = self.apply_hough(image_np, edges)

        kernel_size = 3
        grad_x = self.apply_sobel(image_np,
                                  orient='x',
                                  sobel_kernel=kernel_size,
                                  thresh_min=0,
                                  thresh_max=255)
        grad_y = self.apply_sobel(image_np,
                                  orient='y',
                                  sobel_kernel=kernel_size,
                                  thresh_min=0,
                                  thresh_max=255)
        mag_binary = self.magnitude_threshold(image_np,
                                              sobel_kernel=kernel_size,
                                              thresh_min=0,
                                              thresh_max=255)
        dir_binary = self.direction_threshold(image_np,
                                              sobel_kernel=kernel_size,
                                              thresh_min=0,
                                              thresh_max=np.pi / 2)

        s_binary = self.extract_s_channel(image_np)

        # Select the pixels where both x and y gradients meet the threshold
        # criteria, or the gradient magnitude and direction are both with
        # the threshold values.
        combined = np.zeros_like(dir_binary)
        combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) &
                                                    (dir_binary == 1))] = 1

        combined_binary = np.zeros_like(grad_x)
        combined_binary[(s_binary == 1) | (grad_x == 1)] = 1

        # Get runtime in ms.
        runtime = (time.time() - start_time) * 1000
        self._csv_logger.info('{},{},"{}",{}'.format(time_epoch_ms(),
                                                     self._name, msg.timestamp,
                                                     runtime))

        if self._flags.visualize_lane_detection:
            add_timestamp(msg.timestamp, lines_edges)
            cv2.imshow(self._name, lines_edges)
            cv2.waitKey(1)

        detected_lanes_stream.send(erdust.Message(msg.timestamp, image_np))

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
        vertices = np.array([[(0, imshape[0]), (375, 275), (425, 275),
                              (imshape[1], imshape[0])]],
                            dtype=np.int32)

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Define the Hough transform parameters
        # Make a blank the same size as our image to draw on
        rho = 2  # distance resolution in pixels of the Hough grid
        theta = np.pi / 180  # angular resolution in radians of the Hough grid
        threshold = 15  # min number of votes (intersections in Hough grid cell)
        min_line_length = 40  # min number of pixels making up a line
        max_line_gap = 20  # max gap in pixels between connectable line segments
        line_image = np.copy(image) * 0  # creating a blank to draw lines on

        # Run Hough on edge detected image
        # Output "lines" is an array containing endpoints of detected line
        # segments
        lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold,
                                np.array([]), min_line_length, max_line_gap)

        # Iterate over the output "lines" and draw lines on a blank image
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)

        # Create a "color" binary image to combine with line image
        color_edges = np.dstack((edges, edges, edges))

        # Draw the lines on the edge image
        lines_edges = cv2.addWeighted(color_edges, 0.8, line_image, 1, 0)

        return lines_edges

    def apply_sobel(self,
                    img,
                    orient='x',
                    sobel_kernel=3,
                    thresh_min=20,
                    thresh_max=100):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if orient == 'x':
            # Calculate the derivative in x direction.
            abs_sobel = np.absolute(
                cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
        elif orient == 'y':
            # Calculate the derivative in y direction.
            abs_sobel = np.absolute(
                cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
        # Rescale back to 8 bit integer.
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary = np.zeros_like(scaled_sobel)
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary

    def magnitude_threshold(self,
                            img,
                            sobel_kernel=3,
                            thresh_min=0,
                            thresh_max=255):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag) / 255
        gradmag = (gradmag / scale_factor).astype(np.uint8)
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= thresh_min) & (gradmag <= thresh_max)] = 1
        return binary_output

    def direction_threshold(self,
                            img,
                            sobel_kernel=3,
                            thresh_min=0,
                            thresh_max=np.pi / 2):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Take the absolute value of the gradient direction
        abs_grad_dir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
        binary_output = np.zeros_like(abs_grad_dir)
        binary_output[(abs_grad_dir >= thresh_min)
                      & (abs_grad_dir <= thresh_max)] = 1
        return binary_output

    def extract_s_channel(self, img, thresh_min=170, thresh_max=255):
        # Convert to HLS color space.
        hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # Exctract the s channel
        s_channel = hls_img[:, :, 2]
        binary = np.zeros_like(s_channel)
        binary[(s_channel > thresh_min) & (s_channel <= thresh_max)] = 1
        return binary
