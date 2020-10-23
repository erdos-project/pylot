import random
import time
from collections import namedtuple

from carla import Client, Location, Rotation, Transform, command

import cv2

import numpy as np

import scipy.signal


Threshold = namedtuple("Threshold", "min, max")

thresholds = {
    "gradients": Threshold(min=40, max=130),
    "saturation": Threshold(min=30, max=100),
    "direction": Threshold(min=0.5, max=1),
}

world = None
counter = 0


def find_lane_pixels(binary_warped):
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peaks of the left and right lanes.
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 4:, :], axis=0)
    midpoint = np.int(histogram.shape[0] // 2)
    left_peaks, right_peaks = scipy.signal.find_peaks(
        histogram[:midpoint],
        height=10)[0], scipy.signal.find_peaks(histogram[midpoint:],
                                               height=10)[0]
    if len(left_peaks) >= 1:
        leftx_base = left_peaks[-1]
    else:
        leftx_base = np.argmax(histogram[:midpoint])
    if len(right_peaks) >= 1:
        rightx_base = right_peaks[0] + midpoint
    else:
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 20
    # Set the width of the windows +/- margin
    margin = 30
    # Set minimum number of pixels found to recenter window
    minpix = 150

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    left_recenters, right_recenters = 0, 0
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &
                          (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &
                           (nonzerox < win_xright_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            left_lane_inds.append(good_left_inds)
            left_recenters += 1
        if len(good_right_inds) > minpix:
            cv2.rectangle(out_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            right_lane_inds.append(good_right_inds)
            right_recenters += 1

    # Extract left and right line pixel positions
    leftx, lefty, rightx, righty = None, None, None, None
    if left_recenters > 5:
        left_lane_inds = np.concatenate(left_lane_inds)
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]

    if right_recenters > 5:
        right_lane_inds = np.concatenate(right_lane_inds)
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit, right_fit = None, None
    if leftx is not None and lefty is not None:
        left_fit = np.polyfit(lefty, leftx, 2)

    if rightx is not None and righty is not None:
        right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting.
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx, right_fitx = None, None
    if left_fit is not None:
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        for x, y in zip(left_fitx, ploty):
            cv2.circle(out_img, (int(x), int(y)), 1, (0, 0, 255))

    if right_fit is not None:
        right_fitx = right_fit[0] * ploty**2 + right_fit[
            1] * ploty + right_fit[2]
        for x, y in zip(right_fitx, ploty):
            cv2.circle(out_img, (int(x), int(y)), 1, (0, 0, 255))

    return out_img, ploty, left_fitx, right_fitx


def process_images(msg):
    global world
    global counter
    print("AAA")
    # Convert the BGRA image to BGR.
    image = np.frombuffer(msg.raw_data, dtype=np.dtype('uint8'))
    image = np.reshape(image, (msg.height, msg.width, 4))[:, :, :3]
    cv2.imshow("Base Image", image)
    cv2.imwrite('test_images/image{:04}.png'.format(counter), image)
    counter += 1

    # Convert the image from BGR to HLS.
    s_channel = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)[:, :, 2]

    # Apply the Sobel operator in the x-direction.
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray_image, 100, 120)
    line_img = np.zeros_like(canny, dtype=np.uint8)
    lines = cv2.HoughLinesP(canny,
                            rho=1,
                            theta=np.pi / 180.0,
                            threshold=40,
                            minLineLength=10,
                            maxLineGap=30)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2),
                     color=(255, 0, 0),
                     thickness=2)
    #cv2.imshow("Canny", canny)
    #cv2.imshow("line_img", line_img)
    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)

    # Apply the Sobel operator in the y-direction.
    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    # Get the absolute values of x, y and xy gradients.
    abs_sobelxy = np.sqrt(sobelx**2 + sobely**2)

    # Threshold the magnitude of the gradient.
    scaled_sobel = np.uint8(255 * abs_sobelxy / np.max(abs_sobelxy))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel > thresholds['gradients'].min)
             & (scaled_sobel < thresholds['gradients'].max)] = 1

    # Threshold the color channel.
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > thresholds['saturation'].min)
             & (s_channel < thresholds['saturation'].max)] = 1

    # Take a bitwise or of all our heuristics.
    final_image = np.zeros_like(sxbinary)
    final_image[(s_binary == 1) | (sxbinary == 1)] = 255

    # Convert the image to a bird's eye view.
    #source_points = np.float32([[30, msg.height], [353, 332], [440, 332],
    #                            [780, msg.height]])
    #source_points = np.float32([[163, 500], [353, 332], [440, 332], [636, 500]])
    #source_points = np.float32([[0, msg.height], [352, 323], [404, 323], [725, msg.height]])
    #source_points = np.float32([[117, 500], [338, 332], [417, 332], [616, 500]])
    #source_points = np.float32([[153, 500], [353, 332], [440, 332], [636, 500]])
    #source_points = np.float32([[100, msg.height], [353, 332], [440, 332],
    #                            [700, msg.height]])
    #source_points = np.float32([[80, msg.height], [327, 368], [465, 368], [688, msg.height]])
    #source_points = np.float32([[139, msg.height], [357, 350], [457, 350], [757, msg.height]])
    source_points = np.float32([[475, msg.height], [600, 370], [798, 370],
                                [928, msg.height]])
    offset = 400
    destination_points = np.float32([[offset, msg.height], [offset, 50],
                                     [msg.width - offset, 50],
                                     [msg.width - offset, msg.height]])
    M = cv2.getPerspectiveTransform(source_points, destination_points)
    M_inv = cv2.getPerspectiveTransform(destination_points, source_points)
    warped_image = cv2.warpPerspective(sxbinary * 255, M,
                                       (msg.width, msg.height))
    cv2.imshow("Warped Image", warped_image)

    # Fit the polynomial.
    fit_img, ploty, left_fit, right_fit = fit_polynomial(warped_image)
    cv2.imshow("Fitted image", fit_img)
    result = image
    if left_fit is not None and right_fit is not None:
        filled_image = np.zeros_like(warped_image).astype(np.uint8)
        color_filled = np.dstack((filled_image, filled_image, filled_image))
        left_points = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right_points = np.array(
            [np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left_points, right_points))
        cv2.fillPoly(color_filled, np.int_([points]), (0, 255, 0))
        unwarped = cv2.warpPerspective(color_filled, M_inv,
                                       (msg.width, msg.height))
        result = cv2.addWeighted(image, 1, unwarped, 0.3, 0)
    cv2.imshow("Unwarped", result)
    counter += 1
    print("BBBB")
    world.tick()


def spawn_driving_vehicle(client, world):
    """ This function spawns the driving vehicle and puts it into
    an autopilot mode.

    Args:
        client: The client instance representing the simulation to
          connect to.
        world: The world inside the current simulation.

    Returns:
        An Actor instance representing the vehicle that was just spawned.
    """
    # Get the blueprint of the vehicle and set it to AutoPilot.
    vehicle_bp = random.choice(
        world.get_blueprint_library().filter('vehicle.*'))
    while not vehicle_bp.has_attribute('number_of_wheels') or not int(
            vehicle_bp.get_attribute('number_of_wheels')) == 4:
        vehicle_bp = random.choice(
            world.get_blueprint_library().filter('vehicle.*'))
    vehicle_bp.set_attribute('role_name', 'autopilot')

    # Get the spawn point of the vehicle.
    start_pose = random.choice(world.get_map().get_spawn_points())

    # Spawn the vehicle.
    batch = [
        command.SpawnActor(vehicle_bp, start_pose).then(
            command.SetAutopilot(command.FutureActor, True))
    ]
    vehicle_id = client.apply_batch_sync(batch)[0].actor_id

    # Find the vehicle and return the Actor instance.
    time.sleep(
        0.5)  # This is so that the vehicle gets registered in the actors.
    return world.get_actors().find(vehicle_id)


def spawn_rgb_camera(world, location, rotation, vehicle):
    """ This method spawns an RGB camera with the default parameters and the
    given location and rotation. It also attaches the camera to the given
    actor.

    Args:
        world: The world inside the current simulation.
        location: The Location instance representing the location where
          the camera needs to be spawned with respect to the vehicle.
        rotation: The Rotation instance representing the rotation of the
          spawned camera.
        vehicle: The Actor instance to attach the camera to.

    Returns:
        An instance of the camera spawned in the world.
    """
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    transform = Transform(location=location, rotation=rotation)
    return world.spawn_actor(camera_bp, transform, attach_to=vehicle)


def main():
    global world
    client = Client('localhost', 2000)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / 10
    world.apply_settings(settings)

    # Spawn the vehicle.
    vehicle = spawn_driving_vehicle(client, world)

    # Spawn the camera and register a function to listen to the images.
    camera = spawn_rgb_camera(world, Location(x=2.0, y=0.0, z=1.8),
                              Rotation(roll=0, pitch=0, yaw=0), vehicle)
    camera.listen(process_images)
    world.tick()
    return vehicle, camera, world


if __name__ == "__main__":
    vehicle, camera, world = main()
    try:
        while True:
            time.sleep(1 / 100.0)
            cv2.waitKey(1)
    except KeyboardInterrupt:
        # Destroy the actors.
        vehicle.destroy()
        camera.destroy()
