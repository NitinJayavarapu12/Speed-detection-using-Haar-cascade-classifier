class speedEstimator:
    def __init__(self, distance_between_points, frame_rate):
        self.distance = distance_between_points  # in meters
        self.frame_rate = frame_rate  # in frames per second
        self.prev_position = {}

    def estimate(self, vehicle_id, current_x, width):
        pixels_per_meter = width / self.distance

        if vehicle_id not in self.prev_position:
            self.prev_position[vehicle_id] = current_x
            return 0.0  # Speed cannot be calculated on first detection
        
        speed_pixels_per_sec = abs(current_x - self.prev_position[vehicle_id]) * self.frame_rate
        speed_mps = speed_pixels_per_sec / pixels_per_meter
        speed_kmph = speed_mps * 3.6

        self.prev_position[vehicle_id] = current_x
        return speed_kmph