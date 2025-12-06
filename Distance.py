import math
#lengths should all be in meters
def calculate_distance(height, hypotenuse_length):
    distance = math.sqrt(math.pow(hypotenuse_length,2)-math.pow(height,2))
    return distance

def is_change_in_altitude(height, hypotenuse_length, initial_distance):
    correct_hypotenuse = 0
    correct_hypotenuse = math.sqrt(math.pow(height,2) + math.pow(initial_distance,2))
    difference_between = abs(hypotenuse_length - correct_hypotenuse)
    if (difference_between < 0.1):
        return False
    else:
        return True
def estimate_amount_of_steps(distance):
    amount = 0
    amount = distance/0.45
    return amount