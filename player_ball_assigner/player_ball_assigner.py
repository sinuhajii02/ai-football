import sys
sys.path.append('../')
from utils import get_center_box, measure_distance


class PlayerBallAssigner():
    def __init__(self):
        self.threshold_distance = 70
    
    def assign_ball_to_player(self, players, ball_box):
        ball_position = get_center_box(ball_box)
        minimum_dist = 999999
        assigned_player_id = -1

        for player_id, player in players.items():
            player_bbox = player['bbox']

            distance_left = measure_distance((player_bbox[0], player_bbox[-1]), ball_position)

            distance_right = measure_distance((player_bbox[2], player_bbox[-1]), ball_position)
            min_distance = min(distance_left, distance_right)

            if(min_distance < self.threshold_distance):
                if min_distance < minimum_dist:
                    minimum_dist = min_distance
                    assigned_player_id = player_id

        return assigned_player_id

            
