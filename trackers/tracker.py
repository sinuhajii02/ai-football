from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
import numpy as np
import pandas as pd

sys.path.append('../')
from utils import get_center_box, get_width_box

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.batch_size = 20
        self.tracker = sv.ByteTrack()


    def detect_frames(self, frames):
       detections = []

       # Send in batch
       for i in range(0, len(frames), self.batch_size):
           detections_batch = self.model.predict(frames[i:i + self.batch_size], conf=0.1)
           detections += detections_batch

       return detections
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        

        detections = self.detect_frames(frames)
        tracks={
            "players": [],
            "referees" : [],
            "ball": []
        }
        for frame_num, detection in  enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            # Convert detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert goalie to player
            for index, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[index] = cls_names_inv["player"]


            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == cls_names_inv['player']:
                    tracks['players'][frame_num][track_id] = {"bbox":bbox}

                if class_id == cls_names_inv['referee']:
                    tracks['referees'][frame_num][track_id] = {"bbox":bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                if class_id == cls_names_inv['ball']:
                    tracks['ball'][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks
    
    def interpolate_ball_position(self, ball_positions):
        ball_position = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball = pd.DataFrame(data=ball_position, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball = df_ball.interpolate()

        df_ball = df_ball.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball.to_numpy().tolist()]

        return ball_positions
    

    def draw_circle(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])

        center = get_center_box(bbox)
        width = get_width_box(bbox=bbox) # radius of circle

        cv2.ellipse(frame, center=center, axes=(int(width), int(0.4 * width)), angle=0.0, startAngle=45, endAngle=260, color=color, thickness=2, lineType=cv2.LINE_4)
        

        rectangle_width = 40
        rectangle_height = 20

        x1_rect = center[0] - rectangle_width//2
        x2_rect = center[0] + rectangle_width//2
        y1_rect = (y2 - rectangle_height//2) + 15
        y2_rect = (y2 + rectangle_height//2) + 15
        

        if track_id is not None:
            cv2.rectangle(frame, (int(x1_rect), int(y1_rect)), (int(x2_rect), int(y2_rect)),  color, cv2.FILLED)


            #Visual
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10
            

            cv2.putText(frame, 
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_PLAIN,
                        0.6,
                        color=(0, 0, 0),
                        thickness=2
                        )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y=int(bbox[1])
        x, _= get_center_box(bbox)
        triangle_points = np.array([
            [x, y],
            [x-10, y-20],
            [x+10, y-20],
        ])

        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)


        return frame
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        # draw semi-transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)


        team_ball_control_till_frame = team_ball_control[:frame_num+1]

        # get the number of times each time has the ball
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        team_1_ball_pos = team_1_num_frames/(team_1_num_frames + team_2_num_frames)
        team_2_ball_pos = team_2_num_frames/(team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Possession: {team_1_ball_pos*100:.2f}%", (1370, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2 Ball Possession: {team_2_ball_pos*100:.2f}%", (1370, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

        
    # Create annotations around object
    def draw_circle_around(self, video_frames, tracks, team_ball_control):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Not pollute/changed the actual frame


            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]


            # Draw players
            for track_id, player in player_dict.items():
                color = player.get('team_color', (0, 0, 255))
                frame = self.draw_circle(frame, player['bbox'], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_triangle(frame, player['bbox'], (0, 0, 255))

            for track_id, referee in referee_dict.items():
                frame = self.draw_circle(frame, referee['bbox'], (0, 255, 255))

            # draw ball
            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball['bbox'], (0, 255, 0))



            # draw team statistics: ball possession
            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)
            output_video_frames.append(frame)

        return output_video_frames






