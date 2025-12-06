from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2

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

    def draw_circle(self, frame, bbox, color, track_id):
        y2 = int(bbox[3])

        center = get_center_box(bbox)
        width = get_width_box(bbox=bbox) # radius of circle

        cv2.ellipse(frame, center=center, axes=(int(width), int(0.4 * width)), angle=0.0, startAngle=45, endAngle=235, color=color, thickness=2, lineType=cv2.LINE_4)
        
        return frame



    # Create annotations around object
    def draw_circle_around(self, video_frames, tracks):
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy() # Not pollute/changed the actual frame


            player_dict = tracks['players'][frame_num]
            referee_dict = tracks['referees'][frame_num]
            ball_dict = tracks['ball'][frame_num]


            # Draw players

            for track_id, player in player_dict.items():
                frame = self.draw_circle(frame, player['bbox'], (0, 0, 255), track_id)

            output_video_frames.append(frame)

        return output_video_frames






