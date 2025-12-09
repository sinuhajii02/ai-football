from utils import read_video, save_video
from trackers import Tracker
import cv2
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np

from camera_movement import CameraMovementEstimator

def main():
    # Read video
    video_frames, fps = read_video('input_videos/B1606b0e6_1 (26).mp4')
    # Init tracker
    tracker = Tracker('models/best.pt')

    #Get object tracker
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')
    
    # Camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames, read_from_stub=True, stub_path="stubs/camera_movement_stub.pkl")


    #Interpolate ball positions
    tracks['ball'] = tracker.interpolate_ball_position(tracks['ball'])

    # Assign player teams
    teamAssigner = TeamAssigner()
    teamAssigner.assign_team_color(video_frames[0], tracks['players'][0])

    for frame_num, player_track in enumerate(tracks['players']):
        for player_id, track in player_track.items():
            team = teamAssigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['players'][frame_num][player_id]['team'] = team
            tracks['players'][frame_num][player_id]['team_color'] = teamAssigner.team_colors[team]
            
    # Player ball assigner
    playerBallAssigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = playerBallAssigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['players'][frame_num][assigned_player]['has_ball'] = True
            # Get the team of the player
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])

        else:
            # include the last person that has the ball
            team_ball_control.append(team_ball_control[-1])

    team_ball_control = np.array(team_ball_control)

    # Draw output video
    ## Draw object tracks
    output_video_frames = tracker.draw_circle_around(video_frames=video_frames, tracks=tracks, team_ball_control=team_ball_control)

    # Draw camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)
    # Save video 
    save_video(output_video_frames=output_video_frames, output_video_path='output_videos/output_video.mp4', fps=fps)

if __name__ == '__main__':
    main()