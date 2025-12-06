from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video
    video_frames, fps = read_video('input_videos/B1606b0e6_1 (26).mp4')
    # Init tracker
    tracker = Tracker('models/best.pt')

    #Get object tracker
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Draw output video
    ## Draw object tracks
    output_video_frames = tracker.draw_circle_around(video_frames=video_frames, tracks=tracks)
    # Save video
    save_video(output_video_frames=output_video_frames, output_video_path='output_videos/output_video.mp4', fps=fps)

if __name__ == '__main__':
    main()