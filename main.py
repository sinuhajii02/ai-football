from utils import read_video, save_video
from trackers import Tracker

def main():
    # Read video
    video_frames = read_video('input_videos/B1606b0e6_1 (26).mp4')
    # Init tracker
    tracker = Tracker('models/best.pt')

    #Get object tracker
    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')

    # Save video
    save_video(video_frames, 'output_videos/output_video.mp4')

if __name__ == '__main__':
    main()