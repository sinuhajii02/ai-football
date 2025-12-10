# Football Clip DETECTION + TRACKER using YOLO model🎯

This project performs **player tracking**, **player detection**, **team assigment** , and **camera movement estimation** on football
match footage. It outputs processed video frames with visual overlays for analysis.

<br>


## 🟢 Features

### Player Detection
Uses a computer vision pre-trained model YOLO, to detect players in each frame.

### Player Tracking
Assigns consistent IDs to players accross frames

### Team Assignment
Cluster players by shirt color to assign them to teams.

### (❗️CURRENTLY DOING ALMOST DONE) -> Camera Movement Estimation
Tracks feature points near the edges of the frame to estimate horizontal and vertical camer movement.

### Basic Match Analysis
Include simple metrics such as:
- Ball possesion for each team in that particular footage (number of frames)

<br>


## 🔴 Future Feature Ideas

### Player Trails
Show a short trajectory (e.g. 20 frames) behind each player to see player's movmenet patterns, direction, acceleration.

### Heatmap
Generate heatmap for a player/ball/posession zones

### Automatic Offside Line Detection
Detect last defender and draw the offside line per frame

### Tactical Detection
Detect team formations, passing lanes, pressing intensity

### Highlight Dangerous Events
Detect shots, crosses, through balls, dribbles and highlight them in the video timeline
