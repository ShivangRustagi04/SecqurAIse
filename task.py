import cv2
import numpy as np
import time

def define_quadrants(frame):
    height, width, _ = frame.shape
    quadrants = {
        1: [(0, 0), (width//2, height//2)],
        2: [(width//2, 0), (width, height//2)],
        3: [(0, height//2), (width//2, height)],
        4: [(width//2, height//2), (width, height)]
    }
    return quadrants

def get_quadrant(x, y, quadrants):
    for quadrant, ((x1, y1), (x2, y2)) in quadrants.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return quadrant
    return None

def detect_colored_balls(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'green': ([35, 40, 40], [85, 255, 255]),  # Adjusted Green ball range
        'white': ([0, 0, 180], [180, 30, 255]),   # Adjusted White ball range
        'orange': ([5, 150, 150], [15, 255, 255]),# Adjusted Orange ball range
        'yellow': ([20, 100, 100], [30, 255, 255])# Adjusted Yellow ball range
    }

    ball_positions = []

    for color, (lower, upper) in color_ranges.items():
        lower = np.array(lower)
        upper = np.array(upper)
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Adjusted area threshold for better detection
                x, y, w, h = cv2.boundingRect(cnt)
                ball_positions.append((x + w // 2, y + h // 2, color))
                cv2.circle(frame, (x + w // 2, y + h // 2), 10, (0, 255, 0), 2)

    return ball_positions

def track_balls(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    quadrants = None
    start_time = time.time()
    ball_last_position = {}
    event_log = []

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if quadrants is None:
            quadrants = define_quadrants(frame)

        ball_positions = detect_colored_balls(frame)

        for (x, y, color) in ball_positions:
            quadrant = get_quadrant(x, y, quadrants)
            if color not in ball_last_position:
                ball_last_position[color] = quadrant
                continue

            if ball_last_position[color] != quadrant:
                if ball_last_position[color] is not None:
                    event_type = 'Exit'
                    event_time = time.time() - start_time
                    event_log.append((event_time, ball_last_position[color], color, event_type))
                if quadrant is not None:
                    event_type = 'Entry'
                    event_time = time.time() - start_time
                    event_log.append((event_time, quadrant, color, event_type))
                
                ball_last_position[color] = quadrant

                cv2.putText(frame, f"{event_type} - {color} - {quadrant}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame, f"Time: {event_time:.2f}s", (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        out.write(frame)  # Write the frame to the output video file
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open('event_log.txt', 'w') as f:
        for log in event_log:
            f.write(f"{log[0]:.2f}, {log[1]}, {log[2]}, {log[3]}\n")

if __name__ == "__main__":
    video_path = "AI Assignment video.mp4"
    output_path = "video.avi"
    track_balls(video_path, output_path)
