import cv2
import os


root_dir = r"E:\Dataset1"


video_exts = ('.mp4', '.avi', '.mov', '.mkv')


for dirpath, _, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename.lower().endswith(video_exts):
            input_path = os.path.join(dirpath, filename)

           
            rel_path = os.path.relpath(dirpath, root_dir)
            output_dir = os.path.join("videos_miroir", rel_path)
            os.makedirs(output_dir, exist_ok=True)

          
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_miroir{ext}")

            
            cap = cv2.VideoCapture(input_path)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                flipped = cv2.flip(frame, 1)
                out.write(flipped)

            cap.release()
            out.release()
            print(f"Miroir créé : {output_path}")