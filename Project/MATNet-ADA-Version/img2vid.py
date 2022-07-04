import os
import cv2
def create_video(img_folder_path="./data/vid_1/", generated_vid_path="./data/vid_1.avi", fps=5.0):
    file_names = [os.path.join(img_folder_path, f) for f in os.listdir(
        img_folder_path) if os.path.isfile(os.path.join(img_folder_path, f))]
    file_names.sort()
    frame_arr = []
    for file in file_names:
        img = cv2.imread(file)
        frame_arr.append(img)
    img_shape = (img.shape[1], img.shape[0])  # (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(generated_vid_path, fourcc, fps, img_shape)
    for i in range(len(frame_arr)):
        out.write(frame_arr[i])
    out.release()
    
if __name__ == "__main__":
    create_video()