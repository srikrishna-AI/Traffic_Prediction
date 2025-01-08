import cv2
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import numpy as np
import torch

# Load YOLOv8 model
model = YOLO('https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8l.pt?download=true')

# Helper function for real-time object detection
def real_time_detection():
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model.predict(frame, conf=0.5, device="cuda" if torch.cuda.is_available() else "cpu")

        for result in results:  # Process results
            for box in result.boxes:  # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Decreased font size to 0.8

        # Display the frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        # Stop if the 'Stop Detection' button is clicked
        if st.session_state.get('stop_detection', False):
            break

    cap.release()
    st.session_state['stop_detection'] = False  # Reset the session state after stopping

# Helper function for video detection
def video_detection(uploaded_video):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    skip_frames = 1  # Skip every alternate frame to improve speed

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (skip_frames + 1) != 0:
            continue

        # Perform object detection
        results = model.predict(frame, conf=0.5, device="cuda" if torch.cuda.is_available() else "cpu")

        for result in results:  # Process results
            for box in result.boxes:  # Draw bounding boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 2)  # Constant font size

        # Display the frame
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    cap.release()

# Helper function for image detection
def picture_detection(uploaded_image):
    image = Image.open(uploaded_image)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = model.predict(frame, conf=0.5, device="cuda" if torch.cuda.is_available() else "cpu")

    for result in results:  # Process results
        for box in result.boxes:  # Draw bounding boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            label = f"{model.names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)  # Decreased font size to 0.8

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

# Streamlit UI
def main():
    st.title("Real-Time Traffic Prediction with YOLOv8")

    st.sidebar.title("Select Mode")
    mode = st.sidebar.radio("", ["Real-Time Detection", "Video Detection", "Picture Detection", "About Project"])

    if mode == "Real-Time Detection":
        # Initialize session state for stop detection
        if 'stop_detection' not in st.session_state:
            st.session_state['stop_detection'] = False

        if not st.session_state['stop_detection']:
            if st.button("Start Detection"):
                # Start real-time detection
                real_time_detection()
                st.session_state['stop_detection'] = True  # Set state to indicate detection has started
        else:
            # Show the "Stop Detection" button if detection has started
            stop_button = st.button("Stop Detection")
            if stop_button:
                st.session_state['stop_detection'] = False  # Stop detection when clicked

    elif mode == "Video Detection":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video is not None:
            video_detection(uploaded_video)

    elif mode == "Picture Detection":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            picture_detection(uploaded_image)

    elif mode == "About Project":
        st.write("""
           



### **Real-Time Traffic Prediction with YOLOv8**

This project is an implementation of real-time traffic prediction and object detection using the YOLOv8 (You Only Look Once) model, which is one of the most popular and efficient object detection algorithms in the computer vision domain. The project utilizes the YOLOv8 model to detect various objects in real-time, videos, and images through a user-friendly interface built with **Streamlit**.

---

### **Key Features**:
- **Real-Time Detection**: The application captures live video from the webcam, processes the frames, and uses YOLOv8 to detect objects. Detected objects are displayed with bounding boxes and labels, and the confidence score is shown.
- **Video Detection**: Users can upload video files (MP4, AVI, MOV) for object detection on each frame, making it ideal for analyzing videos with moving traffic or scenes containing multiple objects.
- **Image Detection**: Users can upload static images (JPG, JPEG, PNG) to detect objects within the image. The application provides an instant analysis of the uploaded image with detected object labels and bounding boxes.
- **Interactive UI**: Built using **Streamlit**, users can easily interact with the app to choose the mode (Real-Time, Video, or Image detection) and start/stop detection with a simple button interface.

---

### **Technologies Used**:
1. **YOLOv8 (You Only Look Once)**:  
   YOLOv8 is the latest version of the YOLO family of object detection models. YOLO models are known for their speed and accuracy, making them suitable for real-time applications like this one. YOLOv8 is designed to detect objects in a wide variety of scenarios with state-of-the-art precision.

2. **Streamlit**:  
   Streamlit is an open-source Python framework for building interactive, web-based applications. It's lightweight and extremely easy to use, making it perfect for building the UI for this object detection project.

3. **OpenCV**:  
   OpenCV (Open Source Computer Vision Library) is an open-source computer vision and machine learning software library. In this project, OpenCV is used for video capture, image processing, and handling real-time camera feeds.

4. **Ultralytics YOLOv8 Python Package**:  
   This is a Python package developed by Ultralytics that provides a simple interface for using YOLOv8 for object detection.

5. **Python**:  
   Python is the primary programming language used to develop the application, with dependencies installed through **pip** or **conda**.

---

### **Installation Instructions**:
To run this project, you need to install the following packages:

1. **YOLOv8**: The YOLOv8 model can be installed through the Ultralytics package.
   - You can install it using pip:
     ```bash
     pip install ultralytics
     ```

2. **Streamlit**: For creating the web interface.
   - Install Streamlit with pip:
     ```bash
     pip install streamlit
     ```

3. **OpenCV**: For video and image processing.
   - Install OpenCV with pip:
     ```bash
     pip install opencv-python
     ```

4. **Pillow**: For handling image files (JPEG, PNG).
   - Install Pillow with pip:
     ```bash
     pip install Pillow
     ```

5. **PyTorch**: YOLOv8 uses PyTorch for model inference, so you need to install it.
   - Install PyTorch with pip (you can choose the version based on your system configuration):
     ```bash
     pip install torch torchvision torchaudio
     ```

### **Model Version Reference**:
YOLOv8 can be easily downloaded and used by simply installing the **Ultralytics** package, which automatically downloads the latest version of YOLOv8. However, if you need a specific version of YOLOv8 (e.g., `yolov8n.pt`), you can access it from the official repository or directly from the Ultralytics model hub. 

For example, to use a smaller model (`yolov8n.pt` for lightweight detection):
```python
model = YOLO('yolov8n.pt')
```

To access other versions or more details on YOLOv8, visit the official GitHub repository:
- **Ultralytics YOLOv8 GitHub**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- **YOLOv8 Model Hub**: [https://github.com/ultralytics/yolov8/releases](https://github.com/ultralytics/yolov8/releases)

---

### **How the Project Works**:
1. **Real-Time Detection**: 
   - The webcam feed is captured using OpenCV's `cv2.VideoCapture()`.
   - YOLOv8 processes each frame, detecting objects in real-time.
   - Bounding boxes and labels are drawn on the detected objects, and the output is displayed via Streamlit.

2. **Video Detection**: 
   - Users upload a video file which is processed frame by frame.
   - YOLOv8 identifies objects in each frame, and the results are displayed on the Streamlit UI.

3. **Image Detection**:
   - Users can upload images to the application.
   - YOLOv8 detects and labels objects within the image, providing a visual representation of the objects in the uploaded image.

4. **User Interface**: 
   - The app is built with **Streamlit**, providing a simple and intuitive interface. Users can select the mode (real-time, video, or image), upload content, and see detection results live.

---

### **Project Benefits**:
- **Speed and Efficiency**: The YOLOv8 model is optimized for fast real-time detection while maintaining high accuracy.
- **Flexible**: The project supports various modes of detection — webcam, video, and image uploads — making it suitable for different types of object detection scenarios.
- **Scalable**: You can easily modify or scale the project to integrate more complex detection models or handle larger datasets.

---

### **References**:
- **YOLOv8 Official Repository**: [https://github.com/ultralytics/yolov8](https://github.com/ultralytics/yolov8)
- **Streamlit Documentation**: [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **OpenCV Documentation**: [https://opencv.org/](https://opencv.org/)
- **Ultralytics YOLOv8 Package Documentation**: [https://github.com/ultralytics/ultralytics/blob/main/README.md](https://github.com/ultralytics/ultralytics/blob/main/README.md)

---

### **GitHub Repository**:
For the source code and more details about this project, visit the GitHub repository:  
[**GitHub Repository**](https://github.com/YourGitHubUsername/Real-Time-Traffic-Prediction-YOLOv8)

---

**Developed by:** Sri Krishna Garishapati


        """)

if __name__ == "__main__":
    main()       