
# Real-Time Traffic Detection

This project is a real-time object detection application built using [YOLOv8](https://docs.ultralytics.com/models/yolov8) and [Streamlit](https://streamlit.io/). It uses a webcam feed to perform object detection, highlighting detected objects with bounding boxes and their confidence scores.

---

## Features

- **Real-Time Detection**: Detect objects in live video from your webcam.  
- **Video Detection**: Upload a video file for object detection.  
- **Image Detection**: Detect objects in uploaded images.  
- **User-Friendly Interface**: Simple and interactive UI using Streamlit.  
- **Fullscreen Support**: Switch to fullscreen mode during real-time detection for better visibility.  

---

## Tech Stack

- **YOLOv8**: State-of-the-art object detection model.
- **Streamlit**: For building an interactive web-based application.
- **OpenCV**: For processing video and images.
- **Python**: The core programming language.

---

## Setup and Installation

### Prerequisites
1. Python 3.8+
2. Basic knowledge of Python and Streamlit.

### Installation Steps

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/real-time-object-detection.git
   cd real-time-object-detection
   ```

2. **Install Required Libraries**  
   Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   > Ensure you have the `ultralytics` package installed to use YOLOv8.

3. **Download YOLOv8 Model**  
   The `yolov8n.pt` model is loaded automatically by the `ultralytics` package. If needed, you can download other YOLOv8 models from the [Ultralytics YOLOv8 Models Page](https://github.com/ultralytics/ultralytics).

4. **Run the Application**  
   Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**  
   Open your browser and navigate to `http://localhost:8501`.

---

## Usage

1. **Real-Time Detection**:
   - Select "Real-Time Detection" from the sidebar.
   - Click the "Start Detection" button.
   - Use the fullscreen button in the top-right corner of the detection frame for better visibility.
   - To stop detection, click the "Stop Detection" button.

2. **Video Detection**:
   - Upload a video file in formats like `.mp4`, `.avi`, or `.mov`.
   - The application will process the video and display results.

3. **Image Detection**:
   - Upload an image in formats like `.jpg`, `.jpeg`, or `.png`.
   - Detected objects will be displayed with bounding boxes and labels.

---

## Project Structure

```plaintext
real-time-object-detection/
├── app.py                 # Main Streamlit application file
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Files to ignore in Git
```

---

## Fullscreen Support

This project includes fullscreen functionality for better viewing during real-time detection. To enable fullscreen:
1. Click the "Fullscreen" button in the detection frame.
2. Press the **Escape** key to exit fullscreen mode.

---

## Future Improvements

- Add support for saving the detection results.
- Integrate GPU-based acceleration for faster processing.
- Include additional features like custom object detection models.

---

## Contributing

Contributions are welcome! If you have ideas for improving this project, please open an issue or submit a pull request.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) by Ultralytics for the object detection model.
- [Streamlit](https://streamlit.io/) for the easy-to-use UI framework.
- [OpenCV](https://opencv.org/) for image and video processing.

