# Smart Shopping Glasses for the Visually Impaired - Project &nbsp;[![](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/) [![platform](https://img.shields.io/badge/platform-windows-green.svg)](https://github.com/xenon-19/Gesture_Controller)

A wearable assistive technology prototype designed to help visually impaired users shop independently. By combining real-time computer vision, spatial orientation tracking, and a dynamic web interface, the system identifies products, handles navigational feedback, and manages virtual retail transactions.

## 🔍 How It Works
1. **Vision & Sensing:** A high-definition miniature camera module continuously captures the user's field of view, while an Inertial Measurement Unit (IMU) tracks head orientation and spatial positioning.
2. **Edge AI Processing:** The video feed is processed locally on an ultra-compact single-board computer using optimized object detection models to identify items and checkout triggers (such as ArUco markers or products) instantly.
3. **Transaction Flow & Web UI:** The backend processes these detections to update a digital shopping cart. A dedicated web server routes the state to two separate web panels: a client-side shopping cart interface and an administrative teller interface for handling digital simulation workflows.

## 📁 Project Structure
* `main.py` - The primary Python backend script.
* `templates/cart.html` - The frontend user interface for the shopping cart.
* `templates/teller.html` - The frontend checkout interface for the teller.

## 🛠️ Hardware Components
The prototype is fully integrated into a wearable glasses frame using the following components:
* **Raspberry Pi Zero 2W:** The primary computational engine, acting as the centralized edge AI processor.
* **Smart Glasses Frame:** The physical wearable housing that mounts the camera, sensor array, and wiring discretely.
* **Pi Camera Rev 1.3:** A lightweight, 5MP camera module fixed to the bridge of the glasses to capture high-definition real-time point-of-view streams.
* **MPU-6050 Accelerometer/Gyroscope:** A 6-axis spatial motion sensor used to track orientation, head tilt, and stabilize navigation prompts.
* **3.7V LiPo Battery:** A compact, high-energy-density rechargeable battery providing wireless, tether-free power to the system.
* **Battery Shield:** A step-up power management module that securely boosts the 3.7V LiPo output to a stable 5V required by the Raspberry Pi hardware.
## ⚙️ Software & Dependencies (Raspberry Pi Zero 2W Optimization)
To run object detection efficiently on the resource-constrained Raspberry Pi Zero 2W (512MB RAM), the application utilizes an optimized stack focused on minimizing memory usage and maximizing inference throughput.

### Core Libraries
* **Flask:** The lightweight Python micro-framework used to serve the concurrent `cart.html` and `teller.html` interfaces.
* **OpenCV (`opencv-python-headless`):** Used for low-overhead image parsing and camera stream handling without loading graphical desktop UI layers.
* **Ultralytics YOLO:** Used to implement single-stage object detection pipelines.
* **NCNN / TensorFlow Lite Runtime:** Standard deployment optimization wrappers used to convert heavy PyTorch model weights into high-performance integer formats (e.g., INT8/NCNN format) to avoid CPU bottlenecks on ARM architecture.

### Hardware Interface Drivers
* **SMBus / `smbus2`:** Python libraries enabling I2C communication between the Pi Zero 2W and the MPU-6050 sensor registers.
* **`picamera2` / `rpicam-apps`:** Native Raspberry Pi camera drivers utilized to acquire high-speed video frames directly from the hardware pipeline.
