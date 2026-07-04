import os
import re
import cv2
import ncnn
import numpy as np
import time
import threading
import queue
import sqlite3
import json
import pyttsx3
import sys
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
from picamera2 import Picamera2
import speech_recognition as sr

try:
    from mpu6050 import mpu6050
    HAS_MPU6050 = True
except ImportError:
    HAS_MPU6050 = False

# Fallback sequence matcher configuration
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# ── CONFIGURATIONS & PATHS ────────────────────────────────────────────────────────
PARAM_PATH      = "/home/tracynamayi/yolo/best_ncnn_model/model.ncnn.param"
BIN_PATH        = "/home/tracynamayi/yolo/best_ncnn_model/model.ncnn.bin"
DB_PATH         = "smart_shopping.db"
INPUT_SIZE      = 320
CONF_THRESH     = 0.45
NMS_THRESH      = 0.40
STREAM_PORT     = 8080

TARGET_FPS      = 12  
FRAME_INTERVAL  = 1.0 / TARGET_FPS

CLASSES = ["chillies", "citric", "sheasmooth", "niveafrash", "niveapearl",
           "applecider", "soysauce", "minivaseline"]
PALETTE = [
    (0,255,0),(255,100,0),(0,100,255),(255,0,180),
    (0,220,220),(180,0,255),(0,180,60),(255,220,0),
]

CLASS_TO_PRODUCT = {
    "chillies":     "Tropical heat Chillies spice",
    "citric":       "Tropical heat Citric Acid spice",
    "sheasmooth":   "Nivea Shea Smooth Lotion",
    "niveafrash":   "Nivea Fresh deodorant",
    "niveapearl":   "Nivea Pearl antiperspirant",
    "applecider":   "Apple Cider vinegar",
    "soysauce":     "Soy Sauce",
    "minivaseline": "Mini Vaseline Lotion",
}

PRODUCT_METADATA = {
    "chillies":     "Tropical heat Chillies spice. Used to add heat and rich flavor to culinary dishes.",
    "citric":       "Tropical heat Citric Acid spice. Acts as a natural preservative and souring agent.",
    "sheasmooth":   "Nivea Shea Smooth Lotion. Provides long-lasting skin moisturization using deep care serum.",
    "niveafrash":   "Nivea Fresh deodorant. Offers 48-hour odor protection infused with ocean extracts.",
    "niveapearl":   "Nivea Pearl antiperspirant. Provides smooth underarm protection with precious pearl extracts.",
    "applecider":   "Apple Cider vinegar. Assists in cooking, dressing, and health with five percent acidity.",
    "soysauce":     "Soy Sauce. Traditional savory seasoning adding rich umami profile elements.",
    "minivaseline": "Mini Vaseline Lotion. Heals dry skin and locks in vital health moisture.",
}

# ── THREAD-SAFE SHARED STATE ──────────────────────────────────────────────
infer_queue  = queue.Queue(maxsize=1)
result_lock  = threading.Lock()
frame_lock   = threading.Lock()
cart_lock    = threading.Lock()

latest_dets  = []
latest_frame = None  
infer_ms     = 0.0
digital_cart = []
session_id   = f"sess_{int(time.time())}"

app_state = {
    'running': True,
    'target_product': None,    
    'detected_match': False,  
    'matched_details': None,
    'status_msg': "Idle",
    'checkout_triggered': False  
}

# Real-time multi-modal navigation status state
navigation_state = {
    'active': False,
    'target_shelf': None,
    'target_class': None,
    'distance_status': "Far away",  
    'step_count': 0,
    'last_vocal_alert': 0
}

def get_ip_address():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<your-pi-ip>"

# ── MPU6050 SPATIAL telemetry ENGINE ──────────────────────────────────────
def mpu_navigation_worker(tts):
    if not HAS_MPU6050:
        print("[WARN] mpu6050 library missing. Physical step calculation falling back entirely to vision data.")
        return
    
    try:
        sensor = mpu6050(0x68)
        print("[INFO] MPU6050 Inertial Navigation Sensor linked successfully.")
    except Exception as e:
        print(f"[SENSOR ERROR] Failed to initialize MPU6050 at address 0x68: {e}")
        return

    last_accel_z = 0.0
    
    while app_state['running']:
        try:
            accel_data = sensor.get_accel_data()
            z_accel = accel_data['z']
            
            if navigation_state['active']:
                if abs(z_accel - last_accel_z) > 3.0:
                    navigation_state['step_count'] += 1
                    
                    if navigation_state['distance_status'] == "Far away" and navigation_state['step_count'] > 7:
                        navigation_state['distance_status'] = "Approaching"
                        
                last_accel_z = z_accel
            else:
                navigation_state['step_count'] = 0

        except Exception as err:
            print(f"[SENSOR READ ERROR]: {err}")
            
        time.sleep(0.1)

# ── TEXT MATCHING LOGIC ───────────────────────────────────────────────────
class ProductMatcher:
    def __init__(self, class_to_product_map):
        self.mapping = class_to_product_map

    def _normalize(self, text):
        text = text.lower().strip()
        text = text.replace("where is the", "").replace("where is", "").replace("where are the", "").replace("where are", "")
        text = re.sub(r'[^\w\s]', '', text).strip()
        return text

    def find_best_match(self, user_input, score_threshold=40):
        cleaned_input = self._normalize(user_input)
        if not cleaned_input: return None, 0

        if cleaned_input in ["checkout", "proceed", "proceed to checkout", "pay"]:
            return "CHECKOUT_COMMAND", 100

        if cleaned_input in self.mapping:
            return cleaned_input, 100

        choices = {key: self._normalize(val) for key, val in self.mapping.items()}
        if HAS_RAPIDFUZZ:
            result = process.extractOne(cleaned_input, choices, scorer=fuzz.token_set_ratio)
            if result and result[1] >= score_threshold:
                return result[2], result[1]
        else:
            best_key, best_score = None, 0.0
            for key, val in choices.items():
                score = difflib.SequenceMatcher(None, cleaned_input, val).ratio() * 100
                if score > best_score:
                    best_score = score
                    best_key = key
            if best_score >= score_threshold:
                return best_key, best_score

        return None, 0

# ── DATABASE & BUSINESS LOGIC ─────────────────────────────────────────────
def init_databases():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sections (
            section_id INTEGER PRIMARY KEY, section_name TEXT, aria_marker_id INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS products (
            product_id INTEGER PRIMARY KEY, product_name TEXT, price REAL, section_id INTEGER,
            FOREIGN KEY(section_id) REFERENCES sections(section_id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cart_session (
            session_id TEXT PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS digital_cart (
            cart_id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, product_id INTEGER,
            quantity INTEGER, added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(session_id, product_id), FOREIGN KEY(session_id) REFERENCES cart_session(session_id)
        )
    """)
    
    check = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    if check == 0:
        conn.executemany("INSERT INTO sections VALUES (?, ?, ?)", [
            (1, "Produce Aisle", 10), (2, "Cosmetics Shelf", 20), (3, "Condiments", 30)
        ])
        conn.executemany("INSERT INTO products VALUES (?, ?, ?, ?)", [
            (69800112, "Tropical heat Chillies spice", 110, 1),
            (68111233, "Tropical heat Citric Acid spice", 110, 1),
            (56677889, "Nivea Shea Smooth Lotion", 500, 2),
            (67779993, "Nivea Fresh deodorant", 449, 2),
            (33444433, "Nivea Pearl antiperspirant", 499, 2),
            (55553334, "Apple Cider vinegar", 389, 3),
            (33444887, "Soy Sauce", 219, 3),
            (22119988, "Mini Vaseline Lotion", 199, 2)
        ])
        conn.commit()
    conn.close()

def lookup_product(class_name):
    product_name = CLASS_TO_PRODUCT.get(class_name.lower())
    if not product_name: return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        """SELECT p.product_id, p.product_name, p.price, s.section_name
           FROM products p JOIN sections s ON p.section_id = s.section_id
           WHERE LOWER(p.product_name) = LOWER(?)""", (product_name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def add_to_db_cart(product):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT OR IGNORE INTO cart_session (session_id) VALUES (?)", (session_id,))
        conn.execute(
            """INSERT INTO digital_cart (session_id, product_id, quantity) VALUES (?, ?, 1)
               ON CONFLICT(session_id, product_id) DO UPDATE SET quantity = quantity + 1""",
            (session_id, product["product_id"])
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB ERROR] {e}")

# ── API REST ENDPOINTS ────────────────────────────────────────────────────
def json_response(handler, data, status=200):
    body = json.dumps(data).encode()
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        if path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body><h1>Smart Shopping Pi Server Engine</h1></body></html>")
            return

        if path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                with frame_lock:
                    frame = latest_frame
                if frame is not None:
                    _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    try:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.08)
            return

        if path == "/detections":
            with result_lock: dets = list(latest_dets)
            data = [{"class": CLASSES[cid], "confidence": round(conf, 2)} for (_, _, _, _, conf, cid) in dets]
            json_response(self, {"detections": data})
            return

        if path == "/cart":
            with cart_lock: items = list(digital_cart)
            total = sum(i["price"] * i["qty"] for i in items)
            formatted_cart = [{
                "product_id": item["product_id"], 
                "product_name": item["product_name"],
                "price": item["price"], 
                "qty": item["qty"], 
                "quantity": item["qty"], 
                "section_name": item.get("section_name", "General Aisle")
            } for item in items]
            
            json_response(self, {
                "session_id": session_id, 
                "cart": formatted_cart, 
                "total": round(total, 2),
                "checkout_triggered": app_state['checkout_triggered']
            })
            return

        self.send_response(404); self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/clear_checkout":
            app_state['checkout_triggered'] = False
            json_response(self, {"status": "reset_successful"})
            return
        self.send_response(404); self.end_headers()

# ── LIGHTWEIGHT TTS ────────────────────────────────────────────────────────
class SpeechEngine:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 165)  

    def speak(self, text):
        print(f"[Audio Output]: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

# ── AUDIO MICROPHONE INPUT FUNCTION ───────────────────────────────────
def listen_to_mic(recognizer, source):
    """
    Listens to the Pi microphone and returns the parsed text command.
    Uses the 16000Hz calibration parameters that matched your arecord test.
    """
    print("\n[Listening...] Speak your command now.")
    try:
        # Tweak phrase_time_limit if your navigation queries get cut off
        audio = recognizer.listen(source, timeout=10, phrase_time_limit=5)
        print("[Processing Voice Matrix...]")
        text_command = recognizer.recognize_google(audio)
        print(f"[Heard]: '{text_command}'")
        return text_command.strip()
    except sr.WaitTimeoutError:
        print("[System Status]: Listening timeout, no voice heard.")
        return ""
    except sr.UnknownValueError:
        print("[System Status]: Could not capture audio signatures cleanly.")
        return ""
    except sr.RequestError as e:
        print(f"[System Status]: API Network error; {e}")
        return ""

# ── NAVIGATION & INTERACTIVE CONTROL LOOP ──────────────────────────────────
def interactive_control_loop(tts, matcher):
    time.sleep(2.0)
    print("\n" + "="*60)
    print(" NAIVAS SMART TERMINAL - AUDIO VOICE NAVIGATION MODE")
    print(" Say: 'where is [item]' | 'proceed' to audit cart")
    print("="*60)
    
    # Initialize speech recognizer instance
    recognizer = sr.Recognizer()
    
    # Forcing native hardware parameters directly
    mic = sr.Microphone(sample_rate=16000)
    
    with mic as source:
        print("[INFO] Calibrating microphone noise floors...")
        recognizer.adjust_for_ambient_noise(source, duration=1.5)
        print("[INFO] Microphone Engine Calibration Complete.")
        tts.speak("System initialized. I am listening for your command.")

        while app_state['running']:
            # Capture data via voice instead of sys.stdin
            raw_input = listen_to_mic(recognizer, source)
            
            if not raw_input: continue
            if raw_input.lower() == 'exit':
                app_state['running'] = False
                break

            # Check if the user wants to audit/proceed with checkout processing
            if "proceed" in raw_input.lower() or "checkout" in raw_input.lower():
                with cart_lock:
                    items = list(digital_cart)
                
                if not items:
                    print("[SYSTEM LOG]: Cart session contains 0 items.")
                    tts.speak("Your cart is empty. Cannot execute checkout proceedings.")
                    continue

                print("\n--- ACTIVE BASKET AUDIT RECONCILIATION ---")
                total = 0.0
                for idx, item in enumerate(items):
                    item_cost = item['price'] * item['qty']
                    total += item_cost
                    print(f"[{idx+1}] {item['product_name']} x{item['qty']} - ${item_cost:.2f}")
                print(f"Aggregated Invoice Total: ${total:.2f}\n")

                tts.speak("Reviewing items currently logged in your shopping cart.")
                for idx, item in enumerate(items):
                    tts.speak(f"Item {idx+1}: {item['qty']} units of {item['product_name']}.")
                
                tts.speak(f"The net calculation totals {total:.2f} dollars. Say yes to confirm or no to abort.")
                
                # Listen specifically for confirmation voice prompt
                confirm = listen_to_mic(recognizer, source).lower()
                if any(kw in confirm for kw in ['yes', 'confirm', 'yep', 'proceed']):
                    print("[SYSTEM TRIGGER]: Transferring session matrix state directly to terminal checkout.")
                    tts.speak("Confirmed. Re-routing interface window to teller desk.")
                    with result_lock:
                        app_state['checkout_triggered'] = True
                        app_state['status_msg'] = "Checking Out..."
                else:
                    tts.speak("Transaction hold applied. Returning to tracking environment.")
                continue
                
            matched_class, score = matcher.find_best_match(raw_input, score_threshold=35)
            
            # Explicit capture pattern checking for product shelf navigation queries
            if any(kw in raw_input.lower() for kw in ["where is", "where are", "location", "shelf", "find"]):
                if matched_class and matched_class != "CHECKOUT_COMMAND":
                    product = lookup_product(matched_class)
                    if product:
                        navigation_state['active'] = True
                        navigation_state['target_shelf'] = product['section_name']
                        navigation_state['target_class'] = matched_class
                        navigation_state['distance_status'] = "Far away"
                        navigation_state['step_count'] = 0
                        navigation_state['last_vocal_alert'] = time.time()
                        
                        tts.speak(f"{product['product_name']} is in the {product['section_name']}. Start walking toward the aisle.")
                        
                        # ── STEP 1: SHELF NAVIGATION SUB-LOOP ──
                        while navigation_state['active'] and app_state['running']:
                            now = time.time()
                            if now - navigation_state['last_vocal_alert'] > 4.5:
                                tts.speak(f"Aisle tracking status: You are {navigation_state['distance_status']}.")
                                navigation_state['last_vocal_alert'] = now
                            
                            if navigation_state['distance_status'] == "Near Product":
                                tts.speak(f"Arrived at the {product['section_name']}. Shelf location reached.")
                                navigation_state['active'] = False
                                break
                            time.sleep(0.2)
                        
                        # ── STEP 2: SCANNING SUB-LOOP ──
                        tts.speak(f"Please hold the item up to the camera to scan and match.")
                        
                        with result_lock:
                            app_state['target_product'] = matched_class
                            app_state['detected_match'] = False
                            app_state['matched_details'] = product
                            app_state['status_msg'] = f"Scanning: {matched_class}"
                        
                        scan_attempt_start = time.time()
                        while not app_state['detected_match'] and app_state['running']:
                            if time.time() - scan_attempt_start > 6.0:
                                tts.speak(f"Still waiting. Please center the product label in front of the camera module.")
                                scan_attempt_start = time.time()
                            time.sleep(0.2)
                            
                        if app_state['detected_match']:
                            prod_details = app_state['matched_details']
                            tts.speak(f"Item scanned and verified successfully.")
                            
                            description = PRODUCT_METADATA.get(matched_class, "Details unavailable.")
                            tts.speak(description)
                            tts.speak(f"Price is {prod_details['price']} shillings. Do you want to add this item to your cart?")
                            
                            # Listen for confirmation to add to digital cart
                            confirm = listen_to_mic(recognizer, source).lower()
                            if any(kw in confirm for kw in ['yes', 'add', 'sure', 'yeah']):
                                tts.speak("Added to session.")
                                with cart_lock:
                                    existing = next((i for i in digital_cart if i["product_id"] == prod_details["product_id"]), None)
                                    if existing: 
                                        existing["qty"] += 1
                                    else: 
                                        digital_cart.append({**prod_details, "qty": 1})
                                add_to_db_cart(prod_details)
                            else:
                                tts.speak("Declined.")
                        
                        with result_lock:
                            app_state['target_product'] = None
                            app_state['detected_match'] = False
                            app_state['status_msg'] = "Idle"
                        continue

            if matched_class == "CHECKOUT_COMMAND":
                tts.speak("Proceeding to checkout terminal. Transferring item rows.")
                with result_lock:
                    app_state['checkout_triggered'] = True
                    app_state['status_msg'] = "Checking Out..."
                continue

            if not matched_class:
                tts.speak("Command unresolved. Please ask again.")
                continue

# ── NCNN MATRIX OPERATIONS WORKER THREAD ──────────────────────────────────
def inference_worker(net):
    global latest_dets, infer_ms

    while app_state['running']:
        try: 
            raw_frame = infer_queue.get(timeout=0.5)
        except queue.Empty: 
            continue

        t0 = time.perf_counter()
        mat = ncnn.Mat.from_pixels(raw_frame, ncnn.Mat.PixelType.PIXEL_RGB, INPUT_SIZE, INPUT_SIZE)
        mat.substract_mean_normalize([0,0,0],[1/255.0,1/255.0,1/255.0])

        ex = net.create_extractor()
        ex.set_light_mode(True)
        ex.input("in0", mat)
        _, out = ex.extract("out0")

        num_anchors = out.w
        boxes, scores, class_ids = [], [], []

        for i in range(num_anchors):
            cls_scores = [out[j * num_anchors + i] for j in range(4, out.h)]
            conf = max(cls_scores)
            if conf < CONF_THRESH: continue
            
            cid = int(np.argmax(cls_scores))
            cx = out[0 * num_anchors + i]; cy = out[1 * num_anchors + i]
            bw = out[2 * num_anchors + i]; bh = out[3 * num_anchors + i]
            
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            boxes.append([max(0, x1), max(0, y1), int(bw), int(bh)])
            scores.append(float(conf))
            class_ids.append(cid)

        new_dets = []
        if boxes:
            keep = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=CONF_THRESH, nms_threshold=NMS_THRESH)
            for idx in (keep.flatten() if len(keep) else []):
                x, y, bw, bh = boxes[idx]
                new_dets.append((x, y, bw, bh, scores[idx], class_ids[idx]))

                detected_class_str = CLASSES[class_ids[idx]]

                current_target = app_state['target_product']
                if current_target and detected_class_str == current_target:
                    app_state['detected_match'] = True

                if navigation_state['active'] and detected_class_str == navigation_state['target_class']:
                    bounding_volume_area = bw * bh
                    
                    if bounding_volume_area > 32000:
                        navigation_state['distance_status'] = "Near Product"
                    elif bounding_volume_area > 11000:
                        navigation_state['distance_status'] = "Approaching"

        with result_lock:
            latest_dets = new_dets
            infer_ms = (time.perf_counter() - t0) * 1000

# ── OVERLAY DESIGNS ───────────────────────────────────────────────────────
def draw_detections(frame, dets, cur_ms):
    current_target = app_state['target_product']
    for (x, y, w, h, conf, cid) in dets:
        class_str = CLASSES[cid]
        if current_target and class_str == current_target:
            color_bgr = (0, 255, 0)
            label = f"MATCH: {class_str} {conf:.2f}"
        else:
            r, g, b = PALETTE[cid % len(PALETTE)]
            color_bgr = (b, g, r)
            label = f"{class_str} {conf:.2f}"
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, 2)
        cv2.putText(frame, label, (x+2, max(y-4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_bgr, 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (INPUT_SIZE, 40), (0, 0, 0), -1)
    
    if navigation_state['active']:
        nav_label = f"NAV: {navigation_state['target_shelf']} | Status: {navigation_state['distance_status']}"
        cv2.putText(frame, nav_label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 140, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, f"Latency: {cur_ms:.1f}ms", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {app_state['status_msg']}", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

# ── PRINCIPAL SYSTEM LAUNCH ───────────────────────────────────────────────
def main():
    global latest_frame, latest_dets

    init_databases()
    
    net = ncnn.Net()
    net.opt.num_threads         = 2    
    net.opt.use_fp16_storage    = True
    net.opt.use_fp16_packed     = True
    net.opt.use_fp16_arithmetic = True
    net.opt.use_packing_layout  = True
    net.opt.use_vulkan_compute  = False
    net.load_param(PARAM_PATH)
    net.load_model(BIN_PATH)
    print("[INFO] Model mounted to matrix runtime pipeline.")

    threading.Thread(target=inference_worker, args=(net,), daemon=True).start()
    
    matcher = ProductMatcher(CLASS_TO_PRODUCT)
    tts_engine = SpeechEngine()
    
    threading.Thread(target=interactive_control_loop, args=(tts_engine, matcher), daemon=True).start()
    threading.Thread(target=mpu_navigation_worker, args=(tts_engine,), daemon=True).start()

    HTTPServer.allow_reuse_address = True
    threading.Thread(target=lambda: HTTPServer(("0.0.0.0", STREAM_PORT), Handler).serve_forever(), daemon=True).start()

    pi_ip = get_ip_address()
    print(f"[INFO] Backend active at: http://{pi_ip}:{STREAM_PORT}")

    cam = Picamera2()
    frame_us = int(1_000_000 / TARGET_FPS)
    config = cam.create_video_configuration(
        main={"size": (INPUT_SIZE, INPUT_SIZE), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_us, frame_us)}
    )
    cam.configure(config)
    cam.set_controls({"Sharpness": 1.5, "NoiseReductionMode": 1, "AeExposureMode": 0})
    cam.start()
    time.sleep(1.0)

    next_frame_time = time.monotonic()

    try:
        while app_state['running']:
            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)
            next_frame_time = time.monotonic() + FRAME_INTERVAL

            frame_rgb = cam.capture_array()

            try: 
                infer_queue.put_nowait(frame_rgb.copy())
            except queue.Full: 
                pass

            with result_lock:
                dets = list(latest_dets)
                cur_ms = infer_ms

            display_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            draw_detections(display_frame, dets, cur_ms)

            with frame_lock:
                latest_frame = display_frame

    except KeyboardInterrupt:
        pass
    finally:
        print("\nReleasing Camera Hardware Component Safely...")
        cam.stop()

if __name__ == "__main__":
    main()