import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import PIL.Image
import easyocr
import requests
import re
import joblib
import cv2
from datetime import datetime

# --- 1. App Configuration & Advanced Styling ---
st.set_page_config(page_title="AI Vehicle Appraiser Pro", layout="centered", initial_sidebar_state="collapsed")

# The Bulletproof Emerald Stealth CSS
st.markdown("""
    <style>
@import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@700&family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Edge-to-Edge Cyber-Grid Background */
    .stApp {
        background-color: #020604;
        background-image: 
            radial-gradient(circle at 50% 0%, rgba(0, 230, 118, 0.15) 0%, transparent 60%),
            linear-gradient(rgba(0, 230, 118, 0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 230, 118, 0.07) 1px, transparent 1px);
        background-size: 100% 100%, 30px 30px, 30px 30px;
        color: #e2e8f0;
    }
    
    #MainMenu, footer, header {visibility: hidden;}

/* GLOBAL HEADERS (Forces the sharp font on everything) */
    h1, h2, h3, h4 {
        font-family: 'Syncopate', sans-serif !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* THE PREMIUM MAIN TITLE */
    .premium-title {
        text-align: center;
        font-size: 2.8rem !important;
        font-weight: 700;
        margin-top: 1rem;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
        /* Metallic White Gradient */
        background: linear-gradient(135deg, #ffffff 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* The Neon Green Highlight Word */
    .premium-title span {
        background: linear-gradient(135deg, #00e676 0%, #00994d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(0, 230, 118, 0.4);
    }
    
    .premium-subtitle {
        text-align: center;
        font-family: 'Inter', sans-serif;
        color: #00e676;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 3px;
        text-transform: uppercase;
        opacity: 0.8;
        margin-top: 5px;
        margin-bottom: 2rem;
    }

    /* 2. The Button (Base State) */
    .stButton>button {
        background: rgba(0, 230, 118, 0.03) !important; /* Extremely subtle green tint */
        color: #00e676 !important; /* Make text green by default */
        border: 1px solid #00e676 !important; /* The !important tag nukes the default red */
        border-radius: 2px !important;
        height: 3.5em !important;
        font-family: 'Syncopate', sans-serif !important;
        font-size: 0.9rem !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 12px rgba(0, 230, 118, 0.15) !important;
    }
    
    /* 3. The Button (Hover State) */
    .stButton>button:hover {
        background: #00e676 !important; 
        color: #020302 !important; /* Dark text on neon background */
        box-shadow: 0 0 30px rgba(0, 230, 118, 0.6) !important;
        border-color: #00e676 !important;
        transform: translateY(-2px);
    }

    /* 4. The Button (Kill Streamlit's Red Focus Outline) */
    .stButton>button:focus:not(:active) {
        border-color: #00e676 !important;
        color: #00e676 !important;
        box-shadow: 0 0 15px rgba(0, 230, 118, 0.3) !important;
    }

    /* 5. Glassmorphism Cards */
    .glass-card {
        background: rgba(6, 13, 9, 0.7); /* Deep dark green-grey glass */
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 230, 118, 0.1); /* Subtle green border */
        border-radius: 4px;
        padding: 24px;
        position: relative;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.9);
    }
    
    .glass-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00e676, transparent);
    }
    
    .price-text {
        color: #00e676; 
        font-family: 'Syncopate', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 5px 0;
        text-shadow: 0 0 25px rgba(0, 230, 118, 0.3);
    }
    .sub-text {
        color: #00e676; 
        text-transform: uppercase;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin: 0;
        opacity: 0.7;
    }
    
    .welcome-dash {
        text-align: center;
        padding: 50px 20px;
        background: rgba(0, 230, 118, 0.03);
        border: 1px solid rgba(0, 230, 118, 0.15);
        border-left: 3px solid #00e676; 
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- THE PREMIUM HEADER ---
st.markdown("""
    <div style="text-align: center;">
        <h1 class="premium-title">AI <span>VEHICLE</span> APPRAISER</h1>
        <p class="premium-subtitle">Computer Vision & RDW Telemetry</p>
    </div>
""", unsafe_allow_html=True)

VISION_CLASSES = [
    'AUDI', 'BMW', 'FIAT', 'FORD', 'HYUNDAI', 
    'MAZDA', 'MERCEDES-BENZ', 'NISSAN', 'SUZUKI', 
    'TESLA', 'TOYOTA', 'VOLKSWAGEN', 'VOLVO'
]

# --- 2. Resource Initialization ---
@st.cache_resource
def load_resources():
    # RDW CSV Data
    df = pd.read_csv("rdw_market_data.csv")
    df['datum_eerste_toelating'] = df['datum_eerste_toelating'].astype(str)
    df['year'] = pd.to_datetime(df['datum_eerste_toelating'], format='%Y%m%d', errors='coerce').dt.year
    df['age'] = 2026 - df['year']
    df['merk_clean'] = df['merk'].str.strip().str.upper()

    v_model = tf.keras.models.load_model('pro_vision_model.keras')
    reader = easyocr.Reader(['en'], gpu=False) 
    
    xgb_model = joblib.load('xgboost_pricing_model.pkl')
    brand_encoder = joblib.load('brand_encoder.pkl')
    
    return df, v_model, reader, xgb_model, brand_encoder

df_market, vision_model, ocr_reader, xgb_model, brand_encoder = load_resources()

# --- 3. Live API Function ---
def get_live_rdw_data(kenteken):
    url = f"https://opendata.rdw.nl/resource/m9d7-ebf2.json?kenteken={kenteken}"
    response = requests.get(url)
    if response.status_code == 200 and len(response.json()) > 0:
        return response.json()[0]
    return None

# --- THE ALPR CROPPER (CASCADE LIGHTING UPGRADE) ---
def isolate_dutch_plate(img_pil):
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2HSV)
    img_h, img_w = img_cv.shape[:2]
    
# The Cascade: [Strict Sunlight, Moderate Shadow, Heavy Overcast/Dirt]
    thresholds = [
        (np.array([15, 120, 120]), np.array([35, 255, 255])), 
        (np.array([10, 80, 70]), np.array([40, 255, 255])),   
        (np.array([10, 50, 50]), np.array([45, 255, 255]))    
    ]
    
    # NEW: A wide rectangular brush (25 pixels wide, 5 tall) to bridge the black dashes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    # Standard square brush to clean up random background noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    for lower, upper in thresholds:
        mask = cv2.inRange(hsv, lower, upper)
        
        # 1. 'Close' the horizontal gaps first
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        # 2. 'Open' to remove tiny specks of noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_crop = None
        best_score = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 400: # Ignore tiny specks
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / float(h)
                solidity = area / (w * h)
                
                # Strict shape heuristics
                if 2.0 <= aspect_ratio <= 7.0 and solidity > 0.45 and y > (img_h * 0.15):
                    aspect_score = 1 - (abs(4.5 - aspect_ratio) / 4.5)
                    score = area * aspect_score
                    
                    if score > best_score:
                        best_score = score
                        # Generous padding to prevent OCR suffocation!
                        y1, y2 = max(0, y - 15), min(img_h, y + h + 15)
                        x1, x2 = max(0, x - 40), min(img_w, x + w + 40)
                        best_crop = img_cv[y1:y2, x1:x2]
        
        # IF we found the plate in this lighting pass, STOP searching and return it!
        if best_crop is not None:
            return cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
            
    return None # Only return None if ALL THREE passes fail

# --- 4. Main Interface ---
st.markdown("Upload a photo of a vehicle. For the most accurate valuation, ensure the **license plate** is visible.")
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

# --- 5. Core Logic Pipeline ---
if uploaded_file is not None:
    img = PIL.Image.open(uploaded_file)
    st.image(img, caption='Input Stream', use_container_width=True)
    
    if st.button("Run AI Appraisal", use_container_width=True):
        
        with st.status("Initializing AI Vision Pipeline...", expanded=True) as status:
            st.write("📸 Capturing image stream...")
            st.write("🎯 Running Cascade ALPR algorithms...")
            
            exact_match = False
            target_brand = None
            specific_model = "Unknown Model"
            detected_kenteken = None
            confidence = 0.0
            price_display = "Data Unavailable"
            catalog_price_display = "N/A"
            strategy = "RDW Match (Valuation Unavailable)" 
            estimated_value = 0 
            
            
            # --- STAGE 1: THE ALPR SNIPER ---
            plate_crop_img = isolate_dutch_plate(img)
            
            if plate_crop_img is not None:
                st.image(plate_crop_img, caption="ALPR Target Acquired", width=300)
                ocr_target = plate_crop_img
            else:
                ocr_target = np.array(img.convert('RGB')) 
                
            # --- AGGRESSIVE OCR PRE-PROCESSING ---
            if plate_crop_img is not None:
                # 1. Convert to Grayscale
                gray_target = cv2.cvtColor(np.array(ocr_target), cv2.COLOR_RGB2GRAY)
                # 2. Apply a Sharpening Kernel to crisp up the edges of the 'N'
                sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
                crisp_target = cv2.filter2D(gray_target, -1, sharpen_kernel)
            else:
                crisp_target = ocr_target
                
            # 3. Read text with a strict allowlist and higher magnification!
            ocr_results = ocr_reader.readtext(crisp_target, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-', mag_ratio=3)
            # --------------------------------------
            
            # The Text Parser (Keeping your essential logic!)
            all_text = "".join([text for (bbox, text, prob) in ocr_results])
            clean_string = re.sub(r'[^A-Z0-9]', '', all_text.upper().replace('NL', '').replace('O', '0').replace('Q', '0').replace('I', 'T'))
            
            st.write(f"🔤 OCR Text Parsed: {clean_string}")
            
            # The Bouncer checks every 6-char chunk
            candidates = [clean_string[i:i+6] for i in range(len(clean_string) - 5)]
            for candidate in candidates:
                # Dutch rule: No Vowels, no C/Q, must have numbers + letters
                if not any(char in 'AEIOUCQ' for char in candidate) and \
                any(char.isdigit() for char in candidate) and \
                any(char.isalpha() for char in candidate):

                    live_rdw = get_live_rdw_data(candidate)
                    
                    if not live_rdw:
                        look_alikes = {"6": "G", "G": "6", "0": "O", "O": "0", "8": "B", "B": "8", "5": "S", "S": "5"}
                        for i, char in enumerate(candidate):
                            if char in look_alikes:
                                alt_candidate = candidate[:i] + look_alikes[char] + candidate[i+1:]
                                alt_rdw = get_live_rdw_data(alt_candidate)
                                if alt_rdw:
                                    live_rdw = alt_rdw
                                    st.toast(f"Auto-Corrected Plate: {candidate} ➡️ {alt_candidate}", icon="🪄")
                                    st.write(f"🪄 Heuristic Correction Applied: {candidate} ➡️ {alt_candidate}")
                                    candidate = alt_candidate 
                                    break 

                    if live_rdw:
                        exact_match = True
                        detected_kenteken = candidate
                        target_brand = live_rdw.get('merk', '').strip().upper()
                        
                        # 1. Grab the Year FIRST
                        toelating = live_rdw.get('datum_eerste_toelating', '')
                        reg_year_str = ""
                        if toelating:
                            reg_year = int(str(toelating)[:4])
                            car_age = 2026 - reg_year
                            reg_year_str = f" [{reg_year}]" # Formats it like [2018]
                            
                        # --- THE MINIMALIST ENGINE GENERATOR ---
                        base_model = live_rdw.get('handelsbenaming', 'Unknown').strip()
                        engine_cc = live_rdw.get('cilinderinhoud', '')
                        
                        engine_info = ""
                        if engine_cc and engine_cc.isdigit():
                            liters = round(int(engine_cc) / 1000, 1)
                            engine_info = f" • {liters}L"
                            
                        
                        specific_model = f"{base_model}{reg_year_str}{engine_info}"
                        # ---------------------------------------
                        
                        body_type = live_rdw.get('inrichting', 'Unknown')
                        color = live_rdw.get('eerste_kleur', 'Unknown')
                        weight = live_rdw.get('massa_ledig_voertuig', 'Unknown')
                        catalog_price = float(live_rdw.get('catalogusprijs', 0))
                        
                        # Grab Seats
                        seats = live_rdw.get('aantal_zitplaatsen', 'N/A')
                        
                        # Grab and format the APK Date (From YYYYMMDD to DD-MM-YYYY)
                        # --- DYNAMIC APK TRAFFIC LIGHT SYSTEM ---
                        apk_raw = live_rdw.get('vervaldatum_apk', '')
                        apk_color = "#e2e8f0" # Default White
                        apk_date = "Unknown"
                        
                        if len(apk_raw) == 8:
                            apk_date = f"{apk_raw[-2:]}-{apk_raw[4:6]}-{apk_raw[:4]}"
                            try:
                                # Convert RDW string to a real time object
                                apk_obj = datetime.strptime(apk_raw, "%Y%m%d")
                                days_left = (apk_obj - datetime.now()).days
                                
                                if days_left < 0:
                                    apk_color = "#ef233c" # RED (Expired)
                                    apk_date += " ⚠️"     # Add a warning icon!
                                elif days_left <= 90:
                                    apk_color = "#ffb703" # ORANGE (Expiring Soon)
                                else:
                                    apk_color = "#00e676" # GREEN (Safe)
                            except:
                                pass
                            
                        confidence = 1.0 
                        break

            # --- STAGE 2: VISION AI FALLBACK ---
            if not exact_match:
                img_resized = img.convert("RGB").resize((224, 224))
                img_array = image.img_to_array(img_resized)
                img_tensor = np.expand_dims(img_array, axis=0)
                
                preds = vision_model.predict(img_tensor)
                pred_index = np.argmax(preds)
                target_brand = VISION_CLASSES[pred_index].strip().upper()
                confidence = np.max(preds)
                
                # THE PRO SAFETY LOCK: Raised to 40%
                if confidence < 0.40:
                    # Shut down the terminal in an ERROR state
                    status.update(label="Appraisal Aborted: Insufficient Data", state="error", expanded=True)
                    st.divider()
                    st.error("🚨 Identification Failed")
                    
                    # Dynamic Error UX
                    if plate_crop_img is not None:
                        st.warning(f"License plate was cropped, but the OCR misread it or the RDW database timed out. Vision AI confidence is also too low ({confidence:.1%}).")
                    else:
                        st.warning(f"No license plate found, and the AI cannot confidently recognize the vehicle's features (Confidence: {confidence:.1%}).")
                        
                    st.stop() # Halts execution

                # If it passes 40%, proceed with the fallback valuation
                brand_data = df_market[df_market['merk_clean'] == target_brand]
                if not brand_data.empty:
                    val_median = brand_data['catalogusprijs'].median()
                    price_display = f"€{val_median:,.2f} (Avg)"
                strategy = "Vision AI + Brand Median Fallback"

            # --- STAGE 3: XGBOOST VALUATION (If ALPR hit) ---
            if exact_match and catalog_price > 0:
                try:
                    brand_encoded = brand_encoder.transform([target_brand])[0]
                    features = pd.DataFrame({
                        'merk_encoded': [brand_encoded],
                        'age': [car_age],
                        'catalogusprijs': [catalog_price]
                    })
                    estimated_value = xgb_model.predict(features)[0]
                    price_display = f"€{estimated_value:,.2f}"
                    catalog_price_display = f"€{catalog_price:,.2f}"
                    strategy = "XGBoost Regressor (Exact Spec)"
                except Exception as e:
                    price_display = "Engine Error"
                    strategy = "Calculation Failed"
            # --- STAGE 4: UI OUTPUT RENDERING ---
            
            status.update(label="Appraisal Complete!", state="complete", expanded=True)
        st.divider()
        
        # --- THE FERRARI EASTER EGG INTERCEPTOR ---
        if target_brand == "FERRARI":
            price_display = "DREAM ON! 🏎️💨"
            catalog_price_display = "IF YOU HAVE TO ASK..."
            strategy = "Reality Check Protocol"
            english_color = "Rosso Corsa (Probably)"
        
        if exact_match:
            # 1. THE COLOR TRANSLATOR
            color_mapping = {
                "ZWART": "Black", "WIT": "White", "GRIJS": "Grey", "ROOD": "Red", 
                "BLAUW": "Blue", "GROEN": "Green", "GEEL": "Yellow", "ORANJE": "Orange", 
                "BRUIN": "Brown", "PAARS": "Purple", "ROZE": "Pink", "BEIGE": "Beige", 
                "CREME": "Cream", "DIVERSEN": "Custom"
            }
            english_color = color_mapping.get(color, color).title()

            # --- THE FERRARI EASTER EGG ---
            if target_brand == "FERRARI":
                price_display = "DREAM ON! 🏎️💨"
                catalog_price_display = "IF YOU HAVE TO ASK..."
                strategy = "Reality Check Protocol"
                # Color joke
                if english_color == "Red":
                    english_color = "Rosso Corsa (Obviously)"
                elif english_color == "Yellow":
                    english_color = "Giallo Modena (Respect.)"
                else:
                    english_color = f"{english_color} (Should have bought Red...)"

            # 2. RENDER THE UI HEADERS
            st.markdown(f"## {target_brand} <span style='color:#00e676'>{specific_model}</span>", unsafe_allow_html=True)
            st.success(f"🟢 RDW Telemetry Synced: {detected_kenteken}")
        else:
            st.subheader(f"Identified Brand: {target_brand}")
            st.warning(f"⚠️ License plate unreadable. Using visual estimate ({confidence:.1%} confidence).")
        
        # Calculate Depreciation Delta
        depreciation_html = "<span style='color:#64748b'>N/A</span>"
        if exact_match and catalog_price > 0 and estimated_value > 0 and target_brand != "FERRARI":
            depr_pct = ((catalog_price - estimated_value) / catalog_price) * 100
            depreciation_html = f"<span style='color:#ef233c'>-{depr_pct:.1f}%</span>"

        # 3-Column Glass Cards
        col1, col2, col3 = st.columns(3)
        
        if target_brand == "FERRARI":
            # --- FERRARI EASTER EGG UI (Glowing Red & Silver) ---
            with col1:
                st.markdown(f"""
                <div class='glass-card'>
                    <p class='sub-text'>Market Valuation</p>
                    <p class='price-text' style='color: #ef233c; text-shadow: 0 0 25px rgba(239, 35, 60, 0.4);'>{price_display}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='glass-card' style='border-left-color: #4a5568;'>
                    <p class='sub-text'>Original List Price</p>
                    <p class='price-text' style='color: #cbd5e1; text-shadow: none;'>BOTH YOUR KIDNEYS 🧬</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='glass-card' style='border-left-color: #00e676;'>
                    <p class='sub-text'>Value Lost</p>
                    <p class='price-text' style='text-shadow: none;'>{depreciation_html}</p>
                </div>
                """, unsafe_allow_html=True)
                
        else:
            # --- STANDARD NORMAL CAR UI (Neon Green & White) ---
            with col1:
                st.markdown(f"""
                <div class='glass-card'>
                    <p class='sub-text'>Market Valuation</p>
                    <p class='price-text'>{price_display}</p>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='glass-card' style='border-left-color: #4a5568;'>
                    <p class='sub-text'>Original List Price</p>
                    <p class='price-text' style='color: #ffffff; text-shadow: none;'>{catalog_price_display}</p>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='glass-card' style='border-left-color: #00e676;'>
                    <p class='sub-text'>Value Lost</p>
                    <p class='price-text' style='text-shadow: none;'>{depreciation_html}</p>
                </div>
                """, unsafe_allow_html=True)
        
        if exact_match:
                weight_str = f"{weight} kg" if weight != 'Unknown' else weight
                
                # The fully integrated HTML Flexbox Databank (FLUSH LEFT)
                databank_html = f"""
<div class='glass-card' style='margin-top: 20px; border-left: 3px solid #00e676;'>
    <h3 style='margin-top:0; margin-bottom: 20px; color: #00e676; font-size: 1.2rem; font-family: "Syncopate", sans-serif; text-transform: uppercase;'> Vehicle Telemetry Databank</h3>
    <div style='display: flex; justify-content: space-between; flex-wrap: wrap;'>
        <div style='text-align: left;'>
            <p class='sub-text' style='margin-bottom: 5px;'>BODY TYPE</p>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #e2e8f0;'>{body_type.capitalize()}</p>
        </div>
        <div style='text-align: left;'>
            <p class='sub-text' style='margin-bottom: 5px;'>COLOR</p>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #e2e8f0;'>{english_color}</p>
        </div>
        <div style='text-align: left;'>
            <p class='sub-text' style='margin-bottom: 5px;'>WEIGHT</p>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #e2e8f0;'>{weight_str}</p>
        </div>
        <div style='text-align: left;'>
            <p class='sub-text' style='margin-bottom: 5px;'>SEATS</p>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: #e2e8f0;'>{seats}</p>
        </div>
        <div style='text-align: left;'>
            <p class='sub-text' style='margin-bottom: 5px;'>APK EXPIRY</p>
            <p style='margin: 0; font-size: 1.1rem; font-weight: 600; color: {apk_color}'>{apk_date}</p>
        </div>
    </div>
</div>
"""
                st.markdown(databank_html, unsafe_allow_html=True)

        st.divider()
        st.caption(f"Powered by: {strategy}")
        
else:
    # Start screen
    st.markdown("""
        <div class="welcome-dash">
            <h2 style="color: #cbd5e1; font-weight: 600;">Welcome to Appraiser Pro</h2>
            <p style="color: #64748b; font-size: 1.1rem;">
                Upload a high-quality photo of a vehicle to begin.<br>
                For the most accurate market valuation, ensure the <b>license plate</b> is clearly visible.
            </p>
        </div>
    """, unsafe_allow_html=True)
