import cv2
import mediapipe as mp
import streamlit as st
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import colorsys
import pandas as pd
import subprocess
import time
import webbrowser
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

def apply_selfie_segmentation(image, model_selection):
    with mp_selfie_segmentation.SelfieSegmentation(model_selection=model_selection) as selfie_segmentation:
        results = selfie_segmentation.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)  # Default background color
        output_image = np.where(condition, image, bg_image)
        return output_image

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


def crop_to_face(image, faces):
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    cropped_image = image[y:y + h, x:x + w]
    return cropped_image

def detect_colors(image, num_colors=20, detect_lip_color=False):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure image is in RGB format for KMeans
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_

    filtered_colors = []
    for color in colors:
        r, g, b = color
        if not (r == g == b or abs(r - g) < 15 and abs(r - b) < 15 and abs(g - b) < 15):
            filtered_colors.append(tuple(map(int, color)))

    if detect_lip_color:

        lip_roi = image[150:250, 100:300]  # Example ROI coordinates

        lip_roi_rgb = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2RGB)
        lip_roi_rgb = lip_roi_rgb.reshape((lip_roi_rgb.shape[0] * lip_roi_rgb.shape[1], 3))

        kmeans_lip = KMeans(n_clusters=1)
        kmeans_lip.fit(lip_roi_rgb)
        lip_color = tuple(map(int, kmeans_lip.cluster_centers_[0]))
        filtered_colors.append(lip_color)

    if len(filtered_colors) < 20:
        num_additional_colors = 20 - len(filtered_colors)
        additional_colors = np.random.randint(0, 256, size=(num_additional_colors, 3))
        filtered_colors.extend(map(tuple, additional_colors))

    return filtered_colors[:20]


def rgb_to_hsl(r, g, b):
    h, l, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    h = h * 360  
    s = s * 100  
    l = l * 100  
    return h, s, l

def determine_temperature(hues):
    warm_hues = sum(1 for h in hues if 0 <= h <= 60 or 300 <= h <= 360)
    cool_hues = len(hues) - warm_hues
    return "Warm" if warm_hues >= cool_hues else "Cool"

def determine_depth(lightness_values):
    average_lightness = sum(lightness_values) / len(lightness_values)
    if average_lightness < 20:
        return "Very Dark"
    elif 20 <= average_lightness < 40:
        return "Dark"
    elif 40 <= average_lightness < 60:
        return "Medium"
    elif 60 <= average_lightness < 80:
        return "Light"
    else:
        return "Very Light"

def determine_chroma(saturation_values):
    average_saturation = sum(saturation_values) / len(saturation_values)
    if average_saturation < 20:
        return "Low"
    elif 20 <= average_saturation < 40:
        return "Medium"
    else:
        return "High"

def map_to_season(hues, lightness_values, saturation_values):
    seasons = {
        'Bright Spring': 0,
        'True Spring': 0,
        'Light Spring': 0,
        'Light Summer': 0,
        'True Summer': 0,
        'Soft Summer': 0,
        'Soft Autumn': 0,
        'True Autumn': 0,
        'Deep Autumn': 0,
        'Deep Winter': 0,
        'True Winter': 0,
        'Bright Winter': 0
    }
    for h, l, s in zip(hues, lightness_values, saturation_values):
        if 0 <= h < 45 or 330 <= h <= 360:  
            if s > 50 and l > 50:
                seasons['Bright Spring'] += 1
            elif s > 50 and l <= 50:
                seasons['True Spring'] += 1
            elif s <= 50 and l > 50:
                seasons['Light Spring'] += 1
        elif 45 <= h < 170:  # Cool hues
            if s <= 50 and l > 50:
                seasons['Light Summer'] += 1
            elif s <= 50 and l <= 50:
                seasons['True Summer'] += 1
            elif s > 50 and l <= 50:
                seasons['Soft Summer'] += 1
        elif 170 <= h < 260:  # Muted hues
            if s <= 50 and l <= 50:
                seasons['Soft Autumn'] += 1
            elif s > 50 and l <= 50:
                seasons['True Autumn'] += 1
            elif s > 50 and l > 50:
                seasons['Deep Autumn'] += 1
        elif 260 <= h < 330:  # Cool hues
            if s > 50 and l <= 50:
                seasons['Deep Winter'] += 1
            elif s > 50 and l > 50:
                seasons['True Winter'] += 1
            elif s <= 50 and l > 50:
                seasons['Bright Winter'] += 1

    predominant_season = max(seasons, key=seasons.get)
    return predominant_season

def recommend_colors(season):
    color_map = {
        'Bright Spring': 'red-yellow-blue',
        'True Spring': 'yellow-green-blue',
        'Light Spring': 'pink-beige-blue',
        'Light Summer': 'pastel-blue-green',
        'True Summer': 'blue-green-pink',
        'Soft Summer': 'grey-blue-green',
        'Soft Autumn': 'earth-tone',
        'True Autumn': 'orange-brown-green',
        'Deep Autumn': 'dark-brown-green',
        'Deep Winter': 'black-grey-blue',
        'True Winter': 'black-white-red',
        'Bright Winter': 'bright-blue-red-white'
    }
    colors = color_map.get(season, 'black')
    color_query = '+'.join(colors.split('-'))
    url = f"https://www.myntra.com/{colors}?rawQuery={color_query}"

    color_ranges = {
        'Bright Spring': [(255, 0, 0), (255, 255, 0), (0, 0, 255)],
        'True Spring': [(255, 255, 0), (0, 255, 0), (0, 0, 255)],
        'Light Spring': [(255, 192, 203), (255, 228, 225), (0, 191, 255)],
        'Light Summer': [(173, 216, 230), (144, 238, 144), (152, 251, 152)],
        'True Summer': [(70, 130, 180), (0, 255, 255), (255, 192, 203)],
        'Soft Summer': [(128, 128, 128), (192, 192, 192), (0, 128, 128)],
        'Soft Autumn': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],
        'True Autumn': [(255, 69, 0), (139, 69, 19), (0, 128, 0)],
        'Deep Autumn': [(101, 67, 33), (139, 69, 19), (0, 100, 0)],
        'Deep Winter': [(0, 0, 0), (169, 169, 169), (0, 0, 139)],
        'True Winter': [(0, 0, 0), (255, 255, 255), (255, 0, 0)],
        'Bright Winter': [(0, 0, 255), (255, 0, 0), (255, 255, 255)]
    }
    palette = color_ranges.get(season, [])
    
    return [url, palette]
def open_url(url):
    webbrowser.open(url)

def main():
    # Set page config for better UI
    st.set_page_config(
        page_title="MyPalette - Color Analysis",
        page_icon="🎨",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for modern, minimal design
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            font-size: 1.2rem;
            margin-bottom: 3rem;
            font-weight: 300;
        }
        
        .upload-section {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 2rem;
            margin: 2rem 0;
            border: 2px dashed #e0e0e0;
            transition: all 0.3s ease;
        }
        
        .upload-section:hover {
            border-color: #667eea;
            background: #f0f2ff;
        }
        
        .result-card {
            background: white;
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        
        .color-palette {
            display: flex;
            gap: 10px;
            margin: 1rem 0;
            flex-wrap: wrap;
        }
        
        .color-swatch {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            border: 2px solid white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .cta-button {
            display: inline-block;
            padding: 12px 30px;
            font-size: 16px;
            font-weight: 600;
            color: white;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 25px;
            text-align: center;
            text-decoration: none;
            transition: all 0.3s ease;
            border: none;
            cursor: pointer;
            margin: 1rem 0;
        }
        
        .cta-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
            color: white;
            text-decoration: none;
        }
        
        .progress-container {
            margin: 2rem 0;
        }
        
        .step-indicator {
            display: flex;
            justify-content: center;
            align-items: center;
            margin: 2rem 0;
        }
        
        .step {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: #e0e0e0;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 10px;
            font-weight: bold;
            color: #666;
        }
        
        .step.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .step.completed {
            background: #4caf50;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎨 MyPalette</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Discover your perfect color palette with AI-powered analysis</p>', unsafe_allow_html=True)
    
    # Step indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="step-indicator">
                <div class="step active">1</div>
                <div style="width: 50px; height: 2px; background: #e0e0e0;"></div>
                <div class="step">2</div>
                <div style="width: 50px; height: 2px; background: #e0e0e0;"></div>
                <div class="step">3</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown('<p style="text-align: center; color: #666; margin-top: 1rem;">Upload → Analyze → Shop</p>', unsafe_allow_html=True)
    
    # Upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📸 Upload Your Photo")
        st.markdown("Upload a clear photo of yourself for personalized color analysis")
        
        uploaded_file = st.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        # Update step indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
                <div class="step-indicator">
                    <div class="step completed">✓</div>
                    <div style="width: 50px; height: 2px; background: #4caf50;"></div>
                    <div class="step active">2</div>
                    <div style="width: 50px; height: 2px; background: #e0e0e0;"></div>
                    <div class="step">3</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Processing section
        with st.container():
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown("### 🔬 Analyzing Your Colors...")
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processing steps
            status_text.text("📤 Processing uploaded image...")
            progress_bar.progress(20)
            
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, 1)
            
            status_text.text("🎭 Applying segmentation...")
            progress_bar.progress(40)
            segmented_image = apply_selfie_segmentation(image, 0)
            
            status_text.text("👤 Detecting face...")
            progress_bar.progress(60)
            faces = detect_faces(segmented_image)
            
            status_text.text("🎨 Extracting colors...")
            progress_bar.progress(80)
            cropped_image = crop_to_face(segmented_image.copy(), faces)
            
            if cropped_image is not None:
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Show animation video
                video_file = "./animation.mp4"
                try:
                    with open(video_file, "rb") as video:
                        st.video(video.read())
                except:
                    time.sleep(2)
                
                # Results section
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Final step indicator
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("""
                        <div class="step-indicator">
                            <div class="step completed">✓</div>
                            <div style="width: 50px; height: 2px; background: #4caf50;"></div>
                            <div class="step completed">✓</div>
                            <div style="width: 50px; height: 2px; background: #4caf50;"></div>
                            <div class="step active">3</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Color analysis results
                colors = detect_colors(cropped_image, num_colors=20, detect_lip_color=True)
                hsl_colors = [rgb_to_hsl(r, g, b) for r, g, b in colors]
                hues = [h for h, s, l in hsl_colors]
                lightness_values = [l for h, s, l in hsl_colors]
                saturation_values = [s for h, s, l in hsl_colors]
                
                temperature = determine_temperature(hues)
                depth = determine_depth(lightness_values)
                chroma = determine_chroma(saturation_values)
                season = map_to_season(hues, lightness_values, saturation_values)
                
                recommended_colors_url, recommended_colors_palette = recommend_colors(season)
                
                # Display results in a beautiful card
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 🌟 Your Color Analysis Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**🌡️ Temperature:** {temperature}")
                    st.markdown(f"**🌊 Depth:** {depth}")
                    st.markdown(f"**✨ Chroma:** {chroma}")
                    st.markdown(f"**🍂 Season:** {season}")
                
                with col2:
                    st.markdown("**� Recommended for you:**")
                    st.markdown("Based on your color analysis, we've curated the perfect color palette for your shopping experience on Myntra!")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Call to action
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown("### 🛍️ Ready to Shop?")
                st.markdown("Discover clothing that perfectly matches your color palette!")
                
                st.markdown(
                    f'<a href="{recommended_colors_url}" target="_blank" class="cta-button">🎯 Shop Your Colors on Myntra</a>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("❌ No face detected in the image. Please upload a clear photo with your face visible.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; margin-top: 2rem;">Made with ❤️ using AI • Powered by 12-Season Color Theory</p>',
        unsafe_allow_html=True
    )



if __name__ == '__main__':
    main()
