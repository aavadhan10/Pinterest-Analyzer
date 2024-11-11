import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import json
from anthropic import Anthropic
import base64
import re
from bs4 import BeautifulSoup
import time
from typing import List, Dict

st.set_page_config(
    page_title="Pinterest Style Profile Analyzer",
    page_icon="👗",
    layout="wide"
)

st.title("Pinterest Style Profile Analyzer")
st.write("Analyze your Pinterest style from individual pins or an entire board")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Sidebar instructions
with st.sidebar:
    st.write("### How to Use")
    st.write("""
    1. Choose your input type:
       - Individual pins: Enter multiple Pinterest URLs
       - Board: Enter a single board URL
    2. Click 'Analyze Style' to get:
       - Your color palette
       - Key pieces
       - Style elements
       - Personalized recommendations
    """)
    
    st.write("### Pro Tips")
    st.write("""
    - Boards give better results than individual pins
    - More pins = more accurate analysis
    - Use fashion/outfit focused pins
    - Mix of full body and detail shots work best
    """)

def extract_pinterest_urls(input_url: str) -> List[str]:
    """Extract pins from either a board URL or individual pin URLs"""
    if 'pinterest.com/pin/' in input_url:
        # Individual pins
        return [url.strip() for url in input_url.split('\n') if url.strip()]
    else:
        # Board URL
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(input_url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            pins = []
            # Find pin URLs
            for link in soup.find_all('a', href=re.compile(r'/pin/\d+')):
                pin_url = f"https://pinterest.com{link['href']}"
                if pin_url not in pins:
                    pins.append(pin_url)
            return pins
        except Exception as e:
            st.error(f"Error extracting pins: {str(e)}")
            return []

def analyze_style_profile(urls: List[str], max_pins: int = 20) -> Dict:
    """Analyze multiple pins and aggregate the results"""
    
    if len(urls) > max_pins:
        st.info(f"⚡ Analyzing {max_pins} pins for optimal performance")
        urls = urls[:max_pins]
    
    # Setup progress tracking
    progress_bar = st.progress(0)
    status = st.empty()
    
    # Collection variables
    all_colors = []
    all_pieces = []
    all_styles = []
    all_hair_styles = []
    all_accessories = []
    all_outfit_ideas = []
    all_makeup_tips = []
    
    # Process each pin
    for idx, url in enumerate(urls):
        progress = (idx + 1) / len(urls)
        progress_bar.progress(progress)
        status.write(f"Analyzing pin {idx + 1} of {len(urls)}")
        
        try:
            # Get image from pin
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find image URL
            img_url = None
            og_image = soup.find('meta', property='og:image')
            if og_image and og_image.get('content'):
                img_url = og_image['content']
            
            if not img_url:
                continue
                
            # Get image
            img_response = requests.get(img_url, headers=headers)
            img = Image.open(io.BytesIO(img_response.content)).convert('RGB')
            
            # Color analysis
            img_small = img.resize((150, 150))
            pixels = np.array(img_small).reshape(-1, 3)
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_
            
            # Convert colors to hex
            for color in colors:
                rgb = tuple(map(int, color))
                hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
                all_colors.append(hex_color)
            
            # Style analysis with Claude
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            prompt = """Analyze this fashion image and provide only key details about:
            1. Main pieces/items shown
            2. Style elements and aesthetic
            3. Hair style features
            4. Accessories
            5. One outfit combination suggestion
            6. One makeup suggestion that matches the style

            Format as JSON with keys: pieces, style_elements, hair_style, accessories, outfit_suggestion, makeup_suggestion"""
            
            response = anthropic.beta.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=500,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            analysis = json.loads(response.content[0].text)
            
            # Collect results
            all_pieces.extend(analysis.get('pieces', []))
            all_styles.extend(analysis.get('style_elements', []))
            all_hair_styles.extend(analysis.get('hair_style', []))
            all_accessories.extend(analysis.get('accessories', []))
            all_outfit_ideas.append(analysis.get('outfit_suggestion'))
            all_makeup_tips.append(analysis.get('makeup_suggestion'))
            
            # Small delay to prevent rate limiting
            time.sleep(1)
            
        except Exception as e:
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status.empty()
    
    # Return aggregated results
    return {
        'colors': Counter(all_colors).most_common(8),
        'pieces': Counter(all_pieces).most_common(10),
        'styles': Counter(all_styles).most_common(6),
        'hair_styles': Counter(all_hair_styles).most_common(5),
        'accessories': Counter(all_accessories).most_common(6),
        'outfit_ideas': Counter([x for x in all_outfit_ideas if x]).most_common(5),
        'makeup_tips': Counter([x for x in all_makeup_tips if x]).most_common(4)
    }

def display_style_profile(results: Dict):
    """Display aggregated style profile results"""
    st.write("# 🎯 Your Style Profile")
    
    # Color palette
    if results['colors']:
        st.write("### 🎨 Color Palette")
        cols = st.columns(len(results['colors']))
        for idx, (color, count) in enumerate(results['colors']):
            cols[idx].markdown(
                f'<div style="background-color: {color}; height: 60px; border-radius: 5px;" title="Used {count} times"></div>',
                unsafe_allow_html=True
            )
    
    # Two-column layout for main content
    col1, col2 = st.columns(2)
    
    with col1:
        # Key pieces
        if results['pieces']:
            st.write("### 🛍️ Signature Pieces")
            for piece, count in results['pieces']:
                percentage = (count / len(results['pieces'])) * 100
                st.write(f"- {piece} ({percentage:.0f}%)")
        
        # Style elements
        if results['styles']:
            st.write("### 👗 Style Direction")
            for style, count in results['styles']:
                percentage = (count / len(results['styles'])) * 100
                st.write(f"- {style} ({percentage:.0f}%)")
        
        # Hair styles
        if results['hair_styles']:
            st.write("### 💇‍♀️ Hair Styles")
            for style, count in results['hair_styles']:
                percentage = (count / len(results['hair_styles'])) * 100
                st.write(f"- {style} ({percentage:.0f}%)")
    
    with col2:
        # Accessories
        if results['accessories']:
            st.write("### ✨ Key Accessories")
            for acc, count in results['accessories']:
                percentage = (count / len(results['accessories'])) * 100
                st.write(f"- {acc} ({percentage:.0f}%)")
        
        # Outfit ideas
        if results['outfit_ideas']:
            st.write("### 👔 Outfit Combinations")
            for outfit, _ in results['outfit_ideas']:
                st.write(f"- {outfit}")
        
        # Makeup tips
        if results['makeup_tips']:
            st.write("### 💄 Makeup Suggestions")
            for tip, _ in results['makeup_tips']:
                st.write(f"- {tip}")

# Main app logic
input_type = st.radio("Select input type:", ["Pinterest Board", "Individual Pins"])

if input_type == "Pinterest Board":
    urls_input = st.text_input("Enter Pinterest board URL")
else:
    urls_input = st.text_area("Enter Pinterest pin URLs (one per line)")

if st.button("Analyze Style"):
    if not urls_input:
        st.error("Please enter URL(s)")
    else:
        with st.spinner("🔍 Analyzing your style..."):
            # Get pins
            pins = extract_pinterest_urls(urls_input)
            
            if not pins:
                st.error("No pins found to analyze")
            else:
                # Analyze pins
                st.write(f"📌 Found {len(pins)} pins to analyze")
                results = analyze_style_profile(pins)
                
                # Display results
                display_style_profile(results)

st.markdown("---")
st.caption("Made with ❤️ for fashion enthusiasts")
