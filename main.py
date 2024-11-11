import streamlit as st
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
import requests
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import cv2
import time
import base64
from anthropic import Anthropic
import json

# Initialize Anthropic client
def init_claude():
    api_key = st.secrets["ANTHROPIC_API_KEY"]
    return Anthropic(api_key=api_key)

def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_with_claude(client, image, previous_analyses=None):
    """Analyze image style using Claude"""
    base64_image = encode_image_to_base64(image)
    
    context = """You are a professional fashion and style analyst. Analyze this image and provide detailed insights about:
    1. Clothing items and their details
    2. Style category (e.g., bohemian, minimalist, preppy, etc.)
    3. Occasion appropriateness
    4. Notable design elements
    5. Styling tips based on this look
    
    Provide the analysis in JSON format with these keys: 
    {
        "clothing_items": [], 
        "style_category": "", 
        "occasion": [], 
        "design_elements": [], 
        "styling_tips": []
    }"""
    
    if previous_analyses:
        context += "\nConsider these previous analyses for consistency in style categorization: " + json.dumps(previous_analyses)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": context
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
    
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=messages
    )
    
    try:
        return json.loads(response.content[0].text)
    except:
        return {
            "clothing_items": [],
            "style_category": "unknown",
            "occasion": [],
            "design_elements": [],
            "styling_tips": []
        }

def get_dominant_colors(img, n_colors=5):
    # Resize image for faster processing
    img = img.resize((150, 150))
    pixels = np.float32(img).reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    counts = Counter(labels)
    
    total_pixels = sum(counts.values())
    color_info = []
    
    for i in range(n_colors):
        rgb = tuple(map(int, colors[i]))
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        percentage = (counts[i] / total_pixels) * 100
        color_info.append({
            'hex': hex_color,
            'percentage': percentage,
            'rgb': rgb
        })
    
    return sorted(color_info, key=lambda x: x['percentage'], reverse=True)

def scrape_pinterest_board(url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        time.sleep(5)
        
        # Scroll to load more images
        for _ in range(3):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        img_elements = driver.find_elements(By.CSS_SELECTOR, "img[srcset]")
        image_urls = [img.get_attribute('src') for img in img_elements if 'pinimg' in img.get_attribute('src')]
        
        return list(set(image_urls))
    finally:
        driver.quit()

def main():
    st.title("Advanced Pinterest Style Analyzer")
    st.write("Analyze style elements using computer vision and Claude AI")
    
    # Initialize Claude client
    try:
        claude_client = init_claude()
    except Exception as e:
        st.error("Failed to initialize Claude API. Please check your API key.")
        return
    
    board_url = st.text_input("Enter the URL of a public Pinterest board:")
    
    if st.button("Analyze Board") and board_url:
        try:
            with st.spinner("Scraping Pinterest board..."):
                image_urls = scrape_pinterest_board(board_url)
            
            if not image_urls:
                st.error("No images found. Please check the URL and try again.")
                return
            
            st.success(f"Found {len(image_urls)} images!")
            
            # Initialize collectors
            all_colors = []
            all_analyses = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            
            # Analyze each image
            for i, url in enumerate(image_urls[:10]):  # Limit to 10 images for demo
                try:
                    response = requests.get(url)
                    img = Image.open(io.BytesIO(response.content)).convert('RGB')
                    
                    # Get color analysis
                    colors = get_dominant_colors(img)
                    all_colors.extend(colors)
                    
                    # Get Claude's analysis
                    claude_analysis = analyze_with_claude(claude_client, img, all_analyses)
                    all_analyses.append(claude_analysis)
                    
                    progress_bar.progress((i + 1) / min(len(image_urls), 10))
                    
                except Exception as e:
                    continue
            
            # Display Results
            st.header("Style Analysis Results")
            
            # Color Palette
            st.subheader("Color Palette")
            color_counts = Counter([color['hex'] for color in all_colors])
            top_colors = color_counts.most_common(8)
            
            cols = st.columns(8)
            for i, (color, count) in enumerate(top_colors):
                cols[i].markdown(
                    f'<div style="background-color: {color}; height: 50px; border-radius: 5px;"></div>',
                    unsafe_allow_html=True
                )
                cols[i].write(f'{(count/len(all_colors)*100):.1f}%')
            
            # Style Analysis from Claude
            st.subheader("Style Profile")
            
            # Aggregate style categories
            style_categories = Counter([analysis['style_category'] for analysis in all_analyses])
            dominant_style = max(style_categories.items(), key=lambda x: x[1])[0]
            
            # Aggregate clothing items
            all_items = [item for analysis in all_analyses for item in analysis['clothing_items']]
            common_items = Counter(all_items).most_common(5)
            
            # Display style insights
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Dominant Style:")
                st.write(f"**{dominant_style.title()}**")
                
                st.write("Key Pieces:")
                for item, count in common_items:
                    st.write(f"- {item}")
            
            with col2:
                st.write("Common Occasions:")
                occasions = Counter([occ for analysis in all_analyses for occ in analysis['occasion']])
                for occ, count in occasions.most_common(3):
                    st.write(f"- {occ}")
            
            # Design Elements and Styling Tips
            st.subheader("Style Guide")
            
            # Aggregate design elements and styling tips
            design_elements = set([elem for analysis in all_analyses for elem in analysis['design_elements']])
            styling_tips = set([tip for analysis in all_analyses for tip in analysis['styling_tips']])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Design Elements:")
                for elem in list(design_elements)[:5]:
                    st.write(f"- {elem}")
            
            with col2:
                st.write("Styling Tips:")
                for tip in list(styling_tips)[:5]:
                    st.write(f"- {tip}")
            
            # Overall Style Summary
            st.subheader("Style Summary")
            st.write(f"""
            This Pinterest board showcases a predominantly **{dominant_style}** aesthetic, 
            featuring a color palette dominated by {', '.join([f"**{color}**" for color, _ in top_colors[:3]])}.
            The collection is particularly suited for {', '.join(list(occasions)[:2])}, 
            with an emphasis on {', '.join(list(design_elements)[:2])}.
            """)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please make sure the Pinterest board is public and the URL is correct.")

if __name__ == "__main__":
    main()
