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
import urllib.parse
import time
from typing import List, Dict

st.set_page_config(
    page_title="Pinterest Style Analyzer",
    page_icon="👗",
    layout="wide"
)

st.title("Pinterest Style Analyzer")
st.write("Analyze individual pins or entire boards to discover your style profile")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

# Add sidebar with instructions
with st.sidebar:
    st.write("### How to Use")
    st.write("""
    1. Choose between analyzing individual pins or a Pinterest board
    2. For individual pins:
       - Copy the Pinterest URLs
       - Paste URLs (one per line) in the text box
    3. For boards:
       - Copy the board URL
       - Paste the single board URL
    4. Click 'Analyze Style' to get:
       - Individual image analysis
       - Overall style profile
       - Personalized recommendations
       - Styling tips
    """)
    
    st.write("### Tips for Best Results")
    st.write("""
    - Use clear, full-body outfit images
    - For individual pins, use 3-5 images
    - For boards, first 10 pins will be analyzed
    - Choose images with similar style direction
    - Include different angles/variations of the style
    """)
    
    st.write("### Board Analysis Tips")
    st.write("""
    When analyzing boards:
    - Limited to first 10 pins for performance
    - Each pin analysis is collapsible
    - Overall summary combines all analyzed pins
    - Larger boards may take longer to process
    """)
    def extract_pins_from_board(board_url):
        """Extract all pin URLs from a Pinterest board"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the board page content
        response = requests.get(board_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all pin URLs
        pin_urls = []
        
        # Method 1: Look for pin links
        pin_links = soup.find_all('a', href=re.compile(r'/pin/\d+'))
        for link in pin_links:
            pin_url = f"https://pinterest.com{link['href']}"
            if pin_url not in pin_urls:
                pin_urls.append(pin_url)
        
        # Method 2: Look for pin IDs in data attributes
        pin_elements = soup.find_all(attrs={"data-pin-id": True})
        for element in pin_elements:
            pin_id = element.get("data-pin-id")
            pin_url = f"https://pinterest.com/pin/{pin_id}"
            if pin_url not in pin_urls:
                pin_urls.append(pin_url)
        
        if not pin_urls:
            return False, None, "No pins found in board"
        
        return True, pin_urls, None
        
    except Exception as e:
        return False, None, f"Error extracting pins from board: {str(e)}"

def extract_pinterest_image_url(pin_url):
    """Extract the actual image URL from a Pinterest pin URL"""
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # Get the Pinterest page content
        response = requests.get(pin_url, headers=headers)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for image URLs in various Pinterest formats
        # Method 1: Look for og:image meta tag
        og_image = soup.find('meta', property='og:image')
        if og_image and og_image.get('content'):
            return True, og_image['content'], None
            
        # Method 2: Look for high-res image tags
        images = soup.find_all('img', {'src': re.compile(r'https://i\.pinimg\.com/.*?\.jpg')})
        if images:
            # Sort by size (looking for largest version) and get the first one
            image_urls = [img['src'] for img in images]
            # Prefer originals or larger sizes
            for url in image_urls:
                if 'originals' in url or '736x' in url:
                    return True, url, None
            return True, image_urls[0], None
        
        return False, None, "Could not find image URL in Pinterest page"
        
    except Exception as e:
        return False, None, f"Error extracting Pinterest image: {str(e)}"

def validate_image_url(url):
    """Validate if URL returns a valid image"""
    try:
        # Check if it's a Pinterest URL
        if 'pinterest.com' in url:
            success, image_url, error = extract_pinterest_image_url(url)
            if not success:
                return False, None, error
            url = image_url
        
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '')
        if not any(img_type in content_type.lower() for img_type in ['image/', 'application/octet-stream']):
            return False, None, "URL does not point to an image"
        
        return True, response.content, None
    except requests.RequestException as e:
        return False, None, f"Error fetching URL: {str(e)}"
def get_dominant_colors(img, n_colors=5):
    """Extract dominant colors from image"""
    try:
        img = img.resize((150, 150))
        pixels = np.array(img).reshape(-1, 3)
        
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
                'percentage': percentage
            })
        
        return True, sorted(color_info, key=lambda x: x['percentage'], reverse=True), None
    except Exception as e:
        return False, None, f"Error analyzing colors: {str(e)}"

def analyze_with_claude(image):
    """Analyze image style using Claude"""
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        prompt = """Analyze this image as a style expert and provide both analysis and recommendations. Focus on:

        1. ANALYSIS:
        - Key fashion pieces and their defining features (e.g., 'cropped leather jacket with silver hardware')
        - Color scheme and significant color combinations
        - Hair style and notable hair features
        - Notable style elements and aesthetic
        - Accessories and how they complete the look
        
        2. RECOMMENDATIONS:
        - Suggest 2-3 complete outfit combinations that would fit this style aesthetic
        - Recommend complementary hair styles that match this look
        - Suggest makeup approaches that would enhance this style
        - Propose accessories that would work well with this style
        
        Format response as JSON with these keys: 
        {
            "analysis": {
                "key_pieces": ["item1", "item2"],
                "color_scheme": ["color1", "color2"],
                "hair_style": ["feature1", "feature2"],
                "style_elements": ["element1", "element2"],
                "accessories": ["accessory1", "accessory2"]
            },
            "recommendations": {
                "outfit_combos": ["complete outfit 1", "complete outfit 2"],
                "hair_suggestions": ["hair style 1", "hair style 2"],
                "makeup_tips": ["makeup tip 1", "makeup tip 2"],
                "accessory_ideas": ["accessory idea 1", "accessory idea 2"]
            }
        }
        
        Be specific and descriptive in both analysis and recommendations."""
        
        response = anthropic.beta.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
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
        
        try:
            return True, json.loads(response.content[0].text), None
        except json.JSONDecodeError:
            return False, None, "Error parsing Claude's response"
            
    except Exception as e:
        return False, None, f"Error during Claude analysis: {str(e)}"

def analyze_board_with_progress(urls: List[str]) -> Dict:
    """Analyze multiple pins with progress tracking and rate limiting"""
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_colors = []
    all_analyses = []
    total_urls = len(urls)
    
    for idx, url in enumerate(urls):
        # Update progress
        progress = (idx + 1) / total_urls
        progress_bar.progress(progress)
        status_text.write(f"Processing pin {idx + 1} of {total_urls}")
        
        # Add a small delay to prevent rate limiting
        time.sleep(1)
        
        # Process the pin
        valid, image_data, error = validate_image_url(url)
        if not valid:
            st.warning(f"Skipping pin {idx + 1}: {error}")
            continue
            
        try:
            img = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Create expandable section for each pin
            with st.expander(f"Pin {idx + 1} Analysis", expanded=False):
                # Display image
                st.image(img, width=300)
                
                # Color analysis
                color_success, colors, color_error = get_dominant_colors(img)
                if color_success:
                    all_colors.extend(colors)
                    
                    # Display color palette
                    st.write("🎨 **Color Palette:**")
                    cols = st.columns(5)
                    for i, color in enumerate(colors[:5]):
                        cols[i].markdown(
                            f'<div style="background-color: {color["hex"]}; height: 50px; border-radius: 5px;"></div>',
                            unsafe_allow_html=True
                        )
                
                # Claude analysis
                analysis_success, analysis, analysis_error = analyze_with_claude(img)
                if analysis_success:
                    all_analyses.append(analysis)
                    
                    # Display analysis results
                    if analysis.get("analysis", {}):
                        st.write("#### 📸 Analysis")
                        
                        for key, emoji in [
                            ("key_pieces", "🛍️ **Key Pieces:**"),
                            ("color_scheme", "🎨 **Color Scheme:**"),
                            ("hair_style", "💇‍♀️ **Hair Style:**"),
                            ("accessories", "✨ **Accessories:**"),
                            ("style_elements", "👗 **Style Elements:**")
                        ]:
                            if analysis["analysis"].get(key):
                                st.write(emoji)
                                for item in analysis["analysis"][key]:
                                    st.write(f"- {item}")
                    
                    if analysis.get("recommendations", {}):
                        st.write("#### 💫 Style Recommendations")
                        
                        for key, emoji in [
                            ("outfit_combos", "👔 **Outfit Combinations:**"),
                            ("hair_suggestions", "💁‍♀️ **Hair Style Ideas:**"),
                            ("makeup_tips", "💄 **Makeup Tips:**"),
                            ("accessory_ideas", "👜 **Accessory Ideas:**")
                        ]:
                            if analysis["recommendations"].get(key):
                                st.write(emoji)
                                for item in analysis["recommendations"][key]:
                                    st.write(f"- {item}")
                
        except Exception as e:
            st.warning(f"Error processing pin {idx + 1}: {str(e)}")
            continue
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    return {
        "colors": all_colors,
        "analyses": all_analyses
    }

# Add radio button for input type
input_type = st.radio(
    "Select input type:",
    ["Individual Pins", "Pinterest Board"]
)

if input_type == "Individual Pins":
    urls_input = st.text_area("Enter Pinterest pin URLs (one per line)")
else:
    urls_input = st.text_input("Enter Pinterest board URL")

def display_style_summary(all_colors, all_analyses):
    """Display comprehensive style summary and recommendations"""
    st.write("## 📊 Overall Style Profile & Recommendations")
    
    # Show dominant color palette
    if all_colors:
        st.write("### 🎨 Color Palette")
        color_counts = Counter([color['hex'] for color in all_colors])
        top_colors = color_counts.most_common(8)
        
        cols = st.columns(8)
        for i, (color, count) in enumerate(top_colors):
            cols[i].markdown(
                f'<div style="background-color: {color}; height: 50px; border-radius: 5px;"></div>',
                unsafe_allow_html=True
            )
    
    # Aggregate key pieces
    all_pieces = [piece for analysis in all_analyses for piece in analysis.get("analysis", {}).get("key_pieces", [])]
    common_pieces = Counter(all_pieces).most_common(8)
    
    if common_pieces:
        st.write("### 🛍️ Signature Pieces")
        for piece, count in common_pieces:
            frequency = f"(Found in {count} {'image' if count == 1 else 'images'})"
            st.write(f"- {piece} {frequency}")
    
    # Aggregate hair styles
    all_hair = [style for analysis in all_analyses for style in analysis.get("analysis", {}).get("hair_style", [])]
    common_hair = Counter(all_hair).most_common(5)
    
    if common_hair:
        st.write("### 💇‍♀️ Defining Hair Styles")
        for style, count in common_hair:
            frequency = f"(Found in {count} {'image' if count == 1 else 'images'})"
            st.write(f"- {style} {frequency}")
    
    # Aggregate style elements
    all_elements = [elem for analysis in all_analyses for elem in analysis.get("analysis", {}).get("style_elements", [])]
    common_elements = Counter(all_elements).most_common(5)
    
    if common_elements:
        st.write("### 👗 Overall Style Direction")
        for element, count in common_elements:
            frequency = f"(Found in {count} {'image' if count == 1 else 'images'})"
            st.write(f"- {element} {frequency}")
    
    # Aggregate all recommendations
    st.write("### 💫 Style Recommendations")
    
    # Outfit combinations
    all_outfits = [outfit for analysis in all_analyses for outfit in analysis.get("recommendations", {}).get("outfit_combos", [])]
    outfit_counts = Counter(all_outfits).most_common(6)
    
    if outfit_counts:
        st.write("#### 👔 Top Outfit Combinations")
        for outfit, count in outfit_counts:
            frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
            st.write(f"- {outfit} {frequency}")
    
    # Hair suggestions
    all_hair_ideas = [hair for analysis in all_analyses for hair in analysis.get("recommendations", {}).get("hair_suggestions", [])]
    hair_counts = Counter(all_hair_ideas).most_common(4)
    
    if hair_counts:
        st.write("#### 💁‍♀️ Recommended Hair Styles")
        for hair, count in hair_counts:
            frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
            st.write(f"- {hair} {frequency}")
    
    # Makeup tips
    all_makeup = [tip for analysis in all_analyses for tip in analysis.get("recommendations", {}).get("makeup_tips", [])]
    makeup_counts = Counter(all_makeup).most_common(4)
    
    if makeup_counts:
        st.write("#### 💄 Makeup Suggestions")
        for makeup, count in makeup_counts:
            frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
            st.write(f"- {makeup} {frequency}")
    
    # Accessory ideas
    all_accessories = [acc for analysis in all_analyses for acc in analysis.get("recommendations", {}).get("accessory_ideas", [])]
    accessory_counts = Counter(all_accessories).most_common(5)
    
    if accessory_counts:
        st.write("#### 👜 Accessory Ideas")
        for accessory, count in accessory_counts:
            frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
            st.write(f"- {accessory} {frequency}")
    
    # Add style tips
    st.write("### 💭 Styling Tips")
    st.write("""
    - Mix and match the suggested pieces to create your own unique combinations
    - Use the color palette as a guide when shopping for new items
    - Try different hair and makeup combinations for various occasions
    - Start with basic pieces and add statement accessories to elevate the look
    """)

if st.button("Analyze Style"):
    if not urls_input:
        st.error("Please enter a URL")
    else:
        if input_type == "Individual Pins":
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            if not urls:
                st.error("Please enter at least one valid URL")
            else:
                results = analyze_board_with_progress(urls)
                if results["analyses"]:
                    display_style_summary(results["colors"], results["analyses"])
        else:
            # Extract pins from board
            with st.spinner("Extracting pins from board..."):
                success, board_pins, error = extract_pins_from_board(urls_input)
                if not success:
                    st.error(f"Error processing board: {error}")
                    st.stop()
                
                if not board_pins:
                    st.error("No pins found in board")
                    st.stop()
                
                if len(board_pins) > 10:
                    st.write("⚠️ Analyzing first 10 pins for performance reasons")
                    board_pins = board_pins[:10]
                
                st.write(f"Found {len(board_pins)} pins to analyze")
                
                # Analyze board with progress tracking
                results = analyze_board_with_progress(board_pins)
                
                if results["analyses"]:
                    display_style_summary(results["colors"], results["analyses"])

st.markdown("---")
st.write("Note: For best results, use clear, front-facing outfit images.")
