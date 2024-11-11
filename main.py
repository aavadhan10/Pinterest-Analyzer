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

st.title("Pinterest Style Analyzer")
st.write("Enter Pinterest image URLs to analyze their style")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

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

# Input for image URLs
image_urls = st.text_area("Enter Pinterest image URLs (one per line)")

if st.button("Analyze Style") and image_urls:
    urls = [url.strip() for url in image_urls.split('\n') if url.strip()]
    
    if not urls:
        st.error("Please enter at least one valid URL")
    else:
        all_colors = []
        all_analyses = []
        
        for url in urls[:5]:  # Limit to 5 images for demo
            st.write(f"Processing: {url}")
            
            # Validate and fetch image
            valid, image_data, error = validate_image_url(url)
            if not valid:
                st.error(f"Error with URL {url}: {error}")
                continue
                
            try:
                # Open image
                img = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Display image
                st.image(img, width=300)
                
                # Color analysis
                color_success, colors, color_error = get_dominant_colors(img)
                if not color_success:
                    st.warning(f"Color analysis error: {color_error}")
                else:
                    all_colors.extend(colors)
                
                # Claude analysis
                analysis_success, analysis, analysis_error = analyze_with_claude(img)
                if not analysis_success:
                    st.warning(f"Style analysis error: {analysis_error}")
                else:
                    all_analyses.append(analysis)
                
                # Display individual image analysis
                st.write("### Style Analysis")
                    
                if analysis:
                    # Analysis section
                    st.write("#### 📸 Analysis")
                    
                    if analysis.get("analysis", {}).get("key_pieces"):
                        st.write("🛍️ **Key Pieces:**")
                        for piece in analysis["analysis"]["key_pieces"]:
                            st.write(f"- {piece}")
                    
                    if analysis.get("analysis", {}).get("color_scheme"):
                        st.write("🎨 **Color Scheme:**")
                        for color in analysis["analysis"]["color_scheme"]:
                            st.write(f"- {color}")
                    
                    if analysis.get("analysis", {}).get("hair_style"):
                        st.write("💇‍♀️ **Hair Style:**")
                        for style in analysis["analysis"]["hair_style"]:
                            st.write(f"- {style}")
                    
                    if analysis.get("analysis", {}).get("accessories"):
                        st.write("✨ **Accessories:**")
                        for accessory in analysis["analysis"]["accessories"]:
                            st.write(f"- {accessory}")
                    
                    if analysis.get("analysis", {}).get("style_elements"):
                        st.write("👗 **Style Elements:**")
                        for element in analysis["analysis"]["style_elements"]:
                            st.write(f"- {element}")
                    
                    # Recommendations section
                    st.write("#### 💫 Style Recommendations")
                    
                    if analysis.get("recommendations", {}).get("outfit_combos"):
                        st.write("👔 **Outfit Combinations:**")
                        for outfit in analysis["recommendations"]["outfit_combos"]:
                            st.write(f"- {outfit}")
                    
                    if analysis.get("recommendations", {}).get("hair_suggestions"):
                        st.write("💁‍♀️ **Hair Style Ideas:**")
                        for hair in analysis["recommendations"]["hair_suggestions"]:
                            st.write(f"- {hair}")
                    
                    if analysis.get("recommendations", {}).get("makeup_tips"):
                        st.write("💄 **Makeup Tips:**")
                        for tip in analysis["recommendations"]["makeup_tips"]:
                            st.write(f"- {tip}")
                    
                    if analysis.get("recommendations", {}).get("accessory_ideas"):
                        st.write("👜 **Accessory Suggestions:**")
                        for idea in analysis["recommendations"]["accessory_ideas"]:
                            st.write(f"- {idea}")
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                continue
        
        if all_analyses:
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
            all_outfits = []
            all_hair_ideas = []
            all_makeup_tips = []
            all_accessory_ideas = []
            
            for analysis in all_analyses:
                recs = analysis.get("recommendations", {})
                all_outfits.extend(recs.get("outfit_combos", []))
                all_hair_ideas.extend(recs.get("hair_suggestions", []))
                all_makeup_tips.extend(recs.get("makeup_tips", []))
                all_accessory_ideas.extend(recs.get("accessory_ideas", []))
            
            # Display aggregated recommendations
            st.write("### 💫 Style Recommendations")
            
            if all_outfits:
                st.write("#### 👔 Top Outfit Combinations")
                outfit_counts = Counter(all_outfits).most_common(6)
                for outfit, count in outfit_counts:
                    frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
                    st.write(f"- {outfit} {frequency}")
            
            if all_hair_ideas:
                st.write("#### 💁‍♀️ Recommended Hair Styles")
                hair_counts = Counter(all_hair_ideas).most_common(4)
                for hair, count in hair_counts:
                    frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
                    st.write(f"- {hair} {frequency}")
            
            if all_makeup_tips:
                st.write("#### 💄 Makeup Suggestions")
                makeup_counts = Counter(all_makeup_tips).most_common(4)
                for makeup, count in makeup_counts:
                    frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
                    st.write(f"- {makeup} {frequency}")
            
            if all_accessory_ideas:
                st.write("#### 👜 Accessory Ideas")
                accessory_counts = Counter(all_accessory_ideas).most_common(5)
                for accessory, count in accessory_counts:
                    frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
                    st.write(f"- {makeup} {frequency}")
            
            if all_accessory_ideas:
                st.write("#### 👜 Accessory Ideas")
                accessory_counts = Counter(all_accessory_ideas).most_common(5)
                for accessory, count in accessory_counts:
                    frequency = f"(Suggested {count} {'time' if count == 1 else 'times'})"
                    st.write(f"- {accessory} {frequency}")
            
            # Add final style tips
            st.write("### 💭 Styling Tips")
            st.write("""
            - Mix and match the suggested pieces to create your own unique combinations
            - Use the color palette as a guide when shopping for new items
            - Try different hair and makeup combinations for various occasions
            - Start with basic pieces and add statement accessories to elevate the look
            """)

st.markdown("---")
st.write("Note: For best results, use direct image URLs from Pinterest pins.")

# Add sidebar with instructions
with st.sidebar:
    st.write("### How to Use")
    st.write("""
    1. Find Pinterest pins that represent your desired style
    2. Copy the Pinterest URLs
    3. Paste URLs (one per line) in the text box
    4. Click 'Analyze Style' to get:
        - Individual image analysis
        - Overall style profile
        - Personalized recommendations
        - Styling tips
    """)
    
    st.write("### Tips for Best Results")
    st.write("""
    - Use clear, full-body outfit images
    - Include 3-5 images for better recommendations
    - Choose images with similar style direction
    - Include different angles/variations of the style
    """)
                   
