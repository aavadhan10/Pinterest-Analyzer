import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import json
from anthropic import Anthropic

st.title("Pinterest Style Analyzer")
st.write("Enter Pinterest image URLs to analyze their style")

# Initialize Anthropic client
anthropic = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])

def get_dominant_colors(img, n_colors=5):
    """Extract dominant colors from image"""
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
    
    return sorted(color_info, key=lambda x: x['percentage'], reverse=True)

def analyze_with_claude(image):
    """Analyze image style using Claude"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    prompt = """As a fashion stylist, analyze this image and provide insights about:
    1. Main style elements
    2. Color palette description
    3. Occasion appropriateness
    4. Key pieces identified
    
    Format as JSON with these keys: style_elements, color_description, occasions, key_pieces"""
    
    messages = [
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
    
    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        messages=messages
    )
    
    try:
        return json.loads(response.content[0].text)
    except:
        return {
            "style_elements": [],
            "color_description": "",
            "occasions": [],
            "key_pieces": []
        }

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
            try:
                # Get image from URL
                response = requests.get(url)
                img = Image.open(io.BytesIO(response.content)).convert('RGB')
                
                # Display image
                st.image(img, width=300)
                
                # Color analysis
                colors = get_dominant_colors(img)
                all_colors.extend(colors)
                
                # Claude analysis
                analysis = analyze_with_claude(img)
                all_analyses.append(analysis)
                
                # Display individual image analysis
                st.write("### Image Analysis")
                
                # Color palette
                st.write("Color Palette:")
                cols = st.columns(5)
                for idx, color in enumerate(colors[:5]):
                    cols[idx].markdown(
                        f'<div style="background-color: {color["hex"]}; height: 50px; border-radius: 5px;"></div>',
                        unsafe_allow_html=True
                    )
                
                # Style details
                st.write("Style Elements:", ", ".join(analysis["style_elements"]))
                st.write("Suitable for:", ", ".join(analysis["occasions"]))
                st.write("Key Pieces:", ", ".join(analysis["key_pieces"]))
                
                st.markdown("---")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        if all_analyses:
            st.write("## Overall Style Summary")
            
            # Show dominant color palette
            st.write("### Dominant Colors")
            color_counts = Counter([color['hex'] for color in all_colors])
            top_colors = color_counts.most_common(8)
            
            cols = st.columns(8)
            for i, (color, count) in enumerate(top_colors):
                cols[i].markdown(
                    f'<div style="background-color: {color}; height: 50px; border-radius: 5px;"></div>',
                    unsafe_allow_html=True
                )
            
            # Aggregate style elements
            all_elements = [elem for analysis in all_analyses for elem in analysis["style_elements"]]
            common_elements = Counter(all_elements).most_common(5)
            
            st.write("### Common Style Elements")
            for element, count in common_elements:
                st.write(f"- {element}")
            
            # Aggregate occasions
            all_occasions = [occ for analysis in all_analyses for occ in analysis["occasions"]]
            common_occasions = Counter(all_occasions).most_common(3)
            
            st.write("### Best Suited For")
            for occasion, count in common_occasions:
                st.write(f"- {occasion}")

st.markdown("---")
st.write("Note: For best results, use direct image URLs from Pinterest pins.")
