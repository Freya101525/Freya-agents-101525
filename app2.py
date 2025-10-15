import streamlit as st
import pandas as pd
import yaml
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai
from xai_sdk.client import Client
from xai_sdk.chat import user, system
from openai import OpenAI
import os
import io
import ast
import traceback
import re
from collections import Counter
import time
import base64
import json
try:
    import pytesseract
    from PIL import Image
    import pdf2image
    PYTESSERACT_AVAILABLE = True
except:
    PYTESSERACT_AVAILABLE = False
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except:
    EASYOCR_AVAILABLE = False

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced Agentic Analysis System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Enhanced Theme Definitions ---
themes = {
    "Blue Sky": {
        "primaryColor": "#00BFFF",
        "backgroundColor": "#E6F3FF",
        "secondaryBackgroundColor": "#B3D9FF",
        "textColor": "#003366",
        "accentColor": "#0080FF"
    },
    "Snow White": {
        "primaryColor": "#A0A0A0",
        "backgroundColor": "#FFFFFF",
        "secondaryBackgroundColor": "#F8F8F8",
        "textColor": "#2C2C2C",
        "accentColor": "#D0D0D0"
    },
    "Sparkling Stars": {
        "primaryColor": "#FFD700",
        "backgroundColor": "#0A0E27",
        "secondaryBackgroundColor": "#1A1F3A",
        "textColor": "#E0E0E0",
        "accentColor": "#FFE55C"
    },
    "Alps Forest": {
        "primaryColor": "#228B22",
        "backgroundColor": "#F0FFF0",
        "secondaryBackgroundColor": "#D4EDD4",
        "textColor": "#1B4D1B",
        "accentColor": "#32CD32"
    },
    "Flora Garden": {
        "primaryColor": "#FF69B4",
        "backgroundColor": "#FFF0F5",
        "secondaryBackgroundColor": "#FFE4E9",
        "textColor": "#8B008B",
        "accentColor": "#FF1493"
    },
    "Fresh Air": {
        "primaryColor": "#00CED1",
        "backgroundColor": "#F0FFFF",
        "secondaryBackgroundColor": "#E0F7F7",
        "textColor": "#006B6F",
        "accentColor": "#40E0D0"
    },
    "Deep Ocean": {
        "primaryColor": "#00FFFF",
        "backgroundColor": "#001F3F",
        "secondaryBackgroundColor": "#003366",
        "textColor": "#B0E0E6",
        "accentColor": "#1E90FF"
    },
    "Ferrari Sportscar": {
        "primaryColor": "#FF2800",
        "backgroundColor": "#0D0D0D",
        "secondaryBackgroundColor": "#1A1A1A",
        "textColor": "#FFFFFF",
        "accentColor": "#FF6347"
    },
    "Fendi Casa Luxury": {
        "primaryColor": "#C9A87C",
        "backgroundColor": "#FBF8F3",
        "secondaryBackgroundColor": "#F5EFE6",
        "textColor": "#4A3F35",
        "accentColor": "#D4AF77"
    }
}

# --- State Initialization ---
if 'theme' not in st.session_state:
    st.session_state.theme = "Blue Sky"
if 'last_agent_output' not in st.session_state:
    st.session_state.last_agent_output = ""
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        "Gemini": os.getenv("GEMINI_API_KEY", ""),
        "OpenAI": os.getenv("OPENAI_API_KEY", ""),
        "Grok": os.getenv("XAI_API_KEY", "")
    }
if 'combined_markdown' not in st.session_state:
    st.session_state.combined_markdown = ""
if 'article2_markdown' not in st.session_state:
    st.session_state.article2_markdown = ""
if 'mind_map_relationships' not in st.session_state:
    st.session_state.mind_map_relationships = []
if 'entity_categories' not in st.session_state:
    st.session_state.entity_categories = [
        "Person", "Organization", "Location", "Technology", 
        "Concept", "Product", "Event", "Date", "Metric", "Other"
    ]

# Apply theme with enhanced styling
current_theme = themes[st.session_state.theme]
st.markdown(f"""
<style>
    .stApp {{
        background-color: {current_theme['backgroundColor']};
        color: {current_theme['textColor']};
    }}
    .coral-keyword {{
        color: #FF7F50;
        font-weight: bold;
    }}
    .fancy-header {{
        background: linear-gradient(90deg, {current_theme['primaryColor']}, {current_theme['accentColor']});
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-size: 2em;
        font-weight: bold;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    .progress-container {{
        background-color: {current_theme['secondaryBackgroundColor']};
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }}
    .export-button {{
        background-color: {current_theme['primaryColor']};
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# --- Fancy Progress Indicator ---
def show_fancy_progress(message, duration=2):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.markdown(f"**{message}** {'.' * ((i // 25) % 4)}")
        time.sleep(duration / 100)
    progress_bar.empty()
    status_text.empty()

# --- Export Functions ---
def create_download_link(content, filename, file_type="text"):
    if file_type == "text":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "text/plain"
    elif file_type == "csv":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "text/csv"
    elif file_type == "json":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "application/json"
    elif file_type == "html":
        b64 = base64.b64encode(content.encode()).decode()
        mime = "text/html"
    
    return f'<a href="data:{mime};base64,{b64}" download="{filename}" class="export-button">⬇️ Download {filename}</a>'

def export_combined_markdown():
    return st.session_state.combined_markdown

def export_entities_csv(entities_md):
    # Parse markdown table to CSV
    lines = entities_md.strip().split('\n')
    csv_content = ""
    for line in lines:
        if '|' in line and not line.startswith('|---'):
            csv_content += ','.join([cell.strip() for cell in line.split('|')[1:-1]]) + '\n'
    return csv_content

def export_mind_map_json():
    data = {
        "nodes": list(set([r[0] for r in st.session_state.mind_map_relationships] + 
                         [r[1] for r in st.session_state.mind_map_relationships])),
        "edges": [{"source": r[0], "target": r[1], "relation": r[2]} 
                  for r in st.session_state.mind_map_relationships]
    }
    return json.dumps(data, indent=2)

# --- API Call Functions ---
def call_llm_api(provider, api_key, model_name, prompt, system_prompt="You are a helpful AI assistant."):
    if not api_key:
        return f"Error: {provider} API key not provided"
    
    if provider == "Gemini":
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Gemini API Error: {e}"
    
    elif provider == "OpenAI":
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API Error: {e}"
    
    elif provider == "Grok":
        try:
            client = Client(api_key=api_key, timeout=3600)
            chat = client.chat.create(model=model_name)
            chat.append(system(system_prompt))
            chat.append(user(prompt))
            response = chat.sample()
            return response.content
        except Exception as e:
            return f"Grok API Error: {e}"

# --- OCR Functions ---
def ocr_pdf_pytesseract(pdf_file, pages):
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        images = pdf2image.convert_from_bytes(pdf_bytes, first_page=pages[0], last_page=pages[-1])
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img) + "\n"
        return text
    except Exception as e:
        return f"OCR Error: {e}"

def ocr_pdf_easyocr(pdf_file, pages):
    try:
        pdf_bytes = pdf_file.read()
        pdf_file.seek(0)  # Reset file pointer
        reader = easyocr.Reader(['en'])
        images = pdf2image.convert_from_bytes(pdf_bytes, first_page=pages[0], last_page=pages[-1])
        text = ""
        for img in images:
            result = reader.readtext(img)
            text += " ".join([item[1] for item in result]) + "\n"
        return text
    except Exception as e:
        return f"OCR Error: {e}"

def ocr_pdf_llm(provider, api_key, model_name, pdf_file, pages):
    return "LLM-based OCR requires additional image processing. Please use pytesseract or easyocr."

# --- Text Processing Functions ---
def extract_keywords(text, top_n=50):
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    stop_words = {'that', 'this', 'with', 'from', 'have', 'been', 'were', 'will', 'would', 'could', 'should'}
    words = [w for w in words if w not in stop_words]
    word_freq = Counter(words)
    return [word for word, _ in word_freq.most_common(top_n)]

def highlight_keywords(text, keywords):
    for keyword in keywords:
        pattern = re.compile(r'\b' + re.escape(keyword) + r'\b', re.IGNORECASE)
        text = pattern.sub(f'<span class="coral-keyword">{keyword}</span>', text)
    return text

def create_entities_prompt(text, categories):
    categories_str = ", ".join(categories)
    prompt = f"""Extract exactly 100 unique entities from the following text. 
    Return them as a markdown table with columns: Entity | Type | Description
    
    Available types: {categories_str}
    
    Format example:
    | Entity | Type | Description |
    |--------|------|-------------|
    | Python | Technology | Programming language |
    
    Text (first 8000 chars):
    {text[:8000]}
    
    Return ONLY the markdown table, no other text.
    """
    return prompt

# --- Mind Map Functions ---
def create_interactive_mindmap(relationships):
    if not relationships:
        return "<p>No relationships to display</p>"
    
    G = nx.DiGraph()
    for source, target, relation in relationships:
        G.add_edge(source, target, title=relation, label=relation[:20])
    
    net = Network(height="600px", width="100%", notebook=True, directed=True,
                  bgcolor=themes[st.session_state.theme]['backgroundColor'])
    net.from_nx(G)
    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -8000,
                "springLength": 250
            }
        }
    }
    """)
    net.show("mindmap.html")
    return open("mindmap.html", 'r', encoding='utf-8').read()

@st.cache_data
def load_agents():
    try:
        with open("agents.yaml", 'r') as stream:
            return yaml.safe_load(stream)
    except FileNotFoundError:
        st.error("agents.yaml not found. Please create the file.")
        return {"agents": []}
    except Exception as e:
        st.error(f"Error loading agents.yaml: {e}")
        return {"agents": []}

# --- Sidebar UI ---
with st.sidebar:
    st.markdown('<div class="fancy-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    st.subheader("🎨 Theme Selection")
    st.selectbox("Choose Your Theme", options=list(themes.keys()), key="theme")
    
    st.divider()
    st.subheader("🔑 API Configuration")
    
    with st.expander("Gemini API", expanded=False):
        st.session_state.api_keys["Gemini"] = st.text_input(
            "Gemini API Key", 
            type="password", 
            value=st.session_state.api_keys.get('Gemini', ''),
            key="gemini_key",
            help="Get your key from https://makersuite.google.com/app/apikey"
        )
    
    with st.expander("OpenAI API", expanded=False):
        st.session_state.api_keys["OpenAI"] = st.text_input(
            "OpenAI API Key", 
            type="password", 
            value=st.session_state.api_keys.get('OpenAI', ''),
            key="openai_key",
            help="Get your key from https://platform.openai.com/api-keys"
        )
    
    with st.expander("Grok API", expanded=False):
        st.session_state.api_keys["Grok"] = st.text_input(
            "XAI (Grok) API Key", 
            type="password", 
            value=st.session_state.api_keys.get('Grok', ''),
            key="grok_key",
            help="Get your key from https://x.ai"
        )
    
    st.divider()
    st.subheader("🏷️ Entity Categories")
    categories_text = st.text_area(
        "Edit categories (one per line)",
        "\n".join(st.session_state.entity_categories),
        height=200
    )
    if st.button("Update Categories"):
        st.session_state.entity_categories = [c.strip() for c in categories_text.split('\n') if c.strip()]
        st.success("Categories updated!")

# --- Main Page ---
st.markdown('<div class="fancy-header">✨ Enhanced Agentic Analysis System</div>', unsafe_allow_html=True)

# --- Document Upload Section ---
st.header("📄 Document Processing")

num_docs = st.number_input("How many documents to process?", min_value=1, max_value=10, value=1)

documents = []
for i in range(num_docs):
    with st.expander(f"📑 Document {i+1}", expanded=(i==0)):
        input_method = st.radio(f"Input method for Doc {i+1}", 
                                ['Upload', 'Paste'], 
                                key=f"method_{i}",
                                horizontal=True)
        
        if input_method == 'Upload':
            uploaded = st.file_uploader(
                f"Upload file (txt, markdown, pdf)",
                type=["txt", "md", "pdf"],
                key=f"upload_{i}"
            )
            if uploaded:
                if uploaded.type == "application/pdf":
                    st.subheader("PDF OCR Configuration")
                    ocr_method = st.selectbox(
                        "OCR Method",
                        ["pytesseract" if PYTESSERACT_AVAILABLE else "pytesseract (not installed)", 
                         "easyocr" if EASYOCR_AVAILABLE else "easyocr (not installed)", 
                         "LLM"],
                        key=f"ocr_{i}"
                    )
                    page_range = st.text_input(
                        "Pages to OCR (e.g., 1-5 or 1,3,5)",
                        "1-5",
                        key=f"pages_{i}"
                    )
                    
                    if st.button(f"🔄 Process PDF {i+1}", key=f"process_{i}"):
                        with st.spinner("🔄 Processing PDF..."):
                            show_fancy_progress("Extracting text from PDF", 3)
                            
                            # Parse page range
                            if '-' in page_range:
                                start, end = map(int, page_range.split('-'))
                                pages = list(range(start, end + 1))
                            else:
                                pages = [int(p.strip()) for p in page_range.split(',')]
                            
                            if "pytesseract" in ocr_method and PYTESSERACT_AVAILABLE:
                                text = ocr_pdf_pytesseract(uploaded, pages)
                            elif "easyocr" in ocr_method and EASYOCR_AVAILABLE:
                                text = ocr_pdf_easyocr(uploaded, pages)
                            else:
                                text = "OCR library not available or selected LLM method not implemented"
                            
                            documents.append(text)
                            st.success(f"✅ PDF {i+1} processed successfully!")
                            st.text_area(f"Preview of Doc {i+1}", text[:500] + "...", height=150)
                else:
                    text = uploaded.read().decode('utf-8')
                    documents.append(text)
                    st.success(f"✅ Document {i+1} loaded!")
        else:
            pasted = st.text_area(f"Paste text for Doc {i+1}", height=200, key=f"paste_{i}")
            if pasted:
                documents.append(pasted)
                st.success(f"✅ Text added for Doc {i+1}!")

# --- Combine Documents ---
if st.button("🔗 Combine All Documents", use_container_width=True) and documents:
    with st.spinner("Combining documents..."):
        show_fancy_progress("Analyzing and combining documents", 2)
        
        combined_text = "\n\n---\n\n".join([f"## Document {i+1}\n\n{doc}" for i, doc in enumerate(documents)])
        keywords = extract_keywords(combined_text)
        
        st.session_state.combined_markdown = f"# Combined Document Analysis\n\n{highlight_keywords(combined_text, keywords)}"
        st.session_state.raw_combined = combined_text
        st.success("✅ Documents combined successfully!")

# --- Display Combined Document ---
if st.session_state.combined_markdown:
    st.header("📋 Combined Document")
    
    # Export button for combined markdown
    st.markdown(
        create_download_link(
            st.session_state.raw_combined, 
            "combined_document.md", 
            "text"
        ),
        unsafe_allow_html=True
    )
    
    with st.expander("View Combined Document", expanded=False):
        st.markdown(st.session_state.combined_markdown, unsafe_allow_html=True)
    
    # --- Summary Generation ---
    st.subheader("📝 Comprehensive Summary")
    col1, col2 = st.columns(2)
    summary_provider = col1.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], key="sum_prov")
    
    if summary_provider == "Gemini":
        summary_model = col2.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"], key="sum_model")
    elif summary_provider == "OpenAI":
        summary_model = col2.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5-nano"], key="sum_model")
    else:
        summary_model = col2.selectbox("Model", ["grok-4", "grok-4-fast-reasoning"], key="sum_model")
    
    if st.button("📄 Generate Summary", use_container_width=True):
        with st.spinner("Generating summary..."):
            show_fancy_progress("AI is analyzing the document", 3)
            summary = call_llm_api(
                summary_provider,
                st.session_state.api_keys[summary_provider],
                summary_model,
                f"Create a comprehensive markdown summary with key points, themes, and insights:\n\n{st.session_state.raw_combined[:15000]}"
            )
            st.session_state.summary = summary
    
    if 'summary' in st.session_state:
        st.markdown(st.session_state.summary)
        st.markdown(
            create_download_link(
                st.session_state.summary,
                "summary.md",
                "text"
            ),
            unsafe_allow_html=True
        )
    
    # --- Word Cloud ---
    st.subheader("☁️ Word Cloud Visualization")
    if st.button("🎨 Generate Word Cloud", use_container_width=True):
        try:
            with st.spinner("Creating word cloud..."):
                show_fancy_progress("Generating visualization", 2)
                text = re.sub('<[^<]+?>', '', st.session_state.combined_markdown)
                wordcloud = WordCloud(
                    width=1200, 
                    height=600,
                    background_color=themes[st.session_state.theme]['backgroundColor'],
                    colormap='viridis',
                    max_words=100
                ).generate(text)
                
                fig, ax = plt.subplots(figsize=(15, 8))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig)
                
                # Save and provide download
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
                buf.seek(0)
                st.download_button(
                    "⬇️ Download Word Cloud",
                    buf,
                    "wordcloud.png",
                    "image/png",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error generating word cloud: {e}")
    
    # --- Entity Extraction ---
    st.subheader("🏷️ Entity Extraction (100 Unique Entities)")
    
    if st.button("🔍 Extract Entities", use_container_width=True):
        with st.spinner("Extracting entities..."):
            show_fancy_progress("Identifying entities", 3)
            entity_prompt = create_entities_prompt(
                st.session_state.raw_combined,
                st.session_state.entity_categories
            )
            entities = call_llm_api(
                summary_provider,
                st.session_state.api_keys[summary_provider],
                summary_model,
                entity_prompt
            )
            st.session_state.entities = entities
    
    if 'entities' in st.session_state:
        st.markdown(st.session_state.entities)
        
        # Export entities
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                create_download_link(
                    st.session_state.entities,
                    "entities.md",
                    "text"
                ),
                unsafe_allow_html=True
            )
        with col2:
            csv_entities = export_entities_csv(st.session_state.entities)
            st.markdown(
                create_download_link(
                    csv_entities,
                    "entities.csv",
                    "csv"
                ),
                unsafe_allow_html=True
            )

# --- Agent Execution ---
st.header("🤖 Agentic Execution Pipeline")
agents = load_agents().get('agents', [])

if agents:
    st.info(f"📊 {len(agents)} agents available in agents.yaml")
    num_agents = st.slider("Number of agents to execute", 1, min(len(agents), 31), 1)
    
    if 'current_agent_index' not in st.session_state:
        st.session_state.current_agent_index = 0
    
    if st.session_state.current_agent_index < num_agents:
        agent = agents[st.session_state.current_agent_index]
        
        st.subheader(f"🎯 Agent {st.session_state.current_agent_index + 1}/{num_agents}: {agent['name']}")
        st.info(f"**Category:** {agent.get('category', 'General')} | **Description:** {agent['description']}")
        
        col1, col2 = st.columns(2)
        agent_provider = col1.selectbox("Provider", ["Gemini", "OpenAI", "Grok"], key="agent_prov")
        
        if agent_provider == "Gemini":
            agent_model = col2.selectbox("Model", ["gemini-2.5-flash", "gemini-2.5-flash-lite"], key="agent_model")
        elif agent_provider == "OpenAI":
            agent_model = col2.selectbox("Model", ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5-nano"], key="agent_model")
        else:
            agent_model = col2.selectbox("Model", ["grok-4", "grok-4-fast-reasoning"], key="agent_model")
        
        prompt = st.text_area("Agent Prompt", agent.get('prompt', ''), height=150, key="agent_prompt")
        
        use_previous = st.checkbox("Use previous agent's output as input", value=True)
        
        col_exec, col_next, col_skip = st.columns(3)
        if col_exec.button("▶️ Execute Agent", use_container_width=True):
            with st.spinner("Agent working..."):
                show_fancy_progress(f"Executing {agent['name']}", 2)
                input_data = st.session_state.last_agent_output if (use_previous and st.session_state.last_agent_output) else st.session_state.raw_combined
                result = call_llm_api(
                    agent_provider,
                    st.session_state.api_keys[agent_provider],
                    agent_model,
                    f"{prompt}\n\nData:\n{input_data[:10000]}"
                )
                st.session_state.last_agent_output = result
                st.session_state[f'agent_{st.session_state.current_agent_index}_output'] = result
        
        if col_next.button("⏭️ Next Agent", use_container_width=True):
            st.session_state.current_agent_index += 1
            st.rerun()
        
        if col_skip.button("⏩ Skip Agent", use_container_width=True):
            st.session_state.current_agent_index += 1
            st.rerun()
        
        if st.session_state.last_agent_output:
            st.success("✅ Agent Output:")
            st.markdown(st.session_state.last_agent_output)
            
            # Download agent output
            st.download_button(
                "⬇️ Download Agent Output",
                st.session_state.last_agent_output,
                f"agent_{st.session_state.current_agent_index+1}_output.md",
                use_container_width=True
            )
    else:
        st.success("✅ All agents executed successfully!")
        
        # Show all agent outputs
        with st.expander("📊 View All Agent Outputs"):
            for i in range(num_agents):
                if f'agent_{i}_output' in st.session_state:
                    st.markdown(f"### Agent {i+1}: {agents[i]['name']}")
                    st.markdown(st.session_state[f'agent_{i}_output'])
                    st.markdown("---")
        
        col1, col2 = st.columns(2)
        if col1.button("🔄 Restart Pipeline", use_container_width=True):
            st.session_state.current_agent_index = 0
            st.session_state.last_agent_output = ""
            st.rerun()
        
        if col2.button("💾 Export All Results", use_container_width=True):
            all_results = ""
            for i in range(num_agents):
                if f'agent_{i}_output' in st.session_state:
                    all_results += f"# Agent {i+1}: {agents[i]['name']}\n\n"
                    all_results += st.session_state[f'agent_{i}_output'] + "\n\n---\n\n"
            
            st.download_button(
                "⬇️ Download All Agent Outputs",
                all_results,
                "all_agent_outputs.md",
                use_container_width=True
            )
else:
    st.warning("⚠️ No agents found. Please create agents.yaml file.")

# --- Article 2 & Mind Map ---
st.header("🧠 Mind Map Generation")

article2_input = st.text_area("Paste Article 2 (txt or markdown)", height=200)

if st.button("Process Article 2") and article2_input:
    with st.spinner("Processing Article 2..."):
        show_fancy_progress("Analyzing Article 2", 2)
        keywords2 = extract_keywords(article2_input)
        st.session_state.article2_markdown = highlight_keywords(article2_input, keywords2)
        st.success("✅ Article 2 processed!")

if st.session_state.article2_markdown:
    st.markdown(st.session_state.article2_markdown, unsafe_allow_html=True)
    
    if st.button("Generate Relationships"):
        with st.spinner("Finding relationships..."):
            show_fancy_progress("Mapping connections", 3)
            prompt = f"""Analyze these two documents and create relationships for a mind map.
            Return as: (Source, Target, Relationship) tuples, one per line.
            
            Document 1: {st.session_state.combined_markdown[:5000]}
            Document 2: {st.session_state.article2_markdown[:5000]}
            """
            result = call_llm_api(
                "Gemini",
                st.session_state.api_keys["Gemini"],
                "gemini-2.5-flash",
                prompt
            )
            
            # Parse relationships
            relationships = []
            for line in result.split('\n'):
                if ',' in line:
                    parts = [p.strip().strip('()') for p in line.split(',')]
                    if len(parts) >= 3:
                        relationships.append(tuple(parts[:3]))
            
            st.session_state.mind_map_relationships = relationships
    
    if st.session_state.mind_map_relationships:
        st.subheader("Edit Relationships")
        edited_rels = st.text_area(
            "Relationships (Source, Target, Relation)",
            "\n".join([f"{s}, {t}, {r}" for s, t, r in st.session_state.mind_map_relationships]),
            height=200
        )
        
        if st.button("Update Mind Map"):
            # Parse edited relationships
            new_rels = []
            for line in edited_rels.split('\n'):
                if ',' in line:
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        new_rels.append(tuple(parts[:3]))
            st.session_state.mind_map_relationships = new_rels
            
            # Create mind map
            with st.spinner("Creating interactive mind map..."):
                show_fancy_progress("Building visualization", 2)
                html = create_interactive_mindmap(st.session_state.mind_map_relationships)
                st.components.v1.html(html, height=650)

st.markdown("---")
st.markdown("### 💡 Have questions? Need help?")
st.markdown("Feel free to explore all features and customize to your needs!")
