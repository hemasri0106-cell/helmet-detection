import gradio as gr
import config
from inference import predict_image, predict_video
from ultralytics import YOLO
from pathlib import Path

print("--- APP STARTUP VERIFICATION ---")
print(f"Loaded model: {config.MODEL_PATH}")
try:
    _startup_model = YOLO(config.MODEL_PATH)
    print(f"Model names: {_startup_model.names}")
except:
    pass
print("--------------------------------\n")

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-dark: #0D0D0D;
    --bg-sec: #161616;
    --card-bg: #1F1F1F;
    --border: #2E2E2E;
    --text-primary: #FFFFFF;
    --text-secondary: #A0A0A0;
    --accent: #D9D9D9;
}

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background-color: var(--bg-dark) !important;
    color: var(--text-primary) !important;
}

/* Base Spacing */
.gradio-container {
    padding-top: 1rem !important;
    max-width: 1500px !important;
}

/* Premium Navigation Tabs */
.tabs > div > button {
    background-color: transparent !important;
    border: none !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    padding: 1.5rem 2.5rem !important;
    font-size: 1.15rem !important;
    border-bottom: 3px solid transparent !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.5px;
}
.tabs > div > button:hover {
    color: var(--text-primary) !important;
}
.tabs > div > button.selected {
    color: var(--text-primary) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Hide default borders */
.gradio-container .gr-box, .gradio-container .gr-block {
    background-color: transparent !important;
    border: none !important;
}
.gradio-container .gr-panel {
    background: transparent !important;
}

/* Base card styling */
.premium-card {
    background-color: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    box-shadow: 0 10px 30px rgba(0,0,0,0.4) !important;
    transition: all 0.3s ease !important;
    overflow: hidden;
}
.premium-card:hover {
    transform: translateY(-4px) !important;
    border-color: #444 !important;
    box-shadow: 0 15px 40px rgba(0,0,0,0.6) !important;
}

/* Buttons */
button.primary {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    border: 1px solid #444 !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600 !important;
    padding: 1rem 2rem !important;
    font-size: 1rem !important;
}
button.primary:hover {
    background-color: #1F1F1F !important;
    border-color: var(--accent) !important;
    transform: translateY(-2px);
}

/* Sidebar Sticky */
.sidebar-sticky {
    position: sticky !important;
    top: 2rem;
    align-self: flex-start;
}

/* Hero Section */
.hero-wrapper {
    position: relative;
    text-align: center;
    padding: 6rem 2rem;
    background: radial-gradient(circle at 50% 0%, #1a1a1a 0%, var(--bg-dark) 70%);
    border-radius: 24px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-bg-pattern {
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(#333 1px, transparent 1px);
    background-size: 32px 32px;
    opacity: 0.15;
    z-index: 0;
}
.hero-content {
    position: relative;
    z-index: 1;
}
.hero-content h1 {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 1rem;
    color: var(--text-primary);
}
.hero-content p {
    font-size: 1.15rem;
    color: var(--text-secondary);
    max-width: 650px;
    margin: 0 auto 2.5rem auto;
    line-height: 1.6;
    font-weight: 300;
}
.hero-buttons {
    display: flex;
    gap: 1.5rem;
    justify-content: center;
    margin-top: -6rem;
    position: relative;
    z-index: 2;
    margin-bottom: 3rem;
}

/* Section Headings */
.section-heading {
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 1.5rem;
    color: var(--text-primary);
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.8rem;
}

/* Metrics Grid */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1.5rem;
    margin-bottom: 3rem;
}
.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    text-align: left;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: all 0.3s ease;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.metric-card:hover {
    transform: translateY(-5px);
    border-color: #555;
}
.metric-icon svg {
    width: 28px;
    height: 28px;
    fill: var(--text-secondary);
    margin-bottom: 1rem;
}
.metric-label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.metric-value {
    font-size: 2.8rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-top: 0.5rem;
}

/* Workflow Container */
.workflow-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg-sec);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid var(--border);
    margin-bottom: 3rem;
}
.workflow-step {
    text-align: center;
    flex: 1;
    padding: 0.5rem;
    transition: all 0.3s ease;
    border-radius: 12px;
}
.workflow-step:hover {
    background: var(--card-bg);
    transform: translateY(-3px);
}
.wf-icon svg {
    width: 32px;
    height: 32px;
    fill: var(--text-primary);
    margin-bottom: 0.8rem;
}
.wf-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.wf-desc {
    font-size: 0.8rem;
    color: var(--text-secondary);
    line-height: 1.4;
}
.wf-arrow svg {
    width: 20px;
    height: 20px;
    fill: #444;
}

/* Sidebar Analytics Widget (Quick Stats) */
.analytics-widget {
    background: var(--bg-sec);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 2rem;
}
.analytics-row {
    display: flex;
    justify-content: space-between;
    padding: 0.8rem 0;
    border-bottom: 1px solid #222;
}
.analytics-row:last-child {
    border-bottom: none;
}
.analytics-label {
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.95rem;
}
.analytics-value {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.05rem;
}

/* Sidebar Project Info Cards */
.info-col {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 2rem;
}
.info-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: left;
    transition: all 0.2s ease;
}
.info-card:hover {
    border-color: #444;
    transform: translateX(4px);
}
.info-label {
    display: block;
    font-size: 0.75rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 0.4rem;
}
.info-value {
    display: block;
    font-size: 1.15rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* Prediction Summary Cards (Inference HTML) */
.pred-summary {
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-top: 1.5rem;
}
.pred-stat-card {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--bg-sec);
    border: 1px solid var(--border);
    padding: 1rem 1.2rem;
    border-radius: 12px;
}
.pred-stat-label {
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 0.95rem;
}
.pred-stat-value {
    color: var(--text-primary);
    font-weight: 700;
    font-size: 1.15rem;
}

/* General Layout Fixes */
.hide-label > label { display: none !important; }

/* Sample Image Cards */
.sample-card {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3) !important;
    transition: all 0.3s ease !important;
}
.sample-card:hover {
    transform: translateY(-5px) !important;
    border-color: #555 !important;
}
.sample-text {
    padding: 1rem 0.5rem 0.5rem !important;
}
.sample-text h4 {
    margin: 0 0 0.4rem 0;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
}
.sample-text p {
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.4;
}

/* Accordion overrides */
.gradio-container .gr-accordion {
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
"""

theme = gr.themes.Base(
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0D0D0D",
    block_background_fill="transparent",
    block_border_width="0px",
    button_primary_background_fill="#000000",
    button_primary_background_fill_hover="#111111",
    button_primary_text_color="#FFFFFF",
    body_text_color="#FFFFFF"
)

# --- SVG ICONS ---
SVG_TARGET = '<svg viewBox="0 0 24 24"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm0-14c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z"/></svg>'
SVG_TREND = '<svg viewBox="0 0 24 24"><path d="M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z"/></svg>'
SVG_BAR = '<svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/></svg>'
SVG_IMAGE = '<svg viewBox="0 0 24 24"><path d="M21 19V5c0-1.1-.9-2-2-2H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2zM8.5 13.5l2.5 3.01L14.5 12l4.5 6H5l3.5-4.5z"/></svg>'
SVG_BRAIN = '<svg viewBox="0 0 24 24"><path d="M13 3c-4.97 0-9 4.03-9 9 0 2.12.74 4.07 1.97 5.61L3 21l3.39-2.97C7.93 19.26 9.88 20 12 20c4.97 0 9-4.03 9-9s-4.03-9-9-9zm0 15c-3.31 0-6-2.69-6-6s2.69-6 6-6 6 2.69 6 6-2.69 6-6 6z"/></svg>'
SVG_BOX = '<svg viewBox="0 0 24 24"><path d="M3 3v18h18V3H3zm16 16H5V5h14v14z"/></svg>'
SVG_ARROW = '<svg viewBox="0 0 24 24"><path d="M12 4l-1.41 1.41L16.17 11H4v2h12.17l-5.58 5.59L12 20l8-8z"/></svg>'
SVG_LAYERS = '<svg viewBox="0 0 24 24"><path d="M11.99 18.54l-7.37-5.73L3 14.07l9 7 9-7-1.63-1.27-7.38 5.74zM12 16l7.36-5.73L21 9l-9-7-9 7 1.63 1.27L12 16z"/></svg>'

# --- HTML WRAPPERS ---
def html_metrics():
    return f"""
    <h2 class="section-heading">Evaluation Metrics</h2>
    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-icon">{SVG_TARGET}</div>
            <div class="metric-label">Precision</div>
            <div class="metric-value">{config.EVALUATION_METRICS['Precision']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">{SVG_TREND}</div>
            <div class="metric-label">Recall</div>
            <div class="metric-value">{config.EVALUATION_METRICS['Recall']}</div>
        </div>
        <div class="metric-card">
            <div class="metric-icon">{SVG_BAR}</div>
            <div class="metric-label">mAP@0.5</div>
            <div class="metric-value">{config.EVALUATION_METRICS['mAP@0.5']}</div>
        </div>
    </div>
    """

def html_workflow():
    return f"""
    <h2 class="section-heading">System Workflow</h2>
    <div class="workflow-container">
        <div class="workflow-step"><div class="wf-icon">{SVG_IMAGE}</div><div class="wf-title">Input Media</div><div class="wf-desc">Upload Image/Video</div></div>
        <div class="wf-arrow">{SVG_ARROW}</div>
        <div class="workflow-step"><div class="wf-icon">{SVG_BRAIN}</div><div class="wf-title">YOLOv8 Inference</div><div class="wf-desc">Deep Learning Model</div></div>
        <div class="wf-arrow">{SVG_ARROW}</div>
        <div class="workflow-step"><div class="wf-icon">{SVG_TARGET}</div><div class="wf-title">Detection</div><div class="wf-desc">Class Probability</div></div>
        <div class="wf-arrow">{SVG_ARROW}</div>
        <div class="workflow-step"><div class="wf-icon">{SVG_BOX}</div><div class="wf-title">Bounding Boxes</div><div class="wf-desc">Spatial Mapping</div></div>
        <div class="wf-arrow">{SVG_ARROW}</div>
        <div class="workflow-step"><div class="wf-icon">{SVG_LAYERS}</div><div class="wf-title">Results</div><div class="wf-desc">Prediction Summary</div></div>
    </div>
    """

def html_sidebar_stats():
    return f"""
    <div class="analytics-widget">
        <div class="analytics-row"><span class="analytics-label">Precision</span><span class="analytics-value">{config.EVALUATION_METRICS['Precision']}</span></div>
        <div class="analytics-row"><span class="analytics-label">Recall</span><span class="analytics-value">{config.EVALUATION_METRICS['Recall']}</span></div>
        <div class="analytics-row"><span class="analytics-label">mAP@0.5</span><span class="analytics-value">{config.EVALUATION_METRICS['mAP@0.5']}</span></div>
        <div class="analytics-row" style="border:none;"><span class="analytics-label">Model</span><span class="analytics-value">YOLOv8n</span></div>
    </div>
    """

def html_sidebar_project_info():
    rows = ""
    for k, v in config.PROJECT_INFO.items():
        rows += f"<div class='info-card'><span class='info-label'>{k}</span><span class='info-value'>{v}</span></div>"
    return f"<div class='info-col'>{rows}</div>"

def html_summary(summary, is_video=False):
    if "Error" in summary:
        return "<div class='pred-stat-card' style='border-color:#ff4444;'><span class='pred-stat-label'>Error</span><span class='pred-stat-value'>Model not loaded</span></div>"
    
    if is_video:
        return f"""
        <div class="pred-summary">
            <div class="pred-stat-card"><span class="pred-stat-label">Frames Processed</span><span class="pred-stat-value">{summary.get('Frames Processed', 0)}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Helmet Detections</span><span class="pred-stat-value">{summary.get('Helmet Detections', 0)}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">No Helmet Detections</span><span class="pred-stat-value">{summary.get('No Helmet Detections', 0)}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Average Confidence</span><span class="pred-stat-value">{summary.get('Average Confidence', '0.0')}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Processing Time</span><span class="pred-stat-value">{summary.get('Inference Time', '0s')}</span></div>
        </div>
        """
    else:
        tot = summary.get('With Helmet', 0) + summary.get('Without Helmet', 0)
        return f"""
        <div class="pred-summary">
            <div class="pred-stat-card"><span class="pred-stat-label">Detected Objects</span><span class="pred-stat-value">{tot}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">With Helmet</span><span class="pred-stat-value">{summary.get('With Helmet', 0)}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Without Helmet</span><span class="pred-stat-value">{summary.get('Without Helmet', 0)}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Average Confidence</span><span class="pred-stat-value">{summary.get('Average Confidence', '0.0')}</span></div>
            <div class="pred-stat-card"><span class="pred-stat-label">Inference Time</span><span class="pred-stat-value">{summary.get('Inference Time', '0s')}</span></div>
        </div>
        """

# --- INFERENCE WRAPPERS ---
def handle_image(img_path):
    if not img_path:
        return None, ""
    out_img, summary = predict_image(img_path)
    return out_img, html_summary(summary, is_video=False)

def handle_video(vid_path):
    if not vid_path:
        return None, ""
    out_vid, summary = predict_video(vid_path)
    return out_vid, html_summary(summary, is_video=True)

# --- UI DEFINITION ---
with gr.Blocks(theme=theme, css=custom_css, title="Helmet Detection AI") as app:
    
    with gr.Tabs(elem_classes="tabs") as tabs:
        
        with gr.Tab("Dashboard", id=0):
            with gr.Row():
                # LEFT COLUMN (70%)
                with gr.Column(scale=7):
                    # 1. Hero
                    gr.HTML("""
                    <div class="hero-wrapper">
                        <div class="hero-bg-pattern"></div>
                        <div class="hero-content">
                            <h1>Helmet Detection using YOLOv8</h1>
                            <p>An AI-powered system that detects whether riders are wearing helmets in images and videos. Experience a premium SaaS dashboard providing real-time analytics and high-precision bounding boxes.</p>
                        </div>
                    </div>
                    """)
                    with gr.Row(elem_classes="hero-buttons"):
                        btn_goto_img = gr.Button("Try Image Detection", variant="primary")
                        btn_goto_vid = gr.Button("Try Video Detection", variant="primary")
                        
                    # 2. Evaluation Metrics
                    gr.HTML(html_metrics())
                    
                    # 3. Sample Detection Results
                    gr.HTML("<h2 class='section-heading'>Sample Detections</h2>")
                    cached_images = list(config.CACHE_DIR.glob("dash_*.jpg"))
                    dash_names = {"dash_bus.jpg": "City Bus Scene", "dash_image10.jpg": "Helmet Compliance", "dash_test.jpg": "Motorcyclist"}
                    dash_descs = {
                        "dash_bus.jpg": "Multiple riders detected simultaneously in a complex city environment.",
                        "dash_image10.jpg": "High-confidence bounding boxes drawn on riders complying with safety regulations.",
                        "dash_test.jpg": "Accurate detection isolating the motorcyclist from the background."
                    }
                    if cached_images:
                        with gr.Row(elem_classes="metrics-grid"): # Re-use grid spacing
                            for img_path in cached_images[:4]:
                                title = dash_names.get(img_path.name.lower(), img_path.stem.replace('dash_', '').title())
                                desc = dash_descs.get(img_path.name.lower(), "YOLOv8 object detection model inference result.")
                                with gr.Column(elem_classes="sample-card"):
                                    gr.Image(value=str(img_path), interactive=False, show_label=False)
                                    gr.HTML(f"""
                                    <div class="sample-text">
                                        <h4>{title}</h4>
                                        <p>{desc}</p>
                                    </div>
                                    """)
                    
                    # 4. System Workflow
                    gr.HTML(html_workflow())
                
                # RIGHT COLUMN (30% STICKY)
                with gr.Column(scale=3, elem_classes="sidebar-sticky"):
                    gr.HTML("<h2 class='section-heading'>Quick Stats</h2>")
                    gr.HTML(html_sidebar_stats())
                    
                    gr.HTML("<h2 class='section-heading'>Project Info</h2>")
                    gr.HTML(html_sidebar_project_info())
                    
                    with gr.Accordion("About YOLOv8", open=False):
                        gr.Markdown("""
                        <div style="color:var(--text-secondary); line-height: 1.6; font-size: 0.95rem;">
                        <strong>YOLOv8</strong> is a state-of-the-art object detection model known for speed and accuracy. 
                        It processes images and videos in real-time, ideal for traffic monitoring and safety applications.
                        This project utilizes supervised learning, trained on a custom dataset explicitly labeled for safety compliance.
                        </div>
                        """)
            
        with gr.Tab("Image Detection", id=1):
            gr.HTML("<h2 class='section-heading'>Demo Gallery</h2>")
            demo_images = list(config.DEMO_IMAGES_DIR.glob("*.jpg"))
            demo_names = {"bus.jpg": "City Bus Scene", "image10.jpg": "Helmet Compliance", "test.jpg": "Motorcyclist"}
            with gr.Row(elem_classes="metrics-grid"):
                img_btns = []
                for img_path in demo_images[:4]:
                    with gr.Column(elem_classes="sample-card"):
                        gr.Image(value=str(img_path), interactive=False, show_label=False)
                        btn_name = demo_names.get(img_path.name.lower(), img_path.stem.title())
                        btn = gr.Button(f"Analyze {btn_name}", variant="primary")
                        img_btns.append((img_path, btn))

            gr.HTML("<br><h2 class='section-heading'>Run Inference</h2>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Original Image</h3>")
                    input_image = gr.Image(type="filepath", label="Upload Custom Image", elem_classes="premium-card")
                    infer_btn = gr.Button("Run YOLOv8 Detection", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Annotated Output</h3>")
                    output_image = gr.Image(interactive=False, elem_classes="premium-card")
                    
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Prediction Summary</h3>")
                    summary_html = gr.HTML("<div class='pred-stat-card'><span class='pred-stat-label'>Status</span><span class='pred-stat-value'>Awaiting input...</span></div>")

            for img_path, btn in img_btns:
                btn.click(
                    fn=lambda p=str(img_path): p, inputs=[], outputs=[input_image]
                ).then(
                    fn=handle_image, inputs=[input_image], outputs=[output_image, summary_html]
                )
            infer_btn.click(fn=handle_image, inputs=[input_image], outputs=[output_image, summary_html])
            
        with gr.Tab("Video Detection", id=2):
            gr.HTML("<h2 class='section-heading'>Demo Videos</h2>")
            demo_videos = list(config.DEMO_VIDEOS_DIR.glob("*.mp4"))
            demo_vid_names = {"busy market.mp4": "Busy Market", "city road ride.mp4": "City Road Ride", "highway ride.mp4": "Highway Ride"}
            with gr.Row(elem_classes="metrics-grid"):
                vid_btns = []
                for vid_path in demo_videos[:3]:
                    with gr.Column(elem_classes="sample-card"):
                        gr.Video(value=str(vid_path), interactive=False, show_label=False)
                        vid_name = demo_vid_names.get(vid_path.name.lower(), vid_path.stem.title())
                        btn = gr.Button(f"Process {vid_name}", variant="primary")
                        vid_btns.append((vid_path, btn))

            gr.HTML("<br><h2 class='section-heading'>Run Inference</h2>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Input Video</h3>")
                    input_video = gr.Video(label="Upload MP4", elem_classes="premium-card")
                    infer_vid_btn = gr.Button("Process Video Analytics", variant="primary")
                    
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Processed Output</h3>")
                    output_video = gr.Video(interactive=False, elem_classes="premium-card")
                    
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='margin-bottom:1rem; font-weight:600;'>Detection Analytics</h3>")
                    summary_vid_html = gr.HTML("<div class='pred-stat-card'><span class='pred-stat-label'>Status</span><span class='pred-stat-value'>Awaiting input...</span></div>")

            for vid_path, btn in vid_btns:
                btn.click(
                    fn=lambda p=str(vid_path): p, inputs=[], outputs=[input_video]
                ).then(
                    fn=handle_video, inputs=[input_video], outputs=[output_video, summary_vid_html]
                )
            infer_vid_btn.click(fn=handle_video, inputs=[input_video], outputs=[output_video, summary_vid_html])
            
    btn_goto_img.click(fn=lambda: gr.update(selected=1), outputs=[tabs])
    btn_goto_vid.click(fn=lambda: gr.update(selected=2), outputs=[tabs])
    
    gr.HTML("""
    <div style="text-align:center; padding: 3rem 0; color:var(--text-secondary); border-top: 1px solid var(--border); margin-top: 4rem; font-size: 0.95rem; font-weight: 500;">
        Helmet Detection using YOLOv8 &nbsp;&bull;&nbsp; Developed by Hemasri Challa &nbsp;&bull;&nbsp; <a href="https://github.com/hemasri0106-cell" target="_blank" style="color:var(--text-primary); text-decoration:underline;">GitHub</a> &nbsp;&bull;&nbsp; <a href="http://linkedin.com/in/hemasri-c-b622a7351/" target="_blank" style="color:var(--text-primary); text-decoration:underline;">LinkedIn</a>
    </div>
    """)

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
