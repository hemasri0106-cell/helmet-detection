import gradio as gr
import config
from inference import predict_image, predict_video
from ultralytics import YOLO
from pathlib import Path

print("--- APP STARTUP VERIFICATION ---")
print(f"Loaded model: {config.MODEL_PATH}")
_startup_model = YOLO(config.MODEL_PATH)
print(f"Model names: {_startup_model.names}")
print("--------------------------------\n")

# Premium Theme - Option A
theme = gr.themes.Default(
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#F6F3EE",
    block_background_fill="white",
    block_border_width="1px",
    block_border_color="#E8B86D",
    button_primary_background_fill="#C56B2A",
    button_primary_background_fill_hover="#3F2E2E",
    button_primary_text_color="white",
    body_text_color="#3F2E2E",
    block_title_text_color="#3F2E2E",
    block_label_text_color="#3F2E2E"
)

def run_image_inference(img_path):
    if img_path is None:
        return None, None, "0", "0", "0", "0.0", "0.0s"
    
    gr.Info("Running YOLOv8 Inference...")
    out_img, summary = predict_image(img_path)
    
    if "Error" in summary:
        return None, None, "0", "0", "0", "0.0", "0.0s"
        
    tot = summary['With Helmet'] + summary['Without Helmet']
    return img_path, out_img, str(tot), str(summary['With Helmet']), str(summary['Without Helmet']), summary['Average Confidence'], summary['Inference Time']

def run_video_inference(vid_path):
    if vid_path is None:
        return None, "0", "0", "0", "0.0", "0.0s"
        
    gr.Info("Processing video...")
    out_vid, summary = predict_video(vid_path)
    
    if "Error" in summary:
        return None, "0", "0", "0", "0.0", "0.0s"
        
    return out_vid, str(summary['Frames Processed']), str(summary['Helmet Detections']), str(summary['No Helmet Detections']), summary['Average Confidence'], summary['Inference Time']

def create_dashboard():
    with gr.Tab("Dashboard"):
        gr.Markdown("# Helmet Detection using YOLOv8")
        gr.Markdown("### Computer Vision | Deep Learning | Supervised Learning")
        gr.Markdown("An AI-powered helmet detection system that uses YOLOv8 to identify whether riders are wearing helmets from images and videos. The application demonstrates deep learning-based object detection with interactive inference and model evaluation.")
        
        gr.Markdown("---")
        gr.Markdown("## Evaluation Metrics")
        with gr.Row():
            for metric, val in config.EVALUATION_METRICS.items():
                with gr.Group():
                    gr.Markdown(f"**{metric}**")
                    gr.Markdown(f"# {val}")
                    
        gr.Markdown("---")
        gr.Markdown("## Project Overview & Information")
        with gr.Row():
            for key, val in config.PROJECT_INFO.items():
                with gr.Group():
                    gr.Markdown(f"**{key}**")
                    gr.Markdown(f"{val}")
                    
        gr.Markdown("---")
        gr.Markdown("## System Workflow")
        gr.Markdown("Input Media -> YOLOv8 Inference -> Detection Mapping -> Bounding Boxes -> Prediction Summary")
        
        gr.Markdown("---")
        gr.Markdown("## Sample Detection Results")
        
        cached_images = list(config.CACHE_DIR.glob("dash_*.jpg"))
        if cached_images:
            with gr.Row():
                for img_path in cached_images[:4]:
                    with gr.Column():
                        gr.Image(value=str(img_path), interactive=False, show_label=False)
                        gr.Markdown("**Validation Sample**\nPre-processed detection from the evaluation dataset.")
                        
        gr.Markdown("---")
        with gr.Accordion("About YOLOv8", open=False):
            gr.Markdown("""
            **YOLOv8** (You Only Look Once version 8) is a state-of-the-art object detection model known for its speed and accuracy. 
            It can process images and videos in real-time, making it ideal for traffic monitoring and safety applications.
            It uses a deep Convolutional Neural Network (CNN) architecture to extract spatial features from images.
            This project utilizes supervised learning, meaning the model was trained on a dataset of images where helmets and riders without helmets were explicitly labeled by humans.
            """)

def create_image_tab():
    with gr.Tab("Image Detection"):
        gr.Markdown("## Demo Gallery")
        
        demo_images = list(config.DEMO_IMAGES_DIR.glob("*.jpg"))
        titles = ["Helmet Rider", "No Helmet Rider", "Busy Road", "Multiple Riders"]
        
        with gr.Row():
            img_btns = []
            for i in range(min(4, len(demo_images))):
                with gr.Column():
                    gr.Markdown(f"**{titles[i%len(titles)]}**")
                    img_comp = gr.Image(value=str(demo_images[i]), type="filepath", interactive=False, show_label=False)
                    btn = gr.Button("Analyze Image", variant="primary")
                    img_btns.append((demo_images[i], btn))

        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Your Own Image")
                input_image = gr.Image(type="filepath", label="Upload Custom Image")
                infer_btn = gr.Button("Run YOLOv8 Detection", variant="primary")
                
                gr.Markdown("### Original Image")
                display_original = gr.Image(label="Input", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("### Annotated Image (WITH BOUNDING BOXES)")
                output_image = gr.Image(label="YOLOv8 Output", interactive=False)
                
                gr.Markdown("### Prediction Summary")
                with gr.Row():
                    tot_stat = gr.Textbox(label="Detected Objects", interactive=False)
                    with_stat = gr.Textbox(label="Helmet Count", interactive=False)
                    without_stat = gr.Textbox(label="No Helmet Count", interactive=False)
                with gr.Row():
                    conf_stat = gr.Textbox(label="Avg Confidence", interactive=False)
                    time_stat = gr.Textbox(label="Inference Time", interactive=False)

        # Wire buttons
        for img_path, btn in img_btns:
            btn.click(
                fn=lambda p=str(img_path): p, inputs=[], outputs=[input_image]
            ).then(
                fn=run_image_inference,
                inputs=[input_image],
                outputs=[display_original, output_image, tot_stat, with_stat, without_stat, conf_stat, time_stat]
            )
            
        infer_btn.click(
            fn=run_image_inference,
            inputs=[input_image],
            outputs=[display_original, output_image, tot_stat, with_stat, without_stat, conf_stat, time_stat]
        )

def create_video_tab():
    with gr.Tab("Video Detection"):
        gr.Markdown("## Demo Videos")
        
        demo_videos = list(config.DEMO_VIDEOS_DIR.glob("*.mp4"))
        
        with gr.Row():
            vid_btns = []
            for vid_path in demo_videos:
                with gr.Column():
                    gr.Markdown(f"**{vid_path.stem}**")
                    vid_comp = gr.Video(value=str(vid_path), interactive=False, show_label=False)
                    btn = gr.Button("Analyze Video", variant="primary")
                    vid_btns.append((vid_path, btn))

        gr.Markdown("---")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Upload Your Own Video")
                input_video = gr.Video(label="Upload MP4")
                infer_vid_btn = gr.Button("Process Video Analytics", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Processed Video")
                output_video = gr.Video(label="YOLOv8 Output (WITH BOUNDING BOXES)", interactive=False)
                
                gr.Markdown("### Detection Analytics")
                with gr.Row():
                    frames_stat = gr.Textbox(label="Frames Processed", interactive=False)
                    with_v_stat = gr.Textbox(label="Helmet Detections", interactive=False)
                    without_v_stat = gr.Textbox(label="No Helmet Detections", interactive=False)
                with gr.Row():
                    conf_v_stat = gr.Textbox(label="Avg Confidence", interactive=False)
                    time_v_stat = gr.Textbox(label="Processing Time", interactive=False)

        for vid_path, btn in vid_btns:
            btn.click(
                fn=lambda p=str(vid_path): p, inputs=[], outputs=[input_video]
            ).then(
                fn=run_video_inference,
                inputs=[input_video],
                outputs=[output_video, frames_stat, with_v_stat, without_v_stat, conf_v_stat, time_v_stat]
            )
            
        infer_vid_btn.click(
            fn=run_video_inference,
            inputs=[input_video],
            outputs=[output_video, frames_stat, with_v_stat, without_v_stat, conf_v_stat, time_v_stat]
        )

with gr.Blocks(theme=theme, title="Helmet Detection AI") as app:
    create_dashboard()
    create_image_tab()
    create_video_tab()
    
    gr.Markdown("---")
    gr.Markdown("<center>Helmet Detection using YOLOv8 | Developed by Hemasri Challa | GitHub | LinkedIn</center>")

if __name__ == "__main__":
    app.launch()
