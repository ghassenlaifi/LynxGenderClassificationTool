import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import logging
from fpdf import FPDF
from datetime import datetime

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ====================== MODEL CLASS ======================
class LynxClassifier(nn.Module):
    def __init__(self):
        super(LynxClassifier, self).__init__()
        self.model = torch.hub.load("pytorch/vision", "resnet18", pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)  # Deux classes : Male et Female
        )

    def forward(self, x):
        return self.model(x)


# Charger le modèle
@st.cache_resource()
def load_model():
    model = LynxClassifier()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# Prédiction
def predict_image(image, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities

# Génération du PDF
def generate_pdf(image_path, predicted_class, confidence, date, feedback, uploaded_image_name):
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", size=16)
    pdf.cell(200, 10, txt="Lynx Gender Classification Report", ln=True, align='C')

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction Date: {date}", ln=True, align='C')

    pdf.ln(10)
    pdf.image(image_path, x=60, w=90)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Predicted Gender: {predicted_class}", ln=True, align='C')
    pdf.cell(200, 10, txt=f"Confidence: {confidence:.2f}%", ln=True, align='C')

    morphological_features = [
        "Wider chest and shoulders",
        "Broader face with a more pronounced muzzle",
        "Larger size and overall body structure",
        "Thicker neck and thicker fur around the head"
    ] if predicted_class == "Male" else [
        "Narrower chest and shoulders",
        "More delicate facial features with a softer muzzle",
        "Smaller size with a more slender body",
        "Finer neck and less fur around the head"
    ]

    pdf.ln(10)
    pdf.cell(200, 10, txt="Morphological Features:", ln=True, align='L')
    for feature in morphological_features:
        pdf.cell(200, 10, txt=f"- {feature}", ln=True, align='L')

    if feedback:
        pdf.ln(10)
        pdf.cell(200, 10, txt=f"Additional Comments: {feedback}", ln=True, align='L')

    pdf_path = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uploaded_image_name}.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Streamlit UI
st.set_page_config(
    page_title="Lynx Gender Classifier",
    page_icon="🐱",
    layout="wide"
)

# Custom CSS for design
st.markdown("""<style>
    .center-div { display: flex; justify-content: center; align-items: center; flex-direction: column; }
    .prediction-box { background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin-top: 1.5rem; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    .prediction-box h3 { color: black; font-size: 20px; font-weight: bold; text-align: center; }
    .prediction-box h4 { color: black; font-size: 18px; text-align: center; }
    footer { text-align: center; padding: 1rem; background-color: #f0f2f6; font-family: 'Arial', sans-serif; font-size: 14px; color: #333; }
</style>""", unsafe_allow_html=True)

# Header section
st.title("🐱 Lynx Gender Classification")
st.markdown("""<div style='text-align: center'>
Upload an image of a lynx to classify its gender and receive a detailed analysis report.
</div>""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["Classification", "About"])

with tab1:
    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a lynx image",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear, well-lit image of a lynx for best results"
        )
        
        
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                resized_image = image.resize((800, int(image.size[1] * 800 / image.size[0])), Image.LANCZOS)
                resized_image.save("uploaded_image.jpg")

                st.image(resized_image, caption="Uploaded Image", use_container_width=True)

                model = load_model()
                probabilities = predict_image(image, model)
                classes = ["Male", "Female"]
                predicted_class = classes[torch.argmax(probabilities).item()]
                confidence = probabilities.max().item() * 100

                st.markdown(f"""
                    <div class="prediction-box">
                        <h3>Prediction: {predicted_class}</h3>
                        <h4>Confidence: {confidence:.1f}%</h4>
                    </div>
                """, unsafe_allow_html=True)

                pdf_path = generate_pdf("uploaded_image.jpg", predicted_class, confidence,
                                       datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "", uploaded_file.name)
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📄 Download Detailed Report",
                        data=pdf_file,
                        file_name=f"lynx_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf"
                    )
                
                # Feedback Section
                st.subheader("Your Feedback")

                feedback_rating = st.slider(
                    "Rate the prediction accuracy (1 = Poor, 5 = Excellent)",
                    min_value=1, max_value=5, value=3
                )

                feedback_radio = st.radio(
                    "Was the prediction correct?",
                    options=["Correct", "Incorrect"],
                    index=0,
                    key="feedback_radio",
                    label_visibility="collapsed",
                    horizontal=True
                )

                feedback_comment = st.text_area("Additional Comments", help="Provide any suggestions or comments about the prediction")

                if st.button("Submit Feedback"):
                    st.success("Thank you for your feedback!")
                    feedback_data = {
                        'rating': feedback_rating,
                        'prediction_correct': feedback_radio,
                        'comment': feedback_comment,
                        'image': uploaded_file.name,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    st.write(feedback_data)        
                
            except Exception as e:
                st.error(f"⚠️ Error processing image: {e}")


    with col3:
        st.write("")  # Empty space in the third column


with tab2:
    st.markdown(""" 
               ### About This Tool
        This application uses deep learning to classify the gender of lynx based on their morphological features.
        
        #### How it Works
        1. Upload a clear image of a lynx
        2. Our AI model analyzes the image using various visual features
        3. Get instant results and a detailed PDF report
        
        #### Best Practices for Images
        - Use clear, well-lit photographs
        - Ensure the lynx is the main subject
        - Side-view or front-view images work best
        - Avoid heavily edited or filtered images
        
        #### Model Information
        The classifier uses a custom convolutional neural network trained on a dataset of lynx images. It analyzes various morphological features to determine gender.
        """)


# Footer section styled like the prediction box

st.markdown("""---""", unsafe_allow_html=True)
st.markdown("""<style>
    .footer-box {
        background-color: #f0f2f6; /* Same background as the prediction box */
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1); /* Subtle border for better visibility */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Consistent shadow with prediction box */
        text-align: center;
        font-family: 'Georgia', serif;
        font-size: 14px;
        color: black; /* Text color for readability */
        margin-top: 1.5rem;
    }
    /* Adapts to dark mode */
    .st-dark .footer-box {
        background-color: #31333F; /* Match dark mode prediction box */
        color: white; /* Ensure text is readable in dark mode */
        border: 1px solid rgba(255, 255, 255, 0.1); /* Subtle border for dark mode */
    }
</style>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-box">
    Developed for research purposes. Please use responsibly.
</div>
""", unsafe_allow_html=True)