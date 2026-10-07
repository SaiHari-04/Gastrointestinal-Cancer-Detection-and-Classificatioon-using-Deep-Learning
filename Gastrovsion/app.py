import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors as reportlab_colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="GI Cancer Detection System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================
@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata (cached for performance)"""
    try:
        model = tf.keras.models.load_model('gi_cancer_final.keras')
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
        return model, metadata
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None, None


model, metadata = load_model_and_metadata()


if model is None or metadata is None:
    st.error("⚠️ Model files not found! Train the model first.")
    st.stop()


CLASS_NAMES = metadata['classes']
RISK_MAPPING = metadata['risk_mapping']
IMG_SIZE = metadata['img_size']


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def preprocess_image(image, img_size):
    """Preprocess uploaded image"""
    img_array = np.array(image)
    
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    img_resized = cv2.resize(img_array, (img_size, img_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch


def get_risk_color(risk_level):
    """Get color based on risk level"""
    colors = {
        "High Risk": "#dc3545",
        "Medium Risk": "#ffc107",
        "Low Risk": "#28a745"
    }
    return colors.get(risk_level, "#6c757d")


def predict_image(model, image, img_size):
    """Make prediction on image"""
    processed_img = preprocess_image(image, img_size)
    predictions = model.predict(processed_img, verbose=0)
    
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_class_idx]
    confidence = float(predictions[0][predicted_class_idx] * 100)
    risk_level = RISK_MAPPING[predicted_class]
    
    return predicted_class, confidence, risk_level, predictions[0]


def calculate_cancer_probability(predictions, class_names, risk_mapping):
    """Calculate overall cancer probability"""
    high_risk_prob = 0
    medium_risk_prob = 0
    low_risk_prob = 0
    
    for i, prob in enumerate(predictions):
        risk = risk_mapping[class_names[i]]
        if risk == "High Risk":
            high_risk_prob += prob
        elif risk == "Medium Risk":
            medium_risk_prob += prob
        else:
            low_risk_prob += prob
    
    return {
        'High Risk': float(high_risk_prob * 100),
        'Medium Risk': float(medium_risk_prob * 100),
        'Low Risk': float(low_risk_prob * 100)
    }


def generate_pdf_report(predicted_class, confidence, risk_level, cancer_prob, prob_df, metadata):
    """Generate professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=reportlab_colors.HexColor('#1f6feb'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        textColor=reportlab_colors.HexColor('#666666'),
        spaceAfter=20,
        alignment=TA_CENTER,
        fontName='Helvetica'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=reportlab_colors.HexColor('#1f6feb'),
        spaceAfter=12,
        spaceBefore=12,
        fontName='Helvetica-Bold',
        borderWidth=1,
        borderColor=reportlab_colors.HexColor('#1f6feb'),
        borderPadding=5,
        backColor=reportlab_colors.HexColor('#f8f9fa')
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        textColor=reportlab_colors.black,
        spaceAfter=8,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Header
    elements.append(Paragraph("🏥 GASTROINTESTINAL CANCER DETECTION", title_style))
    elements.append(Paragraph("AI-Powered Medical Image Analysis Report", subtitle_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    report_info_data = [
        ['Report Generated:', datetime.now().strftime('%B %d, %Y at %H:%M:%S')],
        ['Report ID:', f"GI-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
        ['Analysis System:', 'AI Deep Learning Model'],
        ['Model Architecture:', metadata.get('model', 'MobileNetV2')],
        ['Model Accuracy:', f"{metadata.get('mean_accuracy', 0)*100:.2f}% ± {metadata.get('std_accuracy', 0)*100:.2f}%"]
    ]
    
    report_info_table = Table(report_info_data, colWidths=[2.5*inch, 4*inch])
    report_info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), reportlab_colors.HexColor('#e9ecef')),
        ('TEXTCOLOR', (0, 0), (-1, -1), reportlab_colors.black),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(report_info_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Primary Diagnosis Section
    elements.append(Paragraph("PRIMARY DIAGNOSIS", heading_style))
    
    # Get risk color for diagnosis
    risk_color_hex = get_risk_color(risk_level)
    
    diagnosis_data = [
        ['Detected Condition:', predicted_class],
        ['Confidence Score:', f"{confidence:.2f}%"],
        ['Risk Classification:', risk_level],
    ]
    
    diagnosis_table = Table(diagnosis_data, colWidths=[2.5*inch, 4*inch])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), reportlab_colors.HexColor('#e9ecef')),
        ('BACKGROUND', (1, 2), (1, 2), reportlab_colors.HexColor(risk_color_hex)),
        ('TEXTCOLOR', (0, 0), (-1, -1), reportlab_colors.black),
        ('TEXTCOLOR', (1, 2), (1, 2), reportlab_colors.white),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, 1), 'Helvetica'),
        ('FONTNAME', (1, 2), (1, 2), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(diagnosis_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # Cancer Risk Analysis
    elements.append(Paragraph("CANCER RISK ANALYSIS", heading_style))
    
    risk_data = [
        ['Risk Category', 'Probability', 'Assessment'],
        ['High Risk', f"{cancer_prob['High Risk']:.2f}%", '■■■■■■■■■■' if cancer_prob['High Risk'] > 50 else '■■■■■' if cancer_prob['High Risk'] > 25 else '■■'],
        ['Medium Risk', f"{cancer_prob['Medium Risk']:.2f}%", '■■■■■■■■■■' if cancer_prob['Medium Risk'] > 50 else '■■■■■' if cancer_prob['Medium Risk'] > 25 else '■■'],
        ['Low Risk', f"{cancer_prob['Low Risk']:.2f}%", '■■■■■■■■■■' if cancer_prob['Low Risk'] > 50 else '■■■■■' if cancer_prob['Low Risk'] > 25 else '■■'],
    ]
    
    risk_table = Table(risk_data, colWidths=[2*inch, 1.5*inch, 3*inch])
    risk_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BACKGROUND', (0, 1), (-1, 1), reportlab_colors.HexColor('#ffe0e0')),
        ('BACKGROUND', (0, 2), (-1, 2), reportlab_colors.HexColor('#fff4cc')),
        ('BACKGROUND', (0, 3), (-1, 3), reportlab_colors.HexColor('#d4edda')),
        ('TEXTCOLOR', (0, 1), (-1, -1), reportlab_colors.black),
        ('FONTNAME', (0, 1), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, reportlab_colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))
    elements.append(risk_table)
    elements.append(Spacer(1, 0.25*inch))
    
    # All Conditions Probability
    elements.append(Paragraph("DETAILED PROBABILITY DISTRIBUTION", heading_style))
    
    # Prepare probability table data
    prob_table_data = [['Rank', 'Condition', 'Probability', 'Risk Level']]
    for idx, row in prob_df.iterrows():
        prob_table_data.append([
            str(idx + 1),
            row['Condition'],
            f"{row['Probability (%)']:.2f}%",
            row['Risk Level']
        ])
    
    prob_table = Table(prob_table_data, colWidths=[0.6*inch, 2.8*inch, 1.3*inch, 1.8*inch])
    
    # Dynamic styling based on risk
    table_style = [
        ('BACKGROUND', (0, 0), (-1, 0), reportlab_colors.HexColor('#1f6feb')),
        ('TEXTCOLOR', (0, 0), (-1, 0), reportlab_colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('TEXTCOLOR', (0, 1), (-1, -1), reportlab_colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('GRID', (0, 0), (-1, -1), 0.5, reportlab_colors.grey),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]
    
    # Add row colors based on risk
    for idx, row in enumerate(prob_df.iterrows(), start=1):
        _, row_data = row
        if row_data['Risk Level'] == 'High Risk':
            table_style.append(('BACKGROUND', (3, idx), (3, idx), reportlab_colors.HexColor('#ffe0e0')))
        elif row_data['Risk Level'] == 'Medium Risk':
            table_style.append(('BACKGROUND', (3, idx), (3, idx), reportlab_colors.HexColor('#fff4cc')))
        else:
            table_style.append(('BACKGROUND', (3, idx), (3, idx), reportlab_colors.HexColor('#d4edda')))
    
    prob_table.setStyle(TableStyle(table_style))
    elements.append(prob_table)
    elements.append(Spacer(1, 0.3*inch))
    
    # Clinical Recommendations
    elements.append(Paragraph("CLINICAL RECOMMENDATIONS", heading_style))
    
    if risk_level == "High Risk":
        recommendations = """
        <b>⚠️ URGENT MEDICAL ATTENTION REQUIRED</b><br/><br/>
        • Schedule <b>immediate consultation</b> with a board-certified gastroenterologist<br/>
        • Biopsy and comprehensive histopathological examination recommended<br/>
        • Consider additional imaging studies (CT scan, MRI, or PET scan)<br/>
        • Discuss treatment options including potential surgical intervention<br/>
        • Follow institutional cancer screening and management protocols<br/>
        • Arrange for multidisciplinary team consultation<br/>
        • Patient should be informed of findings within 24-48 hours
        """
    elif risk_level == "Medium Risk":
        recommendations = """
        <b>⚠️ MEDICAL CONSULTATION RECOMMENDED</b><br/><br/>
        • Schedule appointment with gastroenterologist within 1-2 weeks<br/>
        • Monitor symptoms closely (abdominal pain, bleeding, unexplained weight loss)<br/>
        • Follow-up endoscopy may be required in 3-6 months<br/>
        • Maintain detailed symptom diary and document any changes<br/>
        • Consider lifestyle modifications (diet optimization, smoking cessation, alcohol reduction)<br/>
        • Discuss family history and genetic risk factors with physician<br/>
        • Regular monitoring and surveillance recommended
        """
    else:
        recommendations = """
        <b>✓ NORMAL FINDINGS - ROUTINE FOLLOW-UP</b><br/><br/>
        • Continue regular screening schedule as per clinical guidelines<br/>
        • Maintain healthy lifestyle with balanced diet rich in fiber<br/>
        • Report any new or concerning symptoms to healthcare provider promptly<br/>
        • Follow standard surveillance protocols for your age group<br/>
        • Next routine endoscopy as recommended by your physician<br/>
        • Continue preventive health measures and regular check-ups<br/>
        • No immediate intervention required
        """
    
    elements.append(Paragraph(recommendations, body_style))
    elements.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    elements.append(Paragraph("MEDICAL DISCLAIMER AND LIMITATIONS", heading_style))
    disclaimer_text = """
    <b>IMPORTANT NOTICE:</b> This report is generated by an artificial intelligence system for 
    <b>educational and research purposes only</b>. This analysis does NOT constitute medical advice, 
    diagnosis, or treatment recommendation. The findings presented in this report must be:<br/><br/>
    
    • <b>Verified by licensed medical professionals</b> including board-certified gastroenterologists<br/>
    • <b>Interpreted in clinical context</b> considering patient history, symptoms, and other diagnostic findings<br/>
    • <b>Confirmed through additional testing</b> including histopathology when indicated<br/>
    • <b>Reviewed as part of comprehensive care</b> by qualified healthcare providers<br/><br/>
    
    <b>Limitations:</b> AI predictions may have false positives/negatives. Image quality, acquisition 
    parameters, and patient-specific factors can affect accuracy. This system has not been approved 
    by regulatory authorities (FDA, CE, etc.) for clinical diagnostic use.<br/><br/>
    
    <b>Clinical Decision-Making:</b> All medical decisions must be made by qualified healthcare 
    professionals based on complete clinical assessment, not solely on AI predictions.<br/><br/>
    
    <b>Data Privacy:</b> This report contains sensitive health information and should be handled 
    according to applicable privacy regulations (HIPAA, GDPR, etc.).
    """
    
    elements.append(Paragraph(disclaimer_text, body_style))
    elements.append(Spacer(1, 0.2*inch))
    
    # Footer
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=reportlab_colors.grey,
        alignment=TA_CENTER
    )
    
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph("_" * 100, footer_style))
    elements.append(Paragraph(
        f"<i>Report generated by AI GI Cancer Detection System v2.0 | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page 1 of 1</i>",
        footer_style
    ))
    elements.append(Paragraph(
        "<i>For research and educational purposes only • Not for clinical use • Requires professional medical interpretation</i>",
        footer_style
    ))
    
    # Build PDF
    doc.build(elements)
    buffer.seek(0)
    return buffer


# ============================================================================
# CUSTOM CSS - PROFESSIONAL DARK THEME
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0d1117;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    /* Header */
    .main-header {
        background: #161b22;
        color: #ffffff;
        padding: 2.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 5px solid #1f6feb;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        color: #8b949e;
        font-weight: 400;
    }
    
    /* Cards */
    .metric-card {
        background: #161b22;
        padding: 1.8rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        margin: 1rem 0;
        border: 1px solid #30363d;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        box-shadow: 0 4px 12px rgba(31,111,235,0.3);
        transform: translateY(-2px);
        border-color: #1f6feb;
    }
    
    .metric-card h3 {
        color: #58a6ff;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    
    .metric-card p {
        color: #8b949e;
        margin: 0;
    }
    
    /* Risk box */
    .risk-box {
        background: #161b22;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 2px solid;
    }
    
    /* Prediction box */
    .prediction-box {
        background: #161b22;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-left: 5px solid #1f6feb;
    }
    
    .prediction-title {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        color: #ffffff;
    }
    
    .prediction-confidence {
        font-size: 1.1rem;
        margin-top: 0.8rem;
        color: #8b949e;
        font-weight: 500;
    }
    
    /* Alert boxes */
    .alert-box {
        background: #161b22;
        padding: 1.8rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    .alert-box h3 {
        margin-top: 0;
        margin-bottom: 1rem;
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .alert-box ul {
        margin: 0;
        padding-left: 1.5rem;
        color: #c9d1d9;
    }
    
    .alert-box li {
        margin: 0.5rem 0;
        line-height: 1.6;
        color: #c9d1d9;
    }
    
    .alert-box strong {
        color: #ffffff;
    }
    
    .alert-high {
        border-color: #f85149;
        background: #1c1416;
    }
    
    .alert-high h3 {
        color: #ff7b72;
    }
    
    .alert-medium {
        border-color: #d29922;
        background: #1c1810;
    }
    
    .alert-medium h3 {
        color: #ffa657;
    }
    
    .alert-low {
        border-color: #3fb950;
        background: #0d1a14;
    }
    
    .alert-low h3 {
        color: #56d364;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #c9d1d9 !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Detectable conditions - FIX FOR TEXT COLOR */
    .condition-card {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: #0d1117;
        border-radius: 8px;
        border-left: 4px solid;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    .condition-card .condition-name {
        color: #ffffff !important;
        font-weight: 600;
        font-size: 0.95rem;
        display: block;
        margin-bottom: 0.3rem;
    }
    
    .condition-card .condition-risk {
        font-size: 0.85rem;
        font-weight: 600;
        display: block;
    }
    
    /* Info boxes */
    .stInfo {
        background-color: #0d1821;
        border-left: 5px solid #1f6feb;
        color: #c9d1d9 !important;
    }
    
    .stInfo * {
        color: #c9d1d9 !important;
    }
    
    .stWarning {
        background-color: #1c1810;
        border-left: 5px solid #d29922;
        color: #c9d1d9 !important;
    }
    
    .stWarning * {
        color: #c9d1d9 !important;
    }
    
    /* Button */
    .stButton > button {
        width: 100%;
        background: #1f6feb;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: #1158c7;
        box-shadow: 0 4px 12px rgba(31, 111, 235, 0.5);
        transform: translateY(-1px);
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    h2 {
        font-weight: 700;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f6feb;
        margin-bottom: 1.5rem;
    }
    
    h3 {
        font-weight: 600;
    }
    
    /* General text */
    p, span, div, li {
        color: #c9d1d9;
    }
    
    /* Caption text */
    .stCaption {
        color: #8b949e !important;
    }
    
    /* Stats boxes */
    .stat-box {
        background: #161b22;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border-top: 4px solid;
    }
    
    .stat-box h2 {
        margin: 0;
        font-size: 2.5rem;
        border: none;
    }
    
    .stat-box p {
        margin: 0.5rem 0 0 0;
        color: #8b949e;
        font-size: 0.95rem;
    }
    
    /* Image container */
    .image-container {
        background: #161b22;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        border: 1px solid #30363d;
    }
    
    /* Download button */
    .stDownloadButton > button {
        width: 100%;
        background: #3fb950;
        color: white;
        font-size: 1rem;
        font-weight: 600;
        padding: 0.75rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 2px 8px rgba(63, 185, 80, 0.4);
    }
    
    .stDownloadButton > button:hover {
        background: #2ea043;
        box-shadow: 0 4px 12px rgba(63, 185, 80, 0.5);
    }
    
    /* Streamlit elements */
    .stMarkdown {
        color: #c9d1d9;
    }
    
    /* File uploader */
    .stFileUploader {
        background: #161b22;
        border: 2px dashed #30363d;
        border-radius: 10px;
    }
    
    .stFileUploader label {
        color: #c9d1d9 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1f6feb;
    }
    
    /* Dataframe */
    .stDataFrame {
        background: #161b22;
        border: 1px solid #30363d;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #1f6feb !important;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <div class="main-title">🏥 AI Gastrointestinal Cancer Detection System</div>
    <div class="main-subtitle">Advanced Deep Learning for Medical Image Analysis</div>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/medical-doctor.png", width=80)
    st.title("📊 System Information")
    
    st.markdown("---")
    
    st.markdown("### 🤖 Model Details")
    st.info(f"""
    **Architecture:** {metadata.get('model', 'MobileNetV2')}  
    **Accuracy:** {metadata.get('mean_accuracy', 0)*100:.1f}% ± {metadata.get('std_accuracy', 0)*100:.1f}%  
    **Classes:** {len(CLASS_NAMES)}  
    **Input Size:** {IMG_SIZE}x{IMG_SIZE}  
    **Training Images:** {metadata.get('total_images', 'N/A')}
    """)
    
    st.markdown("---")
    
    st.markdown("### 📋 Detectable Conditions")
    for idx, class_name in enumerate(CLASS_NAMES, 1):
        risk = RISK_MAPPING[class_name]
        color = get_risk_color(risk)
        st.markdown(f"""
        <div class="condition-card" style="border-left-color: {color};">
            <span class="condition-name">{idx}. {class_name}</span>
            <span class="condition-risk" style="color: {color};">● {risk}</span>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning("""
    This AI system is for **educational and research purposes only**.
    
    ⚕️ Always consult qualified medical professionals for diagnosis and treatment.
    
    🔬 Results must be verified by licensed gastroenterologists.
    """)
    
    st.markdown("---")
    st.caption(f"🕐 Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ============================================================================
# MAIN CONTENT
# ============================================================================


# Upload Section
st.markdown("## 📤 Upload Medical Image")


uploaded_file = st.file_uploader(
    "Upload gastrointestinal endoscopy image for analysis",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Two column layout
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("### 📸 Original Image")
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, use_container_width=True, caption="Uploaded endoscopy image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image info
        st.markdown(f"""
        <div class="metric-card">
            <strong style="color: #58a6ff; font-size: 1.1rem;">📏 Image Properties</strong><br><br>
            <div style="color: #c9d1d9;">
                <strong>Size:</strong> {image.size[0]} x {image.size[1]} pixels<br>
                <strong>Format:</strong> {image.format}<br>
                <strong>Mode:</strong> {image.mode}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🔬 AI Analysis")
        
        # Analyze button
        analyze_clicked = st.button("🚀 ANALYZE IMAGE", type="primary")
        
        if analyze_clicked:
            
            with st.spinner("🔄 Processing image with AI model..."):
                # Prediction
                predicted_class, confidence, risk_level, all_predictions = predict_image(
                    model, image, IMG_SIZE
                )
                
                # Cancer probability
                cancer_prob = calculate_cancer_probability(all_predictions, CLASS_NAMES, RISK_MAPPING)
                
                # Store in session state
                st.session_state['analyzed'] = True
                st.session_state['predicted_class'] = predicted_class
                st.session_state['confidence'] = confidence
                st.session_state['risk_level'] = risk_level
                st.session_state['all_predictions'] = all_predictions
                st.session_state['cancer_prob'] = cancer_prob
            
            st.success("✅ Analysis completed successfully!")
    
    # Show results if analyzed
    if st.session_state.get('analyzed', False):
        predicted_class = st.session_state['predicted_class']
        confidence = st.session_state['confidence']
        risk_level = st.session_state['risk_level']
        all_predictions = st.session_state['all_predictions']
        cancer_prob = st.session_state['cancer_prob']
        
        with col2:
            # Predicted condition
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-title">🎯 {predicted_class}</div>
                <div class="prediction-confidence">Confidence Score: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Risk assessment
            risk_color = get_risk_color(risk_level)
            st.markdown(f"""
            <div class="risk-box" style="border-color: {risk_color}; color: {risk_color};">
                ⚠️ {risk_level}
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence progress bar
            st.markdown("**📊 Confidence Score**")
            st.progress(float(confidence / 100))
        
        st.markdown("---")
        
        # Cancer Risk Analysis
        st.markdown("## 🧬 Cancer Risk Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #dc3545;">
                <h2 style="color: #dc3545;">{cancer_prob['High Risk']:.1f}%</h2>
                <p>High Risk Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #ffc107;">
                <h2 style="color: #ff8800;">{cancer_prob['Medium Risk']:.1f}%</h2>
                <p>Medium Risk Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="stat-box" style="border-color: #28a745;">
                <h2 style="color: #28a745;">{cancer_prob['Low Risk']:.1f}%</h2>
                <p>Low Risk Probability</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk distribution visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart
            fig_pie = go.Figure(data=[go.Pie(
                labels=list(cancer_prob.keys()),
                values=list(cancer_prob.values()),
                hole=0.4,
                marker=dict(colors=['#dc3545', '#ffc107', '#28a745']),
                textinfo='label+percent',
                textfont=dict(size=14, color='white', family='Arial')
            )])
            fig_pie.update_layout(
                title=dict(
                    text="Risk Distribution",
                    font=dict(size=16, color='#ffffff')
                ),
                height=400,
                showlegend=True,
                font=dict(size=13, color='#c9d1d9'),
                paper_bgcolor='#161b22',
                plot_bgcolor='#161b22',
                legend=dict(
                    font=dict(size=12, color='#c9d1d9'),
                    bgcolor='#161b22'
                )
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={'font': {'size': 40, 'color': '#ffffff'}},
                title={'text': "Prediction Confidence", 'font': {'size': 18, 'color': '#ffffff'}},
                gauge={
                    'axis': {
                        'range': [None, 100],
                        'tickfont': {'size': 12, 'color': '#c9d1d9'}
                    },
                    'bar': {'color': risk_color},
                    'steps': [
                        {'range': [0, 50], 'color': "#2d1315"},
                        {'range': [50, 75], 'color': "#2d2210"},
                        {'range': [75, 100], 'color': "#0d2818"}
                    ],
                    'threshold': {
                        'line': {'color': "#ff7b72", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor='#161b22',
                font=dict(size=13, color='#c9d1d9')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown("---")
        
        # Detailed predictions
        st.markdown("## 📊 Detailed Class Probabilities")
        
        # Create probability dataframe
        import pandas as pd
        prob_data = {
            'Condition': CLASS_NAMES,
            'Probability (%)': [float(p * 100) for p in all_predictions],
            'Risk Level': [RISK_MAPPING[cls] for cls in CLASS_NAMES]
        }
        prob_df = pd.DataFrame(prob_data).sort_values('Probability (%)', ascending=False).reset_index(drop=True)
        
        # Bar chart
        fig_bar = px.bar(
            prob_df,
            x='Condition',
            y='Probability (%)',
            color='Risk Level',
            color_discrete_map={
                'High Risk': '#dc3545',
                'Medium Risk': '#ffc107',
                'Low Risk': '#28a745'
            },
            title="Probability Distribution Across All Conditions",
            height=500
        )
        fig_bar.update_layout(
            xaxis_tickangle=-45,
            font=dict(size=13, color='#c9d1d9'),
            showlegend=True,
            paper_bgcolor='#161b22',
            plot_bgcolor='#161b22',
            xaxis=dict(
                gridcolor='#30363d',
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#c9d1d9')
            ),
            yaxis=dict(
                gridcolor='#30363d',
                title_font=dict(size=14, color='#ffffff'),
                tickfont=dict(size=12, color='#c9d1d9')
            ),
            title_font=dict(size=16, color='#ffffff'),
            legend=dict(
                font=dict(size=12, color='#c9d1d9'),
                bgcolor='#161b22',
                bordercolor='#30363d',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Table
        st.markdown("### 📋 Detailed Probability Table")
        st.dataframe(
            prob_df.style.background_gradient(subset=['Probability (%)'], cmap='RdYlGn_r')
            .format({'Probability (%)': '{:.2f}%'}),
            use_container_width=True,
            height=350
        )
        
        st.markdown("---")
        
        # Medical recommendations
        st.markdown("## 💡 Clinical Recommendations")
        
        if risk_level == "High Risk":
            st.markdown("""
            <div class="alert-box alert-high">
                <h3>🚨 Urgent Medical Attention Required</h3>
                <ul>
                    <li>Schedule <strong>immediate consultation</strong> with a board-certified gastroenterologist</li>
                    <li>Biopsy and comprehensive histopathological examination recommended</li>
                    <li>Consider additional imaging studies (CT scan, MRI, or PET scan)</li>
                    <li>Discuss treatment options including potential surgical intervention</li>
                    <li>Follow institutional cancer screening and management protocols</li>
                    <li>Arrange for multidisciplinary team consultation</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        elif risk_level == "Medium Risk":
            st.markdown("""
            <div class="alert-box alert-medium">
                <h3>⚠️ Medical Consultation Recommended</h3>
                <ul>
                    <li>Schedule appointment with gastroenterologist within 1-2 weeks</li>
                    <li>Monitor symptoms closely (abdominal pain, bleeding, unexplained weight loss)</li>
                    <li>Follow-up endoscopy may be required in 3-6 months</li>
                    <li>Maintain detailed symptom diary and document any changes</li>
                    <li>Consider lifestyle modifications (diet optimization, smoking cessation, alcohol reduction)</li>
                    <li>Discuss family history and genetic risk factors with physician</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="alert-box alert-low">
                <h3>✅ Normal Findings - Routine Follow-up</h3>
                <ul>
                    <li>Continue regular screening schedule as per clinical guidelines</li>
                    <li>Maintain healthy lifestyle with balanced diet rich in fiber</li>
                    <li>Report any new or concerning symptoms to healthcare provider promptly</li>
                    <li>Follow standard surveillance protocols for your age group</li>
                    <li>Next routine endoscopy as recommended by your physician</li>
                    <li>Continue preventive health measures and regular check-ups</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Download report
        st.markdown("---")
        st.markdown("## 📥 Export Analysis Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Text report
            report_data = {
                'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Predicted Condition': predicted_class,
                'Confidence Score': f"{confidence:.2f}%",
                'Risk Level': risk_level,
                'High Risk Probability': f"{cancer_prob['High Risk']:.2f}%",
                'Medium Risk Probability': f"{cancer_prob['Medium Risk']:.2f}%",
                'Low Risk Probability': f"{cancer_prob['Low Risk']:.2f}%",
                'Model': metadata.get('model', 'MobileNetV2'),
                'Model Accuracy': f"{metadata.get('mean_accuracy', 0)*100:.1f}%"
            }
            
            report_text = "=" * 70 + "\n"
            report_text += "   GASTROINTESTINAL CANCER DETECTION - AI ANALYSIS REPORT\n"
            report_text += "=" * 70 + "\n\n"
            report_text += f"Generated: {report_data['Timestamp']}\n\n"
            report_text += "-" * 70 + "\n"
            report_text += "ANALYSIS RESULTS:\n"
            report_text += "-" * 70 + "\n\n"
            for key, value in report_data.items():
                if key != 'Timestamp':
                    report_text += f"{key:.<40} {value}\n"
            report_text += "\n" + "-" * 70 + "\n"
            report_text += "ALL CLASS PROBABILITIES:\n"
            report_text += "-" * 70 + "\n\n"
            for idx, row in prob_df.iterrows():
                report_text += f"{row['Condition']:.<45} {row['Probability (%)']:>6.2f}% ({row['Risk Level']})\n"
            report_text += "\n" + "=" * 70 + "\n"
            report_text += "MEDICAL DISCLAIMER:\n"
            report_text += "=" * 70 + "\n"
            report_text += "This is an AI-generated report for research and educational purposes only.\n"
            report_text += "This analysis does NOT constitute medical advice or diagnosis.\n"
            report_text += "Always consult qualified healthcare professionals for medical decisions.\n"
            report_text += "Results must be verified by licensed medical practitioners.\n"
            report_text += "=" * 70 + "\n"
            
            st.download_button(
                label="📄 Download Text Report (TXT)",
                data=report_text,
                file_name=f"GI_Cancer_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col2:
            # PDF report
            pdf_buffer = generate_pdf_report(predicted_class, confidence, risk_level, cancer_prob, prob_df, metadata)
            
            st.download_button(
                label="📕 Download Professional PDF Report",
                data=pdf_buffer,
                file_name=f"GI_Cancer_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )


else:
    # No image uploaded - show instructions
    st.info("👆 Please upload a gastrointestinal endoscopy image to begin AI-powered analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤 Step 1</h3>
            <p style="color: #8b949e;">Upload your high-quality endoscopy image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔬 Step 2</h3>
            <p style="color: #8b949e;">Click analyze to process with AI</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Step 3</h3>
            <p style="color: #8b949e;">Review comprehensive AI predictions</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #161b22; border-radius: 10px; margin-top: 2rem; box-shadow: 0 2px 8px rgba(0,0,0,0.3); border: 1px solid #30363d;">
    <h4 style="color: #58a6ff; margin: 0; font-weight: 700;">🏥 AI Gastrointestinal Cancer Detection System</h4>
    <p style="color: #8b949e; margin: 0.8rem 0; font-size: 0.95rem;">Powered by TensorFlow • Deep Learning • Streamlit</p>
    <p style="font-size: 0.85rem; color: #6e7681; margin: 0.5rem 0;">
        ⚠️ For educational and research purposes only | Not FDA approved | Not for clinical use
    </p>
</div>
""", unsafe_allow_html=True)
