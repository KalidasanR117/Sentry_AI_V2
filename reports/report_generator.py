import os
from fpdf import FPDF

# -------------------- PDF Report Generator --------------------
def generate_pdf_report(event_buffer, summary_text, output_path):
    """
    Generates a PDF report for Mini SentryAI+ events.

    Args:
        event_buffer (list): List of dicts with keys:
                             'frame', 'yolo', 'i3d', 'final', 'screenshot'
        summary_text (str): LLM-generated summary of events
        output_path (str): Path to save the PDF
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # -------------------- Cover Page --------------------
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Mini SentryAI+ Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, summary_text)
    pdf.ln(10)

    # -------------------- Event Pages --------------------
    for event in event_buffer:
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, f"Event Frame: {event['frame']}", ln=True)
        pdf.ln(5)

        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"YOLO Detection: {event['yolo']}", ln=True)
        pdf.cell(0, 8, f"I3D Prediction: {event['i3d']}", ln=True)
        pdf.cell(0, 8, f"Final Severity: {event['final']}", ln=True)
        pdf.ln(5)

        # Add screenshot if exists
        if os.path.exists(event["screenshot"]):
            # Resize to fit page width (max 180)
            pdf.image(event["screenshot"], w=180)
        else:
            pdf.cell(0, 8, "Screenshot not found.", ln=True)

    # -------------------- Save PDF --------------------
    pdf.output(output_path)
