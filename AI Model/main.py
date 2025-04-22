from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError
from typing import Optional
import os
import uuid
import base64
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import aiofiles

app = FastAPI()

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage config
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class AnalysisRequest(BaseModel):
    plant_name: Optional[str] = None
    growth_stage: Optional[str] = None
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = None
    region: Optional[str] = None

async def analyze_image_with_ollama(image_path: str, request_data: AnalysisRequest):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    prompt = f"""Analyze this agricultural image and provide:
    1. Plant disease detection (if any)
    2. Crop yield estimation
    3. Irrigation recommendations 
    4. Fertilization suggestions

    Automatically detect and include:
    - Crop type (if not provided: {request_data.plant_name or 'detect from image'})
    - Growth stage (if not provided: {request_data.growth_stage or 'estimate from image'})
    - Soil type suggestions (if not provided: {request_data.soil_type or 'suggest based on visual cues'})

    Structure response with clear sections using markdown formatting."""
    
    response : ChatResponse = chat(
        model='gemma3',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [base64_image]
        }]
    )
    return response.message.content

def generate_pdf_report(image_path: str, analysis: str, filename: str):
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Custom Styles
    styles.add(ParagraphStyle(
        name='MainHeader',
        fontSize=16,
        textColor='#2E7D32',  # Dark green
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='SectionHeader',
        fontSize=14,
        textColor='#43A047',  # Medium green
        spaceAfter=8,
        fontName='Helvetica-Bold'
    ))
    
    styles.add(ParagraphStyle(
        name='BulletText',
        fontSize=12,
        textColor='#424242',  # Dark gray
        leftIndent=10,
        spaceAfter=6,
        bulletIndent=0,
        bulletFontName='Helvetica',
        bulletFontSize=12
    ))

    # Title Section
    story.append(Paragraph("CROP HEALTH ANALYSIS REPORT", styles['MainHeader']))
    story.append(Spacer(1, 24))

    # Image with Border
    img = Image(image_path, width=400, height=300)
    img.hAlign = 'CENTER'
    story.append(img)
    story.append(Spacer(1, 24))

    # Process Analysis Text
    sections = analysis.split("##")
    for section in sections:
        if not section.strip():
            continue
            
        # Split section into header and content
        parts = section.split(":", 1)
        if len(parts) > 1:
            header, content = parts
            story.append(Paragraph(header.strip() + ":", styles['SectionHeader']))
            
            # Process bullet points
            for line in content.strip().split("*"):
                if line.strip():
                    story.append(Paragraph(
                        f"<bullet>&bull;</bullet> {line.strip()}",
                        styles['BulletText']
                    ))
        else:
            # Handle lines without colons
            for line in section.strip().split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
        
        story.append(Spacer(1, 12))

    # Disclaimer Section
    disclaimer = """
    <para alignment='center' spaceBefore=20>
    <font color='#757575' size=10>
    <i>This analysis is based on visual inspection of the provided image. 
    For precise recommendations, consult with agricultural experts and conduct soil tests.</i>
    </font>
    </para>
    """
    story.append(Paragraph(disclaimer, styles['Normal']))

    doc.build(story)
    return pdf_path

@app.post("/analyze-crop")
async def analyze_crop(
    file: UploadFile = File(...),
    request_data: str = Form(None)
):
    file_path = None
    try:
        # Parse request data
        if request_data:
            try:
                request_obj = AnalysisRequest.model_validate_json(request_data)
            except ValidationError as ve:
                raise HTTPException(
                    status_code=422,
                    detail={"validation_error": ve.errors()}
                )
        else:
            request_obj = AnalysisRequest()

        # Save uploaded file
        file_ext = file.filename.split(".")[-1]
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())
        
        # Perform AI analysis
        analysis_text = await analyze_image_with_ollama(file_path, request_obj)
        
        # Generate PDF report
        pdf_filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = generate_pdf_report(file_path, analysis_text, pdf_filename)
        
        # Cleanup temporary files
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "analysis": analysis_text,
            "pdf_report": pdf_path,
            "pdf_url": f"/reports/{pdf_filename}"
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Processing error: {str(e)}"
        )

# Scheduled cleanup job (implement with Celery/APScheduler if needed)