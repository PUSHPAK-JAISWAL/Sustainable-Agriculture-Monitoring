from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict
import os
import uuid
import base64
import re
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
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

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Enhanced Pydantic Models
class CropIdentification(BaseModel):
    plant_name: str
    scientific_name: str = ""
    growth_stage: str
    health_status: str
    visual_symptoms: List[str]

class DiseaseAnalysis(BaseModel):
    pathogen: str
    pathogen_type: str = ""
    symptoms: List[str]
    severity: str
    lifecycle: str = ""
    risk_factors: List[str]

class SoilAnalysis(BaseModel):
    soil_type: str
    soil_ph: float
    organic_matter: str = ""
    nutrient_levels: Dict[str, str]
    recommendations: List[str]

class YieldAnalysis(BaseModel):
    current_estimate: str
    potential_loss: str
    optimization_strategies: List[str]
    economic_impact: str = ""

class FertilizationPlan(BaseModel):
    npk_ratio: str
    fertilizer_types: List[str]
    application_method: str
    schedule: str
    dosage: str = ""

class IrrigationPlan(BaseModel):
    method: str
    schedule: str
    water_requirements: str
    equipment_recommendations: List[str]

class ManagementPlan(BaseModel):
    irrigation: IrrigationPlan
    fertilization: FertilizationPlan
    cultural_practices: List[str]
    biological_controls: List[str] = []
    chemical_treatments: List[str] = []

class FullReport(BaseModel):
    crop: CropIdentification
    disease: DiseaseAnalysis
    soil: SoilAnalysis
    yield_data: YieldAnalysis
    management: ManagementPlan

class StructuredResponse(BaseModel):
    status: str
    data: Dict

# Enhanced Analysis Prompts
MANAGEMENT_PROMPT = """Create detailed management plan for {plant_name} infected with {pathogen}:
1. Irrigation:
   - Method: [drip/sprinkler/flood]
   - Schedule: [frequency + timing]
   - Water Requirements: [amount per day/week]
   - Recommended Equipment: [list]

2. Fertilization:
   - NPK Ratio: [specific ratio]
   - Recommended Fertilizers: [list with brand names]
   - Application Method: [foliar/soil injection/etc]
   - Schedule: [growth stage-based timing]
   - Dosage: [amount per acre]

3. Cultural Practices:
   - [Detailed list of cultivation practices]
   - [Sanitation measures]
   - [Prevention techniques]

4. Biological Controls: [if applicable]
5. Chemical Treatments: [specific fungicides with active ingredients]"""

SOIL_PROMPT = """Analyze soil conditions for {plant_name} cultivation:
- Soil Type: [texture + composition]
- pH Level: [exact value]
- Organic Matter: [percentage]
- Nutrient Levels: [N-P-K values]
- Recommendations: 
  1. [pH adjustment]
  2. [Organic amendments]
  3. [Mineral supplements]"""

DISEASE_PROMPT = """Analyze {plant_name} disease in {growth_stage} stage:
1. Pathogen: [scientific name + common name]
2. Pathogen Type: [fungal/bacterial/viral]
3. Symptoms: 
   - [list specific visual symptoms]
4. Severity Assessment: [% affected + stage]
5. Pathogen Lifecycle: [brief description]
6. Risk Factors:
   - [environmental conditions]
   - [cultivation practices]"""

# Enhanced Helper Functions
def extract_detailed_value(text: str, key: str) -> str:
    patterns = [
        rf'{key}[*:\s]+([^\n]+)',
        rf'\*{key}\*:\s+([^\n]+)',
        rf'{key}\s*-\s+([^\n]+)',
        rf'{key}\n([^\n]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return clean_text(match.group(1))
    return "Data not available"

def extract_detailed_list(text: str, key: str) -> List[str]:
    section = extract_section(text, key)
    items = re.findall(r'(?:\d+\.?|\*)\s+([^\n]+)', section)
    return [clean_text(item) for item in items if item.strip()]

def clean_text(text: str) -> str:
    return re.sub(r'\*+', '', text).strip()

def extract_section(text: str, key: str) -> str:
    match = re.search(rf'{key}:(.*?)(?=\n\s*\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""

# Enhanced Parsing Functions
def parse_crop_info(text: str) -> CropIdentification:
    return CropIdentification(
        plant_name=extract_detailed_value(text, "Plant Name"),
        scientific_name=extract_detailed_value(text, "Scientific Name"),
        growth_stage=extract_detailed_value(text, "Growth Stage"),
        health_status=extract_detailed_value(text, "Health Status"),
        visual_symptoms=extract_detailed_list(text, "Visual Symptoms")
    )

def parse_disease_info(text: str) -> DiseaseAnalysis:
    return DiseaseAnalysis(
        pathogen=extract_detailed_value(text, "Pathogen"),
        pathogen_type=extract_detailed_value(text, "Pathogen Type"),
        symptoms=extract_detailed_list(text, "Symptoms"),
        severity=extract_detailed_value(text, "Severity Assessment"),
        lifecycle=extract_detailed_value(text, "Pathogen Lifecycle"),
        risk_factors=extract_detailed_list(text, "Risk Factors")
    )

def parse_soil_info(text: str) -> SoilAnalysis:
    def extract_ph_value(ph_text: str) -> float:
        try:
            # Find first numeric value in the text
            match = re.search(r'(\d+\.?\d*)', ph_text)
            if match:
                ph_value = float(match.group(1))
                # Validate pH range
                if 0 <= ph_value <= 14:
                    return ph_value
            return 7.0  # Default neutral pH if parsing fails
        except (ValueError, TypeError):
            return 7.0

    ph_text = extract_detailed_value(text, "pH Level")
    return SoilAnalysis(
        soil_type=extract_detailed_value(text, "Soil Type"),
        soil_ph=extract_ph_value(ph_text),
        organic_matter=extract_detailed_value(text, "Organic Matter"),
        nutrient_levels={
            "N": extract_detailed_value(text, "Nitrogen"),
            "P": extract_detailed_value(text, "Phosphorus"),
            "K": extract_detailed_value(text, "Potassium")
        },
        recommendations=extract_detailed_list(text, "Recommendations")
    )

def parse_management_info(text: str) -> ManagementPlan:
    return ManagementPlan(
        irrigation=parse_irrigation(text),
        fertilization=parse_fertilization(text),
        cultural_practices=extract_detailed_list(text, "Cultural Practices"),
        biological_controls=extract_detailed_list(text, "Biological Controls"),
        chemical_treatments=extract_detailed_list(text, "Chemical Treatments")
    )

def parse_fertilization(text: str) -> FertilizationPlan:
    return FertilizationPlan(
        npk_ratio=extract_detailed_value(text, "NPK Ratio"),
        fertilizer_types=extract_detailed_list(text, "Recommended Fertilizers"),
        application_method=extract_detailed_value(text, "Application Method"),
        schedule=extract_detailed_value(text, "Application Schedule"),
        dosage=extract_detailed_value(text, "Recommended Dosage")
    )

def parse_irrigation(text: str) -> IrrigationPlan:
    return IrrigationPlan(
        method=extract_detailed_value(text, "Irrigation Method"),
        schedule=extract_detailed_value(text, "Irrigation Schedule"),
        water_requirements=extract_detailed_value(text, "Water Requirements"),
        equipment_recommendations=extract_detailed_list(text, "Recommended Equipment")
    )

# Analysis Pipeline
async def analyze_image(image_path: str, prompt: str) -> str:
    with open(image_path, "rb") as f:
        base64_image = base64.b64encode(f.read()).decode("utf-8")
    
    response: ChatResponse = chat(
        model='gemma3',
        messages=[{
            'role': 'user',
            'content': prompt,
            'images': [base64_image]
        }]
    )
    return response.message.content

async def analyze_crop_structure(image_path: str) -> FullReport:
    # Crop Identification
    crop_prompt = """Analyze the crop in this image. Provide:
    - Plant Name: [common name]
    - Scientific Name: [latin name]
    - Growth Stage: [specific stage]
    - Health Status: [Healthy/Unhealthy]
    - Visual Symptoms: [list of visible features]"""
    
    crop_text = await analyze_image(image_path, crop_prompt)
    crop_data = parse_crop_info(crop_text)

    # Disease Analysis
    disease_text = await analyze_image(image_path, DISEASE_PROMPT.format(
        plant_name=crop_data.plant_name,
        growth_stage=crop_data.growth_stage
    ))
    disease_data = parse_disease_info(disease_text)

    # Soil Analysis
    soil_text = await analyze_image(image_path, SOIL_PROMPT.format(
        plant_name=crop_data.plant_name
    ))
    soil_data = parse_soil_info(soil_text)

    # Management Plan
    mgmt_text = await analyze_image(image_path, MANAGEMENT_PROMPT.format(
        plant_name=crop_data.plant_name,
        pathogen=disease_data.pathogen
    ))
    mgmt_data = parse_management_info(mgmt_text)

    return FullReport(
        crop=crop_data,
        disease=disease_data,
        soil=soil_data,
        yield_data=YieldAnalysis(
            current_estimate="Estimation pending",
            potential_loss="Calculating...",
            optimization_strategies=[]
        ),
        management=mgmt_data
    )

def generate_pdf_report(image_path: str, report: FullReport, filename: str) -> str:
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    styles = getSampleStyleSheet()
    # Add custom styles
    styles.add(ParagraphStyle(
        name='Header1', 
        fontSize=16, 
        textColor=colors.darkgreen, 
        spaceAfter=12,
        fontName='Helvetica-Bold'
    ))
    # Add Header2 style here
    styles.add(ParagraphStyle(
        name='Header2', 
        fontSize=14, 
        textColor=colors.darkblue,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='TableHeader', 
        fontSize=10, 
        textColor=colors.white, 
        backColor=colors.HexColor('#2c5f2d'),
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='TableCell', 
        fontSize=9, 
        textColor=colors.black, 
        leading=12,
        fontName='Helvetica'
    ))

    # Create reusable table style
    table_style = TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5f2d')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey),
        ('WORDWRAP', (0,0), (-1,-1), 'LTR'),
        ('LEFTPADDING', (0,0), (-1,-1), 3),
        ('RIGHTPADDING', (0,0), (-1,-1), 3),
    ])

    def create_table(data, col_widths):
        formatted_data = [
            [Paragraph(cell, styles['TableCell']) for cell in row]
            for row in data
        ]
        return Table(formatted_data, colWidths=col_widths, style=table_style)

    elements = [
        Paragraph("Comprehensive Crop Analysis Report", styles['Header1']),
        Image(image_path, width=5*inch, height=3.5*inch, kind='proportional'),
        Spacer(1, 12),
        
        # Crop Identification Section
        Paragraph("Crop Identification", styles['Header2']),
        create_table([
            ["Common Name", report.crop.plant_name],
            ["Scientific Name", report.crop.scientific_name],
            ["Growth Stage", report.crop.growth_stage],
            ["Health Status", report.crop.health_status],
            ["Visual Symptoms", "\n• ".join(report.crop.visual_symptoms)]
        ], [1.5*inch, 4.5*inch]),
        Spacer(1, 10),
        
        # Disease Analysis Section
        Paragraph("Disease Analysis", styles['Header2']),
        create_table([
            ["Pathogen", f"{report.disease.pathogen} ({report.disease.pathogen_type})"],
            ["Severity", report.disease.severity],
            ["Symptoms", "\n• ".join(report.disease.symptoms)],
            ["Risk Factors", "\n• ".join(report.disease.risk_factors)]
        ], [1.5*inch, 4.5*inch]),
        Spacer(1, 10),
        
        # Management Plan Section
        Paragraph("Management Plan", styles['Header2']),
        create_table([
            ["Irrigation Method", report.management.irrigation.method],
            ["Fertilizer Types", "\n".join(report.management.fertilization.fertilizer_types)],
            ["Cultural Practices", "\n• ".join(report.management.cultural_practices)],
            ["Chemical Treatments", "\n• ".join(report.management.chemical_treatments)]
        ], [2*inch, 4*inch]),
    ]

    doc.build(elements)
    return pdf_path

@app.post("/analyze-crop", response_model=StructuredResponse)
async def analyze_crop_endpoint(
    file: UploadFile = File(...),
    request_data: Optional[str] = Form(None)
):
    try:
        # Validate and save file
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid image format")
        
        filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())

        # Generate analysis
        report = await analyze_crop_structure(file_path)
        
        # Create PDF
        pdf_filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
        pdf_path = generate_pdf_report(file_path, report, pdf_filename)
        
        # Cleanup
        os.remove(file_path)
        
        return StructuredResponse(
            status="success",
            data={
                "report": report.dict(),
                "pdf_url": f"/reports/{pdf_filename}"
            }
        )
        
    except Exception as e:
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)