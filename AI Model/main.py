from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError
from typing import Optional, List, Dict
import os
import uuid
import base64
import re
import asyncio
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

# Enhanced Pydantic Models with user input
class UserInput(BaseModel):
    plant_name: Optional[str] = None
    growth_stage: Optional[str] = None
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = None
    region: Optional[str] = None

class CropIdentification(BaseModel):
    plant_name: str
    scientific_name: str
    growth_stage: str
    health_status: str
    visual_symptoms: List[str]

class DiseaseAnalysis(BaseModel):
    pathogen: str
    pathogen_type: str
    symptoms: List[str]
    severity: str
    lifecycle: str
    risk_factors: List[str]

class SoilAnalysis(BaseModel):
    soil_type: str
    soil_ph: float
    organic_matter: str
    nutrient_levels: Dict[str, str]
    recommendations: List[str]

class YieldAnalysis(BaseModel):
    current_estimate: str
    potential_loss: str
    optimization_strategies: List[str]
    economic_impact: str

class FertilizationPlan(BaseModel):
    npk_ratio: str
    fertilizer_types: List[str]
    application_method: str
    schedule: str
    dosage: str

class IrrigationPlan(BaseModel):
    method: str
    schedule: str
    water_requirements: str
    equipment_recommendations: List[str]

class ManagementPlan(BaseModel):
    irrigation: IrrigationPlan
    fertilization: FertilizationPlan
    cultural_practices: List[str]
    biological_controls: List[str]
    chemical_treatments: List[str]

class FullReport(BaseModel):
    crop: CropIdentification
    disease: DiseaseAnalysis
    soil: SoilAnalysis
    yield_data: YieldAnalysis
    management: ManagementPlan

class StructuredResponse(BaseModel):
    status: str
    data: Dict

# Enhanced Analysis Prompts with user input integration
BASE_PROMPT = """Analyze this agricultural image and provide:
{user_input_section}
- Plant Name: {plant_name}
- Growth Stage: {growth_stage}
- Visible Soil Characteristics: {soil_type}
- Observed Health Indicators: {health_status}
- Key Visual Symptoms: {visual_symptoms}

Provide concise responses using bullet points and short phrases. Limit lists to 3 items maximum."""

DISEASE_PROMPT = """Based on previous analysis:
{base_analysis}

Identify disease characteristics:
1. Pathogen: [scientific + common name]
2. Type: [fungal/bacterial/viral]
3. Symptoms: {symptoms}
4. Severity: [% affected + stage]
5. Lifecycle: [brief description]
6. Risk Factors: {risk_factors}

Provide concise responses using bullet points and short phrases. Limit lists to 3 items maximum."""

SOIL_PROMPT = """Using previous data:
{base_analysis}

Analyze soil:
1. Type: {soil_type}
2. pH: {soil_ph}
3. Organic Matter: [% estimate]
4. Nutrients:
   - Nitrogen: [level]
   - Phosphorus: [level]
   - Potassium: [level]
5. Recommendations: [list]

Provide concise responses using bullet points and short phrases. Limit lists to 3 items maximum."""

# Add Yield Analysis Prompt
YIELD_PROMPT = """Based on this analysis:
{base_analysis}
{disease_analysis}
{soil_analysis}

Estimate crop yield and economic impact:
1. Current Yield Estimate: [kg/ha or ton/acre with reasoning]
2. Potential Loss: [% loss + reasons]
3. Optimization Strategies: [list 5 specific actions]
4. Economic Impact: [USD/ha analysis]

Provide concise responses using bullet points and short phrases. Limit lists to 3 items maximum."""

# Update Management Prompt
MANAGEMENT_PROMPT = """Create detailed management plan from:
{base_analysis}
{disease_analysis}
{soil_analysis}

Structure your response:

[Irrigation]
Method: [drip/sprinkler/flood]
Schedule: [frequency + timing]
Water Requirements: [daily/weekly amount]
Equipment: [list 3 specific tools]

[Fertilization]
NPK Ratio: [specific ratio]
Fertilizers: [list 3 brand names]
Application Method: [foliar/soil injection/etc]
Schedule: [growth stage-based timing]
Dosage: [amount per acre]

[Cultural Practices]
- [Detailed list of cultivation practices]
- [Sanitation measures]
- [Prevention techniques]

[Biological Controls]
- [list 3 natural controls]

[Chemical Treatments]
- [list 3 specific products]

Provide concise responses using bullet points and short phrases. Limit lists to 3 items maximum."""

# Enhanced Helper Functions with fallbacks
def build_user_input_section(user_input: UserInput) -> str:
    sections = []
    if user_input.plant_name:
        sections.append(f"- User-Provided Plant: {user_input.plant_name}")
    if user_input.growth_stage:
        sections.append(f"- User-Provided Growth Stage: {user_input.growth_stage}")
    if user_input.soil_type:
        sections.append(f"- User-Provided Soil Type: {user_input.soil_type}")
    if user_input.soil_ph:
        sections.append(f"- User-Provided Soil pH: {user_input.soil_ph}")
    if user_input.region:
        sections.append(f"- User-Provided Region: {user_input.region}")
    return "\n".join(sections)

async def analyze_image_once(image_path: str, prompt: str) -> str:
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

async def analyze_text(prompt: str) -> str:
    response: ChatResponse = chat(model='gemma3', messages=[{'role': 'user', 'content': prompt}])
    return response.message.content

def clean_markdown(text: str) -> str:
    """Remove markdown formatting from text"""
    return re.sub(r'\*+', '', text).strip()

def safe_extract(text: str, key: str, default: str = "Data not available") -> str:
    patterns = [
        rf'{key}[:\-]+(.*?)(?=\n\s*\w+|\Z)',  # Matches key: value or key - value
        rf'{key}\s+-\s+(.*?)(?=\n|$)',
        rf'{key}\n(.*?)(?=\n|$)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return clean_markdown(match.group(1).strip())
    return default

def safe_extract_list(text: str, key: str) -> List[str]:
    section = safe_extract(text, key, "")
    items = re.findall(r'(?:\d+\.?|\*|\-)\s+(.*?)(?=\n\s*(?:\d+\.?|\*|\-)|$)', section)
    return [clean_markdown(item) for item in items if item.strip()]


def extract_ph_value(ph_text: str) -> float:
    """Improved pH extraction with range handling and fallbacks"""
    try:
        # First try to find explicit pH pattern
        ph_match = re.search(
            r'pH[\s\-]*:?\s*(\d+\.?\d*)\s*[-–]?\s*(\d+\.?\d*)?', 
            ph_text, 
            re.IGNORECASE
        )
        
        if ph_match:
            # Use first value if range is found (e.g., 6.5-7.2 -> 6.5)
            value = float(ph_match.group(1))
            if 0 <= value <= 14:
                return value

        # Fallback to first numeric value in text
        numbers = re.findall(r'\d+\.?\d*', ph_text)
        if numbers:
            value = float(numbers[0])
            if 0 <= value <= 14:
                return value
            return float(numbers[0])  # Return even if out of range

        return 7.0  # Final fallback
    except (ValueError, TypeError):
        return 7.0

# Analysis Pipeline with user input integration
async def analyze_crop_structure(image_path: str, user_input: UserInput) -> FullReport:
    # Base analysis with user input
    base_prompt = BASE_PROMPT.format(
        user_input_section=build_user_input_section(user_input),
        plant_name=user_input.plant_name or "[Detect from image]",
        growth_stage=user_input.growth_stage or "[Estimate growth stage]",
        soil_type=user_input.soil_type or "[Analyze soil characteristics]",
        health_status="[Assess plant health]",
        visual_symptoms="[List visible symptoms]"
    )
    
    base_analysis = await analyze_image_once(image_path, base_prompt)
    
    # First stage: Independent analyses
    disease_task = analyze_text(DISEASE_PROMPT.format(
        base_analysis=base_analysis,
        symptoms=user_input.plant_name or "common symptoms",
        risk_factors=user_input.region or "general risk factors"
    ))
    
    soil_task = analyze_text(SOIL_PROMPT.format(
        base_analysis=base_analysis,
        soil_type=user_input.soil_type or "[Analyze soil type]",
        soil_ph=user_input.soil_ph or "[Estimate pH]"
    ))
    
    disease_text, soil_text = await asyncio.gather(disease_task, soil_task)
    
    # Second stage: Dependent analyses
    management_task = analyze_text(MANAGEMENT_PROMPT.format(
        base_analysis=base_analysis,
        disease_analysis=disease_text,
        soil_analysis=soil_text
    ))
    
    yield_task = analyze_text(YIELD_PROMPT.format(
        base_analysis=base_analysis,
        disease_analysis=disease_text,
        soil_analysis=soil_text
    ))
    
    management_text, yield_text = await asyncio.gather(management_task, yield_task)

    return FullReport(
        crop=CropIdentification(
            plant_name=safe_extract(base_analysis, "Plant Name", "Unknown Plant"),
            scientific_name=safe_extract(base_analysis, "Scientific Name", "Not available"),
            growth_stage=safe_extract(base_analysis, "Growth Stage", "Unknown Stage"),
            health_status=safe_extract(base_analysis, "Health Status", "Unknown"),
            visual_symptoms=safe_extract_list(base_analysis, "Visual Symptoms") or ["No visible symptoms detected"]
        ),
        disease=DiseaseAnalysis(
            pathogen=safe_extract(disease_text, "Pathogen", "Unknown Pathogen"),
            pathogen_type=safe_extract(disease_text, "Type", "Not identified"),
            symptoms=safe_extract_list(disease_text, "Symptoms") or ["No specific symptoms identified"],
            severity=safe_extract(disease_text, "Severity", "Not assessed"),
            lifecycle=safe_extract(disease_text, "Lifecycle", "Lifecycle information unavailable"),
            risk_factors=safe_extract_list(disease_text, "Risk Factors") or ["General risk factors present"]
        ),
        soil=SoilAnalysis(
            soil_type=safe_extract(soil_text, "Type", user_input.soil_type or "Soil type not determined"),
            soil_ph=extract_ph_value(safe_extract(soil_text, "pH", str(user_input.soil_ph or "6.5"))),
            organic_matter=safe_extract(soil_text, "Organic Matter", "Not estimated"),
            nutrient_levels={
                "N": safe_extract(soil_text, "Nitrogen", "N/A"),
                "P": safe_extract(soil_text, "Phosphorus", "N/A"),
                "K": safe_extract(soil_text, "Potassium", "N/A")
            },
            recommendations=safe_extract_list(soil_text, "Recommendations") or ["General soil amendments recommended"]
        ),
        management=ManagementPlan(
            irrigation=IrrigationPlan(
                method=safe_extract(management_text, "Method", "Drip irrigation"),
                schedule=safe_extract(management_text, "Schedule", "Every 3-5 days"),
                water_requirements=safe_extract(management_text, "Water Requirements", "2-4 cm/week"),
                equipment_recommendations=safe_extract_list(management_text, "Equipment") or ["Standard irrigation tools"]
            ),
            fertilization=FertilizationPlan(
                npk_ratio=safe_extract(management_text, "NPK Ratio", "10-10-10"),
                fertilizer_types=safe_extract_list(management_text, "Fertilizers") or ["Balanced NPK fertilizer"],
                application_method=safe_extract(management_text, "Application Method", "Soil incorporation"),
                schedule=safe_extract(management_text, "Schedule", "Bi-weekly"),
                dosage=safe_extract(management_text, "Dosage", "As per manufacturer guidelines")
            ),
            cultural_practices=safe_extract_list(management_text, "Cultural Practices") or ["Standard cultivation practices"],
            biological_controls=safe_extract_list(management_text, "Biological Controls") or ["Natural predators"],
            chemical_treatments=safe_extract_list(management_text, "Chemical Treatments") or ["General fungicides"]
        ),
        yield_data=YieldAnalysis(
            current_estimate=safe_extract(yield_text, "Current Yield Estimate", "Calculating..."),
            potential_loss=safe_extract(yield_text, "Potential Loss", "Estimating..."),
            optimization_strategies=safe_extract_list(yield_text, "Optimization Strategies") or ["Crop rotation", "Improved irrigation"],
            economic_impact=safe_extract(yield_text, "Economic Impact", "Pending analysis")
        )
    )

# PDF Generation with improved formatting
def generate_pdf_report(image_path: str, report: FullReport, filename: str) -> str:
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name='Header1', fontSize=14, textColor=colors.darkgreen, 
        spaceAfter=6, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Header2', fontSize=12, textColor=colors.darkblue,
        spaceAfter=4, fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        name='Body', fontSize=10, leading=12, 
        spaceAfter=4, wordWrap='LTR'
    ))

    elements = []
    
    # Header Section
    elements.append(Paragraph("Comprehensive Crop Analysis Report", styles['Header1']))
    elements.append(Image(image_path, width=5*inch, height=3.5*inch, kind='proportional'))
    elements.append(Spacer(1, 12))

    def create_section(title: str, content):
        elements.append(Paragraph(title, styles['Header2']))
        elements.append(content)
        elements.append(Spacer(1, 8))

    # Crop Identification Section
    crop_content = [
        ["Plant Name:", Paragraph(report.crop.plant_name, styles['Body'])],
        ["Scientific Name:", Paragraph(report.crop.scientific_name, styles['Body'])],
        ["Growth Stage:", Paragraph(report.crop.growth_stage, styles['Body'])],
        ["Health Status:", Paragraph(report.crop.health_status, styles['Body'])],
        ["Visual Symptoms:", Paragraph('\n'.join([f"• {s}" for s in report.crop.visual_symptoms]), styles['Body'])]
    ]
    crop_table = Table(crop_content, colWidths=[1.8*inch, 5.2*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    create_section("1. Crop Identification", crop_table)

    # Disease Analysis Section
    disease_content = [
        ["Pathogen:", Paragraph(f"{report.disease.pathogen} ({report.disease.pathogen_type})", styles['Body'])],
        ["Severity:", Paragraph(report.disease.severity, styles['Body'])],
        ["Lifecycle:", Paragraph(report.disease.lifecycle, styles['Body'])],
        ["Symptoms:", Paragraph('\n'.join([f"• {s}" for s in report.disease.symptoms]), styles['Body'])],
        ["Risk Factors:", Paragraph('\n'.join([f"• {rf}" for rf in report.disease.risk_factors]), styles['Body'])]
    ]
    disease_table = Table(disease_content, colWidths=[1.8*inch, 5.2*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    create_section("2. Disease Analysis", disease_table)

    # Soil Analysis Section
    soil_content = [
        ["Soil Type:", Paragraph(report.soil.soil_type, styles['Body'])],
        ["pH Level:", Paragraph(f"{report.soil.soil_ph:.1f}", styles['Body'])],
        ["Organic Matter:", Paragraph(report.soil.organic_matter, styles['Body'])],
        ["Nutrient Levels:", Paragraph(
            f"Nitrogen (N): {report.soil.nutrient_levels['N']}\n"
            f"Phosphorus (P): {report.soil.nutrient_levels['P']}\n"
            f"Potassium (K): {report.soil.nutrient_levels['K']}", 
            styles['Body']
        )],
        ["Recommendations:", Paragraph('\n'.join([f"• {r}" for r in report.soil.recommendations]), styles['Body'])]
    ]
    soil_table = Table(soil_content, colWidths=[1.8*inch, 5.2*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    create_section("3. Soil Analysis", soil_table)

    # Management Plan Section
    management_elements = [
        Paragraph("4. Management Plan", styles['Header1']),
        Spacer(1, 8)
    ]

    # Irrigation Plan
    irrigation_content = [
        ["Method:", Paragraph(report.management.irrigation.method, styles['Body'])],
        ["Schedule:", Paragraph(report.management.irrigation.schedule, styles['Body'])],
        ["Water Requirements:", Paragraph(report.management.irrigation.water_requirements, styles['Body'])],
        ["Equipment:", Paragraph('\n'.join([f"• {e}" for e in report.management.irrigation.equipment_recommendations]), styles['Body'])]
    ]
    irrigation_table = Table(irrigation_content, colWidths=[1.8*inch, 5.2*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    management_elements.extend([Paragraph("Irrigation Plan", styles['Header2']), irrigation_table, Spacer(1, 8)])

    # Fertilization Plan
    fertilization_content = [
        ["NPK Ratio:", Paragraph(report.management.fertilization.npk_ratio, styles['Body'])],
        ["Fertilizers:", Paragraph('\n'.join([f"• {f}" for f in report.management.fertilization.fertilizer_types]), styles['Body'])],
        ["Application Method:", Paragraph(report.management.fertilization.application_method, styles['Body'])],
        ["Schedule:", Paragraph(report.management.fertilization.schedule, styles['Body'])],
        ["Dosage:", Paragraph(report.management.fertilization.dosage, styles['Body'])]
    ]
    fertilization_table = Table(fertilization_content, colWidths=[1.8*inch, 5.2*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    management_elements.extend([Paragraph("Fertilization Plan", styles['Header2']), fertilization_table, Spacer(1, 8)])

    # Additional Management Components
    management_elements.extend([
        Paragraph("Cultural Practices", styles['Header2']),
        Paragraph('\n'.join([f"• {cp}" for cp in report.management.cultural_practices]), styles['Body']),
        Spacer(1, 8),
        Paragraph("Biological Controls", styles['Header2']),
        Paragraph('\n'.join([f"• {bc}" for bc in report.management.biological_controls]), styles['Body']),
        Spacer(1, 8),
        Paragraph("Chemical Treatments", styles['Header2']),
        Paragraph('\n'.join([f"• {ct}" for ct in report.management.chemical_treatments]), styles['Body'])
    ])
    
    elements.extend(management_elements)

    # Yield Analysis Section
    yield_content = [
        ["Current Estimate:", Paragraph(report.yield_data.current_estimate, styles['Body'])],
        ["Potential Loss:", Paragraph(report.yield_data.potential_loss, styles['Body'])],
        ["Optimization Strategies:", Paragraph('\n'.join([f"• {os}" for os in report.yield_data.optimization_strategies]), styles['Body'])],
        ["Economic Impact:", Paragraph(report.yield_data.economic_impact, styles['Body'])]
    ]
    yield_table = Table(yield_content, colWidths=[2.2*inch, 4.8*inch], style=[
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.lightgrey)
    ])
    create_section("5. Yield Analysis", yield_table)

    # Footer
    elements.append(Spacer(1, 15))
    elements.append(Paragraph(
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ParagraphStyle(
            name='Footer',
            fontSize=9,
            textColor=colors.grey,
            alignment=1  # Center aligned
        )
    ))

    doc.build(elements)
    return pdf_path

@app.post("/analyze-crop", response_model=StructuredResponse)
async def analyze_crop_endpoint(
    file: UploadFile = File(...),
    user_input: Optional[str] = Form(None)
):
    try:
        # Validate input
        if not file.content_type.startswith('image/'):
            raise HTTPException(400, "Invalid image format")
        
        # Parse user input
        user_data = UserInput()
        if user_input:
            try:
                user_data = UserInput.model_validate_json(user_input)
            except ValidationError as e:
                raise HTTPException(422, detail=e.errors())
        
        # Process file
        filename = f"{uuid.uuid4()}.{file.filename.split('.')[-1]}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())

        # Generate analysis
        report = await analyze_crop_structure(file_path, user_data)
        
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