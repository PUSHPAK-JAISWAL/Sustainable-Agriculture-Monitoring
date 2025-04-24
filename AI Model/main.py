from fastapi import FastAPI, Form, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ollama import ChatResponse, chat
from pydantic import BaseModel, ValidationError, validator
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

# Temporary storage config
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ========== Pydantic Models ==========
class AnalysisRequest(BaseModel):
    plant_name: Optional[str] = None
    growth_stage: Optional[str] = None
    soil_type: Optional[str] = None
    soil_ph: Optional[float] = None
    region: Optional[str] = None

class CropOverview(BaseModel):
    type: str
    growthStage: str
    soilPreferences: str
    estimatedSoilPH: Dict[str, float]
    region: str

class DiseaseAnalysis(BaseModel):
    pathogen: str
    symptoms: List[str]
    severity: str
    riskFactors: List[str]

class YieldProjection(BaseModel):
    currentEstimate_kgPerHa: Dict[str, int]
    potentialLossPercent: Dict[str, int]
    optimizationPotentialPercent: Dict[str, int]

class IrrigationPlan(BaseModel):
    method: str
    schedule_daysBetween: Dict[str, int]
    waterRequirement_mmPerDay: Dict[str, int]

class NutritionStage(BaseModel):
    NPK: List[int]
    application: Optional[str] = None
    applicationFrequency: Optional[str] = None

class NutritionPlan(BaseModel):
    vegetativeStage: NutritionStage
    floweringFruiting: NutritionStage
    soilAmendments: List[str]

class CulturalPractices(BaseModel):
    pruning: List[str]
    cropRotation: str
    diseasePrevention: List[str]

class ManagementPlan(BaseModel):
    irrigation: IrrigationPlan
    nutrition: NutritionPlan
    culturalPractices: CulturalPractices

class AnalysisReport(BaseModel):
    cropOverview: CropOverview
    diseaseAnalysis: DiseaseAnalysis
    yieldProjection: YieldProjection
    managementPlan: ManagementPlan
    disclaimer: str

class StructuredResponse(BaseModel):
    status: str
    data: Dict

async def analyze_image_with_ollama(image_path: str, request_data: AnalysisRequest) -> str:
    """
    Analyzes agricultural images using Ollama's AI model with enhanced error handling
    and model configuration.
    """
    try:
        # Validate image file existence
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Read and encode image
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

        # Build dynamic prompt with fallback values
        prompt = f"""Analyze this agricultural image and provide a professional crop health report with these sections:

## Crop Overview
- Type: {request_data.plant_name or "Detect from image"}
- Growth Stage: {request_data.growth_stage or "Estimate from plant morphology"}
- Soil Preferences: {request_data.soil_type or "Suggest based on visual clues"}
- Soil pH: {request_data.soil_ph or "Estimate from soil color"}
- Region: {request_data.region or "Identify from image"}

## Disease Analysis
1. Pathogen Identification: [Scientific name if detectable]
2. Symptom Description: [Detailed visual indicators]
3. Severity Assessment: Low/Medium/High
4. Risk Factors: [Environmental conditions contributing]

## Yield Projection
- Current Estimate: [kg/ha or lbs/acre]
- Potential Losses: [% if disease present]
- Optimization Potential: [% with interventions]

## Management Plan
### Irrigation
- Recommended Schedule: [Days between watering]
- Method: [Drip/Sprinkle/Flood]
- Water Requirements: [mm/day or inches/week]

### Nutrition
- Vegetative Stage: [NPK ratio] 
- Application Schedule: [Timing details]
- Flowering/Fruiting Stage: [NPK ratio]
- Application Frequency: [Frequency details] 
- Soil Amendments: [List of amendments]

### Cultural Practices
- Pruning Recommendations
- Crop Rotation Advice
- Disease Prevention Measures

Format using markdown headers (##) and bullet points. Be concise and technical."""

        # Configure model parameters
        response : ChatResponse = chat(
            model='gemma3',  # Using a more stable model version
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [base64_image]
            }]
        )

        # Validate response format
        if not response.message or not response.message.content:
            raise ValueError("Empty response from AI model")

        # Basic content validation
        required_sections = [
            "Crop Overview",
            "Disease Analysis",
            "Yield Projection",
            "Management Plan"
        ]
        
        for section in required_sections:
            if section not in response.message.content:
                raise ValueError(f"Missing required section in response: {section}")

        return response.message.content

    except FileNotFoundError as fnf_error:
        raise HTTPException(status_code=400, detail=str(fnf_error))
    except IOError as io_error:
        raise HTTPException(status_code=500, detail=f"Image reading error: {str(io_error)}")
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"AI analysis failed: {str(e)}. Please check: "
                   "1. Ollama server is running\n"
                   "2. Model is downloaded (try 'ollama pull gemma:7b')\n"
                   "3. System resources are available"
        )

# ========== Helper Functions ==========
# ========== Updated Helper Functions ==========
def parse_analysis_text(analysis: str) -> AnalysisReport:
    def extract_section(content: str, section_name: str) -> str:
        # More flexible section matching with header variations
        pattern = rf'#+\s*{re.escape(section_name)}.*?\n(.*?)(?=\n#+\s*|\Z)'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def extract_key_value(section: str, key: str) -> str:
        # Multiple pattern matching for different markdown variations
        patterns = [
            rf'-\s*\*?{re.escape(key)}:?\*?\s*-\s*(.+?)(?=\n\s*-|\Z)',  # - *Key* - Value
            rf'\*?{re.escape(key)}:?\*?\s*(.+?)(?=\n\s*\*|\Z)',        # *Key* Value
            rf'{re.escape(key)}:\s*(.+?)(?=\n|$)'                     # Key: Value
        ]
        
        for pattern in patterns:
            match = re.search(pattern, section, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return "Data not available"

    def parse_npk(text: str) -> List[int]:
        """
        Extracts NPK values from text with multiple format support:
        - "10-20-30"
        - "15:15:30"
        - "NPK 20/10/5"
        - "10 20 30"
        """
        try:
            # Match common NPK patterns using regex
            match = re.search(r'(\d+)[/\-: ]+(\d+)[/\-: ]+(\d+)', text)
            if match:
                return [int(match.group(1)), int(match.group(2)), int(match.group(3))]
            
            # Fallback to simple number extraction
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 3:
                return list(map(int, numbers[:3]))
            if len(numbers) > 0:
                return list(map(int, numbers)) + [0]*(3-len(numbers))
                
            return [0, 0, 0]
        except Exception as e:
            print(f"NPK parsing error: {str(e)}")
            return [0, 0, 0]

    def parse_range(text: str) -> Dict[str, float]:
        # Handle percentages and different range formats
        text = text.replace('%', '').replace(' ', '')
        numbers = re.findall(r'\d+\.?\d*', text)
        return {
            'min': float(numbers[0]) if numbers else 0.0,
            'max': float(numbers[1]) if len(numbers) > 1 else float(numbers[0]) if numbers else 0.0
        }

    def safe_split(text: str, delimiter: str = '[;,]+') -> List[str]:
        # Split with multiple possible delimiters
        return [item.strip() for item in re.split(delimiter, text) if item.strip()]

    # Enhanced section parsing with fallbacks
    co_section = extract_section(analysis, "Crop Overview")
    crop_overview = CropOverview(
        type=extract_key_value(co_section, "Type") or "Crop type not identified",
        growthStage=extract_key_value(co_section, "Growth Stage") or "Growth stage not determined",
        soilPreferences=extract_key_value(co_section, "Soil Preferences") or "Soil type not specified",
        estimatedSoilPH=parse_range(extract_key_value(co_section, "Soil pH")),
        region=extract_key_value(co_section, "Region") or "Region not specified"
    )

    # Disease Analysis with empty list prevention
    da_section = extract_section(analysis, "Disease Analysis")
    disease_analysis = DiseaseAnalysis(
        pathogen=extract_key_value(da_section, "Pathogen Identification") or "No pathogen identified",
        symptoms=safe_split(extract_key_value(da_section, "Symptom Description")),
        severity=extract_key_value(da_section, "Severity Assessment") or "Not assessed",
        riskFactors=safe_split(extract_key_value(da_section, "Risk Factors"))
    )

    # Yield Projection with percentage handling
    yp_section = extract_section(analysis, "Yield Projection")
    yield_projection = YieldProjection(
        currentEstimate_kgPerHa=parse_range(extract_key_value(yp_section, "Current Estimate")),
        potentialLossPercent=parse_range(extract_key_value(yp_section, "Potential Losses")),
        optimizationPotentialPercent=parse_range(extract_key_value(yp_section, "Optimization Potential"))
    )

    # Management Plan parsing with subsections
    mp_section = extract_section(analysis, "Management Plan")
    
    # Irrigation with unit normalization
    irr_section = extract_section(mp_section, "Irrigation")
    irrigation = IrrigationPlan(
        method=extract_key_value(irr_section, "Method") or "Irrigation method not specified",
        schedule_daysBetween=parse_range(extract_key_value(irr_section, "Recommended Schedule")),
        waterRequirement_mmPerDay=parse_range(extract_key_value(irr_section, "Water Requirements"))
    )

    # Nutrition with NPK validation
    nut_section = extract_section(mp_section, "Nutrition")
    nutrition = NutritionPlan(
        vegetativeStage=NutritionStage(
            NPK=parse_npk(extract_key_value(nut_section, "Vegetative Stage")) or [0, 0, 0],
            application=extract_key_value(nut_section, "Application Schedule") or "Not specified",
            applicationFrequency=None
        ),
        floweringFruiting=NutritionStage(
            NPK=parse_npk(extract_key_value(nut_section, "Flowering/Fruiting")) or [0, 0, 0],
            application=extract_key_value(nut_section, "Application Schedule") or "Not specified",
            applicationFrequency=extract_key_value(nut_section, "Application Frequency") or "Frequency not specified"
        ),
        soilAmendments=safe_split(extract_key_value(nut_section, "Soil Amendments"))
    )

    # Cultural Practices with empty list handling
    cp_section = extract_section(mp_section, "Cultural Practices")
    cultural_practices = CulturalPractices(
        pruning=safe_split(extract_key_value(cp_section, "Pruning Recommendations")),
        cropRotation=extract_key_value(cp_section, "Crop Rotation Advice") or "No rotation advice provided",
        diseasePrevention=safe_split(extract_key_value(cp_section, "Disease Prevention Measures"))
    )

    return AnalysisReport(
        cropOverview=crop_overview,
        diseaseAnalysis=disease_analysis,
        yieldProjection=yield_projection,
        managementPlan=ManagementPlan(
            irrigation=irrigation,
            nutrition=nutrition,
            culturalPractices=cultural_practices
        ),
        disclaimer="This assessment is based solely on visual image data. Field inspection and laboratory diagnostics are recommended."
    )

# ========== PDF Generation ==========
def create_pdf_element(content: str, style: ParagraphStyle, bullet: bool = False):
    if bullet:
        return Paragraph(f"• {content}", style)
    return Paragraph(content, style)

def generate_pdf_report(image_path: str, report: AnalysisReport, filename: str) -> str:
    pdf_path = os.path.join(UPLOAD_DIR, filename)
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Header1', fontSize=14, leading=18, spaceAfter=12, textColor=colors.darkgreen))
    styles.add(ParagraphStyle(name='Header2', fontSize=12, leading=16, spaceAfter=8, textColor=colors.green))
    styles.add(ParagraphStyle(name='Body', fontSize=11, leading=14, spaceAfter=6, textColor=colors.darkgrey))
    
    elements = []
    
    # Title and Image
    elements.append(Paragraph("Crop Health Analysis Report", styles['Header1']))
    elements.append(Spacer(1, 12))
    elements.append(Image(image_path, width=4*inch, height=3*inch))
    elements.append(Spacer(1, 15))
    
    # Crop Overview Table
    co_data = [
        ["Crop Type", report.cropOverview.type],
        ["Growth Stage", report.cropOverview.growthStage],
        ["Soil Preferences", report.cropOverview.soilPreferences],
        ["Soil pH Range", 
            f"{report.cropOverview.estimatedSoilPH.get('min', 0.0)}-{report.cropOverview.estimatedSoilPH.get('max', 0.0)}"],
        ["Region", report.cropOverview.region]
    ]
    elements.append(Paragraph("Crop Overview", styles['Header2']))
    elements.append(Table(co_data, style=[('BACKGROUND', (0,0), (-1,0), colors.lightgreen)]))
    elements.append(Spacer(1, 12))
    
    # Disease Analysis
    elements.append(Paragraph("Disease Analysis", styles['Header2']))
    elements.append(Paragraph(f"Pathogen: {report.diseaseAnalysis.pathogen}", styles['Body']))
    elements.append(Paragraph("Symptoms:", styles['Body']))
    for symptom in report.diseaseAnalysis.symptoms:
        elements.append(create_pdf_element(symptom, styles['Body'], bullet=True))
    elements.append(Spacer(1, 12))
    
    # Yield Projection
    yp_data = [
        ["Metric", "Min", "Max"],
        ["Current Estimate (kg/ha)", 
        report.yieldProjection.currentEstimate_kgPerHa.get('min', 0), 
        report.yieldProjection.currentEstimate_kgPerHa.get('max', 0)],
        ["Potential Loss (%)", 
        report.yieldProjection.potentialLossPercent.get('min', 0), 
        report.yieldProjection.potentialLossPercent.get('max', 0)],
        ["Optimization Potential (%)", 
        report.yieldProjection.optimizationPotentialPercent.get('min', 0), 
        report.yieldProjection.optimizationPotentialPercent.get('max', 0)]
    ]
    elements.append(Paragraph("Yield Projection", styles['Header2']))
    elements.append(Table(yp_data, style=[('BACKGROUND', (0,0), (-1,0), colors.lightgreen)]))
    elements.append(Spacer(1, 12))
    
    # Management Plan
    elements.append(Paragraph("Management Plan", styles['Header1']))
    
    # Irrigation
    elements.append(Paragraph("Irrigation", styles['Header2']))
    irr_data = [
        ["Method", report.managementPlan.irrigation.method],
        ["Schedule (days)", 
        f"{report.managementPlan.irrigation.schedule_daysBetween.get('min', 0)}-{report.managementPlan.irrigation.schedule_daysBetween.get('max', 0)}"],
        ["Water Requirement (mm/day)", 
        f"{report.managementPlan.irrigation.waterRequirement_mmPerDay.get('min', 0)}-{report.managementPlan.irrigation.waterRequirement_mmPerDay.get('max', 0)}"]
    ]
    elements.append(Table(irr_data))
    elements.append(Spacer(1, 10))
    
    # Nutrition
    elements.append(Paragraph("Nutrition", styles['Header2']))
    nut_data = [
        ["Stage", "NPK", "Application"],
        ["Vegetative", 
         "-".join(map(str, report.managementPlan.nutrition.vegetativeStage.NPK)), 
         report.managementPlan.nutrition.vegetativeStage.application],
        ["Flowering/Fruiting", 
         "-".join(map(str, report.managementPlan.nutrition.floweringFruiting.NPK)), 
         report.managementPlan.nutrition.floweringFruiting.applicationFrequency]
    ]
    elements.append(Table(nut_data, style=[('BACKGROUND', (0,0), (-1,0), colors.lightgreen)]))
    elements.append(Paragraph(f"Soil Amendments: {', '.join(report.managementPlan.nutrition.soilAmendments)}", styles['Body']))
    elements.append(Spacer(1, 10))
    
    # Cultural Practices
    elements.append(Paragraph("Cultural Practices", styles['Header2']))
    elements.append(Paragraph("Pruning Recommendations:", styles['Body']))
    for practice in report.managementPlan.culturalPractices.pruning:
        elements.append(create_pdf_element(practice, styles['Body'], bullet=True))
    
    elements.append(Paragraph(f"Crop Rotation: {report.managementPlan.culturalPractices.cropRotation}", styles['Body']))
    
    elements.append(Paragraph("Disease Prevention:", styles['Body']))
    for prevention in report.managementPlan.culturalPractices.diseasePrevention:
        elements.append(create_pdf_element(prevention, styles['Body'], bullet=True))
    
    # Footer
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} | AI Agriculture Assistant", 
                            ParagraphStyle(name='Footer', fontSize=9, textColor=colors.grey)))
    
    doc.build(elements)
    return pdf_path

@app.post("/analyze-crop", response_model=StructuredResponse)
async def analyze_crop(
    file: UploadFile = File(..., description="Image file of the crop (JPEG/PNG)"),
    request_data: Optional[str] = Form(
        None,
        example='{"plant_name": "Tomatoes", "soil_ph": 6.5}',
        description="Optional JSON parameters for analysis"
    )
):
    file_path = None
    try:
        # Parse and validate request data
        request_obj = AnalysisRequest()
        if request_data:
            try:
                request_obj = AnalysisRequest.model_validate_json(request_data)
            except ValidationError as ve:
                error_messages = []
                for error in ve.errors():
                    field = ".".join(str(loc) for loc in error['loc'])
                    error_messages.append(f"{field}: {error['msg']}")
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Request data validation failed",
                        "fields": error_messages
                    }
                )

        # Validate file type
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in {'png', 'jpg', 'jpeg'}:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid file type",
                    "supported_types": ["png", "jpg", "jpeg"]
                }
            )

        # Process file
        filename = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        async with aiofiles.open(file_path, "wb") as buffer:
            await buffer.write(await file.read())

        # Analysis pipeline
        try:
            analysis_text = await analyze_image_with_ollama(file_path, request_obj)
            parsed_report = parse_analysis_text(analysis_text)
        except HTTPException as he:
            raise he
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Analysis failed", "message": str(e)}
            )

        # Generate PDF report
        try:
            pdf_filename = f"report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
            pdf_path = generate_pdf_report(file_path, parsed_report, pdf_filename)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Report generation failed", "message": str(e)}
            )

        # Cleanup temporary files
        if os.path.exists(file_path):
            os.remove(file_path)

        return StructuredResponse(
            status="success",
            data={
                "report": parsed_report.dict(),
                "pdf": {
                    "filePath": pdf_path.replace("\\", "/"),
                    "url": f"/reports/{pdf_filename}"
                }
            }
        )

    except HTTPException as he:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise he
    except Exception as e:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=500,
            detail={"error": "Unexpected server error", "message": str(e)}
        )