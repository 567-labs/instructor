# pip install openai instructor
from pydantic import BaseModel, field_validator, Field
import openai
import instructor
from tqdm import tqdm

client = instructor.from_openai(openai.OpenAI())

classes = {
    "11-0000": "Management",
    "13-0000": "Business and Financial Operations",
    "15-0000": "Computer and Mathematical",
    "17-0000": "Architecture and Engineering",
    "19-0000": "Life, Physical, and Social Science",
    "21-0000": "Community and Social Service",
    "23-0000": "Legal",
    "25-0000": "Education Instruction and Library",
    "27-0000": "Arts, Design, Entertainment, Sports and Media",
    "29-0000": "Healthcare Practitioners and Technical",
    "31-0000": "Healthcare Support",
    "33-0000": "Protective Service",
    "35-0000": "Food Preparation and Serving",
    "37-0000": "Building and Grounds Cleaning and Maintenance",
    "39-0000": "Personal Care and Service",
    "41-0000": "Sales and Related",
    "43-0000": "Office and Administrative Support",
    "45-0000": "Farming, Fishing and Forestry",
    "47-0000": "Construction and Extraction",
    "49-0000": "Installation, Maintenance, and Repair",
    "51-0000": "Production Occupations",
    "53-0000": "Transportation and Material Moving",
    "55-0000": "Military Specific",
    "99-0000": "Other",
}


class SOCCode(BaseModel):
    reasoning: str = Field(
        default=None,
        description="Step-by-step reasoning to get the correct classification",
    )
    code: str

    @field_validator("code")
    def validate_code(cls, v):
        if v not in classes:
            raise ValueError(f"Invalid SOC code, {v}")
        return v


def classify_job(description: str) -> SOCCode:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=SOCCode,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": f"You are an expert at classifying job descriptions into Standard Occupational Classification (SOC) codes. from the following list: {classes}",
            },
            {
                "role": "user",
                "content": f"Classify this job description into the most appropriate SOC code: {description}",
            },
        ],
    )
    return response


if __name__ == "__main__":
    # gpt-3.5-turbo: 16/20
    # gpt-3.5-turbo (COT): 18/20
    # gpt-4-turbo: 20/20

    job_descriptions = [
        (
            "Develop and design complex software applications for various industries, including finance, healthcare, and e-commerce",
            "15-0000",  # Computer and Mathematical Occupations
        ),
        (
            "Provide comprehensive technical support and troubleshooting for enterprise-level software products, ensuring seamless user experience",
            "15-0000",  # Computer and Mathematical Occupations
        ),
        (
            "Teach a diverse range of subjects to elementary school students, fostering their intellectual and social development",
            "25-0000",  # Education, Training, and Library Occupations
        ),
        (
            "Conduct cutting-edge research in various academic fields at a renowned university, contributing to the advancement of knowledge",
            "25-0000",  # Education, Training, and Library Occupations
        ),
        (
            "Design visually appealing and strategically effective logos, branding, and marketing materials for clients across different industries",
            "27-0000",  # Arts, Design, Entertainment, Sports, and Media Occupations
        ),
        (
            "Perform as part of a professional musical group, entertaining audiences and showcasing artistic talent",
            "27-0000",  # Arts, Design, Entertainment, Sports, and Media Occupations
        ),
        (
            "Diagnose and treat a wide range of injuries and medical conditions, providing comprehensive healthcare services to patients",
            "29-0000",  # Healthcare Practitioners and Technical Occupations
        ),
        (
            "Assist doctors and nurses in delivering high-quality patient care, ensuring the smooth operation of healthcare facilities",
            "31-0000",  # Healthcare Support Occupations
        ),
        (
            "Patrol assigned areas to enforce laws and ordinances, maintaining public safety and order in the community",
            "33-0000",  # Protective Service Occupations
        ),
        (
            "Prepare and serve a diverse menu of delectable meals in a fast-paced restaurant environment",
            "35-0000",  # Food Preparation and Serving Related Occupations
        ),
        (
            "Maintain the cleanliness and upkeep of various buildings and facilities, ensuring a safe and presentable environment",
            "37-0000",  # Building and Grounds Cleaning and Maintenance Occupations
        ),
        (
            "Provide a range of beauty services, such as haircuts, styling, and manicures, to help clients look and feel their best",
            "39-0000",  # Personal Care and Service Occupations
        ),
        (
            "Engage with customers in a retail setting, providing excellent service and assisting them in finding the products they need",
            "41-0000",  # Sales and Related Occupations
        ),
        (
            "Perform a variety of clerical duties in an office environment, supporting the overall operations of the organization",
            "43-0000",  # Office and Administrative Support Occupations
        ),
        (
            "Cultivate and harvest a wide range of crops, contributing to the production of food and other agricultural products",
            "45-0000",  # Farming, Fishing, and Forestry Occupations
        ),
        (
            "Construct and build various structures, including residential, commercial, and infrastructure projects",
            "47-0000",  # Construction and Extraction Occupations
        ),
        (
            "Repair and maintain a diverse range of mechanical equipment, ensuring their proper functioning and longevity",
            "49-0000",  # Installation, Maintenance, and Repair Occupations
        ),
        (
            "Operate specialized machinery and equipment in a manufacturing setting to produce high-quality goods",
            "51-0000",  # Production Occupations
        ),
        (
            "Transport freight and goods across different regions, ensuring timely and efficient delivery",
            "53-0000",  # Transportation and Material Moving Occupations
        ),
        (
            "Serve in the armed forces, protecting the nation and its citizens through various military operations and duties",
            "55-0000",  # Military Specific Occupations
        ),
    ]

    correct = 0
    errors = []
    for description, expected_code in tqdm(job_descriptions):
        try:
            predicted_code = None
            result = classify_job(description)
            predicted_code = result.code
            assert result.code == expected_code, (
                f"Expected {expected_code}, got {result.code} for description: {description}"
            )
            correct += 1
        except Exception as e:
            errors.append(
                f"Got {classes.get(predicted_code, 'Unknown')} expected {classes.get(expected_code, 'Unknown')}"
            )

    print(f"{correct} out of {len(job_descriptions)} tests passed!")
    for error in errors:
        print(error)
