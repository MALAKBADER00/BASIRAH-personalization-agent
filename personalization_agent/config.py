import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_MODEL = "llama3-8b-8192"
OPENAI_MODEL = "gpt-4o"

# Trust Thresholds (removed personas - single victim type)

TRUST_THRESHOLDS = {
    "no_info": (0, 3.9999),
    "basic_info": (4, 6.9999),
    "sensitive_info": (7, 8.9999),
    "critical_info": (9, 10.0000)
}


# Single victim configuration (average user)
VICTIM_CONFIG = {
    "initial_trust": 4.0,
    "trust_increment": 1.0,
    "resistance": 0.5
}

# Information Categories
INFO_CATEGORIES = {
    "basic": ["name", "location", "job"],
    "sensitive": ["phone", "email", "address", "ssn", "credit_card", "account"], 
    "critical": ["password", "credit card PIN"]
}
info_data = {
            "Username / Handle": "@ahmed_rashid",
            "Age": 28,
            "Gender": "Male",
            "Manager's Name": "Khalid Ali",
            "Location (City)": "Muscat",
            "Nationality": "Omani",
            "Current Job Title": "Engineer",
            "Employer or Company Name": "PDO",
            "Education (Degree)": "B.Eng. Electrical",
            "University / College Attended": "Sultan Qaboos University",
            "Relationship Status": "Single",
            "Phone Number": "968-92-345-781",
            "Birthday": "1997-04-18",
            "Languages Spoken": "Arabic, English",
            "Religion": "Islam",
            "Hobbies": "Hiking, electronics",
            "Favorite Brands": "Omantel, Samsung",
            "Family Members Mentioned": "Brother: Salman",
            "Close Friends (Tagged Often)": "Khalid, Saif",
            "Favorite Food": "Shuwa",
            "Favorite Travel Destination": "Dubai",
            "Recent Travel Location": "Bangkok",
            "Current Residence (Neighborhood)": "Al-Khuwair",
            "License Plate Number": "M 12345",
            "Car Make/Model": "Toyota Land Cruiser",
            "phone": "9665-8234",
            "job": "engineer at PDO",
            "address": "Building 42, Al-Khuwair, Muscat",
            "password": "Muscat2024!",
            "ssn": "Civil ID: ***-***-789",
            "account": "Omantel account number ending in 4291",
            "credit_card": "Oman Arab Bank card ending in 8756",
            "Name": "Ahmed Al-Rashid",
            "phone": "9665-8234",
            "email": "ahmed.rashid@gmail.com",
            "location": "Muscat, Oman",
            "job": "engineer at PDO",
            "address": "Building 42, Al-Khuwair, Muscat",
            "password": "Muscat2024!",
            "ssn": "Civil ID: ***-***-789",
            "account": "Omantel account number ending in 4291",
            "credit_card": "Oman Arab Bank card ending in 8756"
        }

PERSONAS = {
        "Name": "Ahmed Al-Rashed",
        "Username / Handle": "@ahmed_rashid",
        "Manager's Name": "Khalid Ali",
        # "Age": 28,
        # "Gender": "Male",
        "Location (City)": "Muscat",
        "Current Job Title": "Engineer",
        "Employer or Company Name": "PDO",
        "Education (Degree)": "B.Eng. Electrical",
        "University / College Attended": "Sultan Qaboos University",
        "Phone Number": "92345781",
        "Email Address": "ahmed.rashid@gmail.com",
        "Birthday": "1997-04-18",
        "Hobbies": "Hiking",
        "Favorite Brands": "Samsung",
        "Family Members Mentioned": "Brother: Salman",
        "Close Friends (Tagged Often)": "Saif",
        # "Favorite Food": "Shuwa",
        # "Favorite Travel Destination": "Dubai",
        # "Recent Travel Location": "Bangkok",
        "Current Residence (Neighborhood)": "Al-Khuwair",
        # "License Plate Number": "M 12345",
        "Car Make/Model": "Toyota Land Cruiser"
    }