def banking_service(age):
    if age < 18:
        return "You are eligible for a Youth Savings Account."
    elif 18 <= age < 25:
        return "You are eligible for a Student Checking Account."
    elif 25 <= age < 60:
        return "You are eligible for a Standard Checking Account."
    elif age >= 60:
        return "You are eligible for a Senior Savings Account."
    else:
        return "Invalid age provided."
    