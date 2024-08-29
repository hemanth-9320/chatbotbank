from flask import Flask, render_template, request, jsonify
import os
import pyttsx3
import base64
from io import BytesIO
from PIL import Image
import predict
import Emotion

app = Flask(__name__)

# Configure the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global variable to track if the user has uploaded an image before
first_upload = True

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def banking_service(age):
    if age < 18:
        return "Minor Savings Accounts: Accounts designed for children, often with parental control, encouraging early savings habits. Custodial Accounts (UGMA/UTMA): Accounts held in the child’s name but managed by an adult, used for future education or other expenses. Prepaid Debit Cards: Controlled spending options with parental oversight, teaching budgeting basics."
    elif 18 <= age <= 25:
        return "Student Accounts: Low or no fees, designed for students with limited income. Mobile Banking: Easy access to banking services via mobile apps, supporting digital natives."
    elif 26 <= age <= 35:
        return "Credit Cards: Building credit history with rewards and cashback tailored to lifestyle. Investment Accounts: Options for stocks, mutual funds, and retirement savings. Home Loans: Flexible mortgage options for first-time homebuyers."
    elif 36 <= age <= 50:
        return "Retirement Planning: Pension plans and retirement savings accounts. Wealth Management: Personalized financial planning and investment strategies. Insurance Services: Life, health, and disability insurance for family protection."
    elif 51 <= age <= 65:
        return "Retirement Savings Accounts: Focus on maximizing retirement savings. Long-Term Care Insurance: Coverage for potential healthcare needs in retirement. Estate Planning Services: Trusts, wills, and wealth transfer planning."
    elif age > 65:
        return "Pension and Annuity Management: Steady income sources post-retirement. Healthcare Savings Accounts: Plans like Medicare Supplement Savings Accounts. Safe Deposit Boxes: Secure storage for important documents and valuables."
    else:
        return "Invalid age provided."

@app.route('/')
def home():
    global first_upload
    first_upload = True  # Reset for each new session
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global first_upload

    data = request.get_json()
    if data and 'image' in data:
        image_data = data['image'].split(",")[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
        image.save(file_path)

        if first_upload:
            # First upload: Provide age, gender, and banking service
            age = predict.age(file_path)
            gender = predict.gender(file_path)
            exp = Emotion.predict()
            
            output = f"The predicted age is {age} and the predicted gender is {gender}. Your expression is {exp}."
            service = banking_service(age)
            output += f" {service}"
            
            # Re-initialize the pyttsx3 engine each time
            engine = pyttsx3.init()
            engine.say(output)
            engine.runAndWait()
            
            first_upload = False  # Update flag to indicate first upload is complete
            
            return jsonify({'age': age, 'gender': gender, 'service': service})
        else:
            # Subsequent uploads: Provide only emotion-based message
            exp = Emotion.predict()
            if exp == 'happy':
                message = "You seem to be happy with my recommendations! Do check out our website for more services!!"
            elif exp == 'sad':
                message = "You seem to be sad! In case you want to learn more about us, you can ask me!"
            else:
                message = "Thanks for using our service. Let me know if you need anything else!"
            
            # Re-initialize the pyttsx3 engine each time
            engine = pyttsx3.init()
            engine.say(message)
            engine.runAndWait()
            
            return jsonify({'message': message})
    else:
        return jsonify({'error': 'No image provided'}), 400

@app.route('/voice', methods=['POST'])
def voice_interaction():
    text = request.json.get('text', '')
    if "hello" in text.lower():
        response = "Hello! How can I assist you today?"
    elif "banking services" in text.lower():
        response = "Please capture the image."
    elif "how do i do that" in text.lower():
        response = "You can check out our website or visit our nearest branch for more details."
    else:
        response = "I didn't understand that. Please try again."

    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()
    return jsonify({'message': response})

if __name__ == '__main__':
    app.run(debug=False)
