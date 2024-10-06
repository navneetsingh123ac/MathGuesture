import cvzone
import numpy as np
import cv2
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai
from PIL import Image
import streamlit as st

st.set_page_config(layout="wide")
math_img = Image.open('math.jpg')
resized_math_img = math_img.resize((1200, 200))
st.image(resized_math_img)

col1, col2 = st.columns([2, 1])
with col1:
    run = st.checkbox('Run', value=True)
    FRAME_WINDOW = st.image([])
with col2:
    st.title("Answer")
    output_text_area = st.subheader("")

genai.configure(api_key="AIzaSyBYAv4AYm1ArHiVdlJCO7QAHR9XRuu0hRg")
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1200)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)


def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList1 = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers1 = detector.fingersUp(hand1)
        print(fingers1)
        return fingers1, lmList1
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:  # Index finger is up

        current_pos = lmList[8][0:2]  # Get the position of the tip of the index finger
        if prev_pos is None:
            prev_pos = current_pos
        else:
            cv2.line(canvas, prev_pos, current_pos, (255, 0, 255),
                     10)  # Draw line between previous and current positions
        prev_pos = current_pos  # Update previous position

    elif fingers == [0, 0, 0, 0, 1]:
        canvas = np.zeros_like(img)

    else:
        prev_pos = None
    return prev_pos, canvas

def SendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math Problem", pil_image])
        return response.text

prev_pos = None
canvas = None
output_text = ""
# Continuously get frames from the webcam
while run:

    success, img = cap.read()
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)
    if info:
        fingers, lmList1 = info
        prev_pos, canvas = draw(info, prev_pos, canvas)  # Update both prev_pos and canvas
        output_text = SendToAI(model, canvas, fingers)

    image_combine = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    FRAME_WINDOW.image(image_combine,channels="BGR")
    output_text_area.text(output_text)

    cv2.waitKey(1)
