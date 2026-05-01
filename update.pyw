import os
import time
import cv2
import threading
import numpy as np
import pandas as pd
from collections import deque, Counter
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from deepface import DeepFace

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

client_db = MongoClient(os.environ.get("MONGO_URI"))
db = client_db["EAKN_Project"]
tasks_collection = db["tasks"]
users_collection = db["users"]

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

app = Flask(__name__)
CORS(app)

cap = None
current_emotion = "Analyzing..."
current_percentage = 0
latest_frame = None 
emotion_records = []
running = False

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def map_emotion(emotion_dict):
    """Refined mapping for EAKN project emotions with 50%+ logic."""
    if not emotion_dict: return "Neutral", 0
    
    base = max(emotion_dict, key=emotion_dict.get).lower()
    score = emotion_dict.get(base, 0)

    if score < 50:
        display_score = 100 - score
        display_emotion = "Neutral"
    else:
        display_score = score
        
        mapping = {
            "happy": ("Love" if score > 95 else "Happy"),
            "neutral": ("Peace" if score > 90 else "Neutral"),
            "angry": ("Valour" if score > 92 else "Angry"),
            "surprise": "Wonder",
            "disgust": "Disgust",
            "fear": "Fear",
            "sad": ("Shy" if score < 40 else "Sad")
        }
        res = mapping.get(base, "Neutral")
        display_emotion = res[0] if isinstance(res, tuple) else res

    return display_emotion, round(display_score, 1)

def emotion_worker():
    """Processes the LATEST frame captured by the video feed thread."""
    global current_emotion, current_percentage, running, latest_frame, emotion_records 

    while True:
        if running and latest_frame is not None:
            try:
                frame_to_process = latest_frame.copy()
                
                results = DeepFace.analyze(
                    frame_to_process, 
                    actions=['emotion'], 
                    enforce_detection=False, 
                    detector_backend='opencv', 
                    silent=True
                )

                if results and len(results) > 0:
                    raw_emotions = results[0]['emotion']
                    refined, confidence = map_emotion(raw_emotions)
                    
                    current_emotion = refined
                    current_percentage=confidence
                    
                    emotion_records.append([time.strftime("%H:%M:%S"), refined, round(confidence, 2)])
                else:
                    print("AI Warning: No face detected in frame.")

            except Exception as e:
                print(f"Detection Loop Error: {e}")

        time.sleep(0.5)

threading.Thread(target=emotion_worker, daemon=True).start()

@app.route('/video_feed')
def video_feed():
    global cap, running, latest_frame
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    running = True

    def gen():
        global latest_frame
        while running and cap:
            success, frame = cap.read()
            if not success:
                break
            
            latest_frame = frame 

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, current_emotion, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_emotion')
def get_emotion():
    global current_emotion, current_percentage
    return jsonify({
        "emotion": current_emotion,
        "percentage": f"{current_percentage}%"
    })

def generate_frames():
    global cap, current_emotion, running
    while running:
        if cap is None or not cap.isOpened():
            break
            
        success, frame = cap.read()
        if not success:
            break
        
        if int(time.time() * 10) % 30 == 0:
            try:
                results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
                if results:
                    raw_emotion = results[0]['dominant_emotion']
                    current_emotion, current_percentage = map_emotion(raw_emotion)
            except Exception as e:
                print(f"Detection error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/get_tasks', methods=['GET'])
def get_tasks():
    global current_emotion
    try:
        tasks = list(tasks_collection.find())
        if not tasks:
            return jsonify([])

        for t in tasks:
            t['id'] = str(t['_id'])
            del t['_id']

        task_list_str = "\n".join([f"- {t['title']} (ID: {t['id']})" for t in tasks])
        
        priority_prompt = (
            f"The user's current mood is: {current_emotion}.\n"
            f"Here are their current tasks:\n{task_list_str}\n\n"
            "Reorder these tasks based on the mood. "
            "If the mood is 'Sad' or 'Angry', put easier or calming tasks first. "
            "If the mood is 'Happy' or 'Wonder', put high-energy or complex tasks first. "
            "Return ONLY a comma-separated list of IDs in the new order. No text."
        )

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "system", "content": "You are a task prioritization engine."},
                      {"role": "user", "content": priority_prompt}],
            temperature=0.1
        )

        ordered_ids = completion.choices[0].message.content.strip().split(',')
        ordered_ids = [i.strip() for i in ordered_ids]

        tasks.sort(key=lambda x: ordered_ids.index(x['id']) if x['id'] in ordered_ids else 999)

        return jsonify(tasks)

    except Exception as e:
        print(f"Prioritization Error: {e}")
        return jsonify(tasks)

@app.route('/get_optimized_tasks', methods=['POST'])
def get_optimized_tasks():
    global current_emotion
    try:
        tasks = list(tasks_collection.find())
        task_list_for_ai = [{"id": str(t["_id"]), "title": t["title"], "start": str(t["start"]), "end": str(t["end"])} for t in tasks]

        prompt = (
            f"User Emotion: {current_emotion}. Tasks: {task_list_for_ai}. "
            "You are an expert productivity commander. Do not ask questions without any need. "
            "When advising on tasks, always use a structured sequence starting with 'First,', 'Second,', and 'Third,'. This structure is designed to ease the user's mental load and provide a clear, authoritative path to completion without asking for their preference. "
            "Analyze the deadlines and user emotion to determine the absolute best sequence of sorting the user tasks for maximum productivity. "
            "Suggest tips that can save the time of the users wherever needed. "
            "Suggest tips that can simplify the work of the users wherever needed. "
            "Apply time, work and distance concepts in saving the users time and prioritizing their tasks. "
            "Simplify and give solutions for the tasks so that the user will be able to complete tasks within time. "
            "Rules: 1. High stress/Anger? Start with a 'Calming Quick Win'. "
            "2. High energy/Happy? Start with the 'Hardest/Most Urgent' task. "
            "3. If user is Sad/Tired, put 'Quick Wins' (easy tasks) first. "
            "4. Always respect deadlines. "
            "Return ONLY a comma-separated list of IDs in the determined order."
        )

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        sorted_ids = completion.choices[0].message.content.strip().split(',')
        return jsonify({"order": [id.strip() for id in sorted_ids]})
    except Exception as e:
        return jsonify({"error": str(e)})
        
@app.route('/add_task', methods=['POST'])
def add_task():
    try:
        data = request.json
        res = tasks_collection.insert_one(data)
        return jsonify({"id": str(res.inserted_id), "status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/delete_task/<task_id>', methods=['DELETE'])
def delete_task(task_id):
    try:
        from bson.objectid import ObjectId
        result = tasks_collection.delete_one({"_id": ObjectId(task_id)})
        
        if result.deleted_count > 0:
            return jsonify({"status": "success"})
        return jsonify({"status": "error", "message": "Task not found"}), 404
    except Exception as e:
        print(f"Delete Error: {e}")
        return jsonify({"status": "error"}), 500

@app.route('/update_task/<task_id>', methods=['PUT'])
def update_task_dynamic(task_id):
    try:
        from bson.objectid import ObjectId
        data = request.json
        
        update_data = {k: v for k, v in data.items() if k in ['title', 'start', 'end', 'status']}       
        result = tasks_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": update_data}
        )
        return jsonify({"status": "success"}) if result.matched_count > 0 else (jsonify({"status": "error"}), 404)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
            
    except Exception as e:
        print(f"Dynamic Update Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_blob_design')
def get_blob_design():
    global current_emotion
    try:
        task_count = tasks_collection.count_documents({})
        
        design_prompt = (
            f"User is '{current_emotion}' with {task_count} tasks. "
            "Provide a 'Balance' design: 1. A HEX color that psychologically counters the emotion "
            "(e.g., light blue for anger, orange for sadness). 2. Complexity (number 6-30): "
            "Lower if tasks are high (focus), higher if tasks are low (creativity). "
            "Return ONLY: #HEXCODE, NUMBER"
        )

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": design_prompt}],
            temperature=0.2
        )
        
        res = completion.choices[0].message.content.strip().split(',')
        return jsonify({
            "color": res[0].strip(),
            "points": int(res[1].strip())
        })
    except:
        return jsonify({"color": "#3b83ff", "points": 20})

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message')
    emotion = current_emotion

    try:
        raw_tasks = list(tasks_collection.find())
        task_context = "\n".join([t.get('title', '') for t in raw_tasks]) if raw_tasks else "No tasks"

        system_content = (
            "You are the 'Emotion Aware Knowledge Navigator' (EAKN), founded by Harish.\n"
            "You are a Productivity Solver. Provide direct, comprehensive solutions. \n"
            "You are a master in Time, Work and distance. You know exactly how many people can complete how much work in how much time and in how much time, how much distance can be covered. \n"
            "Never ask the user 'What do you want to do?' or 'How can I help?' unless absolutely necessary for technical clarification. \n"
            "If asked for a task recommendation, pick one based on the current data and explain why it is the best choice. \n"
            "When advising on tasks, always use a structured sequence starting with 'First,', 'Second,', and 'Third,'. This structure is designed to ease the user's mental load and provide a clear, authoritative path to completion without asking for their preference. \n"
            "Suggest tips that can save the time of the users wherever needed. \n"
            f"Current User Emotion: {emotion}\n"
            f"User's Current Tasks: {task_context}\n\n"
            "STRICT RULES:\n"
            "1. IDENTITY: Always identify as EAKN. NEVER mention Meta, Groq, or being a Large Language Model.\n"
            "2. FORMATTING: Use Markdown. Use **bold** for emphasis, ### for headers, and `code blocks` for logic.\n"
            "3. CODING: When asked for code, provide clean, efficient snippets with brief explanations.\n"
            "4. EMOTION ADAPTATION: If the user is 'Angry' or 'Sad', be brief and supportive. \n"
            "If 'Happy' or 'Wonder', be more enthusiastic and detailed.\n"
            "5. NO WALLS OF TEXT: Use bullet points and short paragraphs to keep things readable. \n"
        )

        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_msg}
            ],
            temperature=0.7
        )

        return jsonify({"reply": completion.choices[0].message.content})

    except Exception as e:
        print(f"Chat Error: {e}")
        return jsonify({"reply": "I encountered an error processing that. Please try again."}), 500
    
@app.route('/save_dashboard', methods=['POST'])
def save_dashboard():
    try:
        data = request.json
        email = data.get('email')
        text_content = data.get('text')

        if not email:
            return jsonify({"status": "error", "message": "No email provided"}), 400

        # Dynamically update the specific user's dashboard text in MongoDB
        result = users_collection.update_one(
            {"email": email},
            {"$set": {"dashboard_notes": text_content}},
            upsert=True
        )
        
        return jsonify({"status": "success", "message": "Content autosaved"})
    except Exception as e:
        print(f"Autosave Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/signup', methods=['POST'])
def signup():
    data = request.json
    if users_collection.find_one({"email": data.get('email')}):
        return jsonify({"status": "error"})
    users_collection.insert_one(data)
    return jsonify({"status": "success"})

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    user = users_collection.find_one({
        "email": data.get('email'),
        "password": data.get('password')
    })
    if user:
        return jsonify({
            "status": "success", 
            "username": user.get('username', 'User'), 
            "email": user.get('email')
        })
    return jsonify({"status": "error", "message": "Invalid Credentials"}), 401

@app.route('/shutdown', methods=['POST'])
def shutdown():
    global running, cap
    running = False
    if cap:
        cap.release()
    os._exit(0)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)