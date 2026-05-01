require("dotenv").config();

const express = require("express");
const mongoose = require("mongoose");
const multer = require("multer");
const cors = require("cors");

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
.then(()=>console.log("MongoDB Connected"))
.catch(err=>console.error(err));

const File = mongoose.model("File", {
    name: String,
    data: Buffer,
    contentType: String
});

const Task = mongoose.model("Task", {
    title: String,
    start: Date, 
    end: Date,  
    priority: { type: Number, default: 0 },
    status: { type: String, default: "pending" },
    emotion_at_creation: String
});

const Chat = mongoose.model("Chat", {
    sender: String,
    text: String,
    timestamp: Date
});

const storage = multer.memoryStorage();
const upload = multer({ storage });

app.post("/upload", upload.single("file"), async (req, res) => {

    const allowedTypes = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.ms-powerpoint",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "text/plain",
        "text/csv",
        "application/json",
        "image/jpeg",
        "image/png",
        "image/webp"
    ];

    if (!req.file) {
        return res.status(400).send("No file uploaded");
    }

    if (!allowedTypes.includes(req.file.mimetype)) {
        return res.status(400).send("File type not allowed");
    }

    await File.create({
        name: req.file.originalname,
        data: req.file.buffer,
        contentType: req.file.mimetype
    });

    res.sendStatus(200);
});

app.get("/files", async (req, res) => {
    const files = await File.find({}, { data: 0 });
    res.json(files);
});

app.get("/file/:id", async (req, res) => {
    const file = await File.findById(req.params.id);
    if (!file) return res.status(404).send("File not found");

    res.set("Content-Type", file.contentType);
    res.send(file.data);
});

app.delete("/delete/:id", async (req, res) => {
    await File.findByIdAndDelete(req.params.id);
    res.sendStatus(200);
});

app.post("/add_task", async (req, res) => {
    const { title, start, end } = req.body;
    const task = await Task.create({ title, start, end });
    res.json(task);
});

app.get("/get_tasks", async (req, res) => {
    const tasks = await Task.find();
    res.json(tasks);
});

app.put("/update_task/:id", async (req, res) => {
    try {
        const updatedTask = await Task.findByIdAndUpdate(req.params.id, req.body, { new: true });
        res.json(updatedTask);
    } catch (err) {
        res.status(500).send("Dynamic update failed");
    }
});

app.delete("/delete_task/:id", async (req, res) => {
    await Task.findByIdAndDelete(req.params.id);
    res.sendStatus(200);
});

app.post("/save_chat", async (req, res) => {
    const { sender, text } = req.body;
    await Chat.create({ sender, text, timestamp: new Date() });
    res.sendStatus(200);
});

app.post("/clear_chat", async (req, res) => {
    await Chat.deleteMany({});
    res.sendStatus(200);
});

app.post("/update_current_emotion", async (req, res) => {
    const { emotion } = req.body;
    await Chat.create({ sender: "System", text: `Detected: ${emotion}`, timestamp: new Date() });
    res.sendStatus(200);
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});