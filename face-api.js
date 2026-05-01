video.addEventListener('play', () => {
    setInterval(async () => {
        const detections = await faceapi.detectAllFaces(video, new faceapi.TinyFaceDetectorOptions())
            .withFaceLandmarks()
            .withFaceExpressions();

        if (detections.length > 0) {
            const expressions = detections[0].expressions;
            // Get the emotion with the highest confidence score
            const dominantEmotion = Object.keys(expressions).reduce((a, b) => expressions[a] > expressions[b] ? a : b);
            
            // Update UI and send to backend
            updateEmotionUI(dominantEmotion);
            saveEmotionToDB(dominantEmotion);
        }
    }, 1000); // Detect every 1 second
});