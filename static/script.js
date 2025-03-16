document.getElementById("upload-form").addEventListener("submit", async function(event) {
    event.preventDefault();

    let formData = new FormData();
    let imageInput = document.getElementById("image-input").files[0];
    
    if (!imageInput) {
        alert("Please select an image first.");
        return;
    }

    formData.append("image", imageInput);

    let response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let result = await response.json();

    if (result.error) {
        alert(result.error);
        return;
    }

    document.getElementById("prediction-text").innerText = "Prediction: " + result.prediction;
    document.getElementById("confidence-text").innerText = "Confidence: " + result.confidence;
    document.getElementById("result").classList.remove("hidden");
});
