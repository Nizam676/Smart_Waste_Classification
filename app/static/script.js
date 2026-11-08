
const imageInput = document.getElementById("imageInput");
const uploadArea = document.getElementById("uploadArea");
const uploadContent = document.getElementById("uploadContent");
const previewContainer = document.getElementById("previewContainer");
const imagePreview = document.getElementById("imagePreview");
const previewName = document.getElementById("previewName");
const browseButton = document.getElementById("browseButton");
const removeImageButton = document.getElementById("removeImage");
const classifyButton = document.getElementById("classifyButton");
const buttonText = document.getElementById("buttonText");
const buttonArrow = document.getElementById("buttonArrow");
const uploadMessage = document.getElementById("uploadMessage");

const emptyResult = document.getElementById("emptyResult");
const resultContent = document.getElementById("resultContent");
const resultStatus = document.getElementById("resultStatus");
const predictedCategory = document.getElementById("predictedCategory");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");
const predictionBars = document.getElementById("predictionBars");

const recommendationCard = document.getElementById("recommendationCard");
const recommendationTitle = document.getElementById("recommendationTitle");
const recommendationBin = document.getElementById("recommendationBin");
const recommendationInstructions = document.getElementById("recommendationInstructions");
const recommendationBenefits = document.getElementById("recommendationBenefits");

const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];

let selectedFile = null;
let previewUrl = null;
let isPredicting = false;


// Open the file picker.
browseButton.addEventListener("click", () => {
    imageInput.click();
});


// Handle file selection.
imageInput.addEventListener("change", (event) => {
    const file = event.target.files[0];

    if (file) {
        handleSelectedFile(file);
    }
});


// Handle drag and drop.
uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("drag-over");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("drag-over");
});

uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("drag-over");

    const file = event.dataTransfer.files[0];

    if (file) {
        handleSelectedFile(file);
    }
});


// Validate and display the selected image.
function handleSelectedFile(file) {
    clearMessage();

    if (!ALLOWED_TYPES.includes(file.type)) {
        showMessage("Please select a JPG, PNG or WEBP image.");
        return;
    }

    if (file.size > MAX_FILE_SIZE) {
        showMessage("Image size must be below 5 MB.");
        return;
    }

    selectedFile = file;

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }

    previewUrl = URL.createObjectURL(file);

    imagePreview.src = previewUrl;
    previewName.textContent = file.name;

    uploadContent.hidden = true;
    previewContainer.hidden = false;
    classifyButton.disabled = false;

    resetPreviousResult();
}


// Remove the selected image.
removeImageButton.addEventListener("click", () => {
    clearSelectedImage();
});

function clearSelectedImage() {
    selectedFile = null;
    imageInput.value = "";

    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
        previewUrl = null;
    }

    imagePreview.removeAttribute("src");
    previewName.textContent = "";

    previewContainer.hidden = true;
    uploadContent.hidden = false;
    classifyButton.disabled = true;

    clearMessage();
    resetPreviousResult();
}


// Reset old prediction and recommendation.
function resetPreviousResult() {
    emptyResult.hidden = false;
    resultContent.hidden = true;
    recommendationCard.hidden = true;

    resultStatus.textContent = "Waiting";
    resultStatus.classList.remove("complete");

    predictedCategory.textContent = "—";
    confidenceValue.textContent = "0%";
    confidenceFill.style.width = "0%";
    predictionBars.replaceChildren();
}


// Display an error message.
function showMessage(message) {
    uploadMessage.textContent = message;
}

function clearMessage() {
    uploadMessage.textContent = "";
}


// Convert a recommendation field into a list.
function addListItems(container, items) {
    container.replaceChildren();

    const values = Array.isArray(items) ? items : [items];

    values.forEach((item) => {
        if (item === null || item === undefined || item === "") {
            return;
        }

        const li = document.createElement("li");
        li.textContent = String(item);
        container.appendChild(li);
    });
}


// Display prediction probabilities.
function displayPredictionBars(predictions) {
    predictionBars.replaceChildren();

    const entries = Object.entries(predictions || {})
        .sort((a, b) => Number(b[1]) - Number(a[1]));

    entries.forEach(([category, probability]) => {
        const value = Math.max(0, Math.min(100, Number(probability) || 0));

        const row = document.createElement("div");
        row.className = "prediction-row";

        const name = document.createElement("span");
        name.className = "prediction-name";
        name.textContent = category;

        const track = document.createElement("div");
        track.className = "prediction-track";

        const fill = document.createElement("div");
        fill.className = "prediction-fill";
        fill.style.width = `${value}%`;

        track.appendChild(fill);

        const percent = document.createElement("span");
        percent.className = "prediction-percent";
        percent.textContent = `${value.toFixed(1)}%`;

        row.append(name, track, percent);
        predictionBars.appendChild(row);
    });
}


// Display the classification result.
function displayPrediction(data) {
    const prediction = data.prediction;
    const recommendation = data.recommendation;

    if (!prediction || !prediction.category) {
        throw new Error("The server returned an incomplete prediction.");
    }

    emptyResult.hidden = true;
    resultContent.hidden = false;

    resultStatus.textContent = "Completed";
    resultStatus.classList.add("complete");

    predictedCategory.textContent = prediction.category;

    // Confidence may be returned as a decimal (0–1) or percentage (0–100).
    let confidence = Number(prediction.confidence) || 0;

    if (confidence <= 1) {
        confidence *= 100;
    }

    confidence = Math.max(0, Math.min(100, confidence));

    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceFill.style.width = `${confidence}%`;

    displayPredictionBars(prediction.all_predictions);

    if (recommendation) {
        recommendationTitle.textContent =
            recommendation.title || "Recycling recommendation";

        recommendationBin.textContent =
            recommendation.bin || "Follow local disposal guidance";

        addListItems(
            recommendationInstructions,
            recommendation.instructions || []
        );

        addListItems(
            recommendationBenefits,
            recommendation.benefits || []
        );

        recommendationCard.hidden = false;
    }
}


// Send the image to the Flask backend.
classifyButton.addEventListener("click", async () => {
    if (!selectedFile || isPredicting) {
        return;
    }

    isPredicting = true;
    classifyButton.disabled = true;

    buttonText.textContent = "Analyzing image...";
    buttonArrow.textContent = "⏳";

    clearMessage();
    resetPreviousResult();

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.message || "Unable to classify this image.");
        }

        displayPrediction(data);

        uploadMessage.style.color = "#24734a";
        showMessage("Image classified successfully.");
    } catch (error) {
        console.error("Prediction error:", error);

        uploadMessage.style.color = "#b34d42";
        showMessage(error.message || "Something went wrong. Please try again.");
    } finally {
        isPredicting = false;
        classifyButton.disabled = !selectedFile;

        buttonText.textContent = "Classify waste";
        buttonArrow.textContent = "→";
    }
});


// Release the preview URL when the page is closed.
window.addEventListener("beforeunload", () => {
    if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
    }
});
