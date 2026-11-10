
let confidenceChart = null;

let analysisHistory = [];


// =========================================
// ANALYZE FEELINGS
// =========================================

async function analyzeFeeling() {

    const userText = document.getElementById("userText").value.trim();

    const analyzeBtn = document.getElementById("analyzeBtn");

    const loading = document.getElementById("loading");

    const resultSection = document.getElementById("resultSection");

    const errorMessage = document.getElementById("errorMessage");

    const safetyMessage = document.getElementById("safetyMessage");


    if (!userText) {

        errorMessage.textContent =
            "⚠️ Please write something about how you are feeling.";

        errorMessage.classList.remove("hidden");

        resultSection.classList.add("hidden");

        return;
    }


    errorMessage.classList.add("hidden");

    resultSection.classList.add("hidden");

    safetyMessage.classList.add("hidden");

    loading.classList.remove("hidden");

    analyzeBtn.disabled = true;

    analyzeBtn.textContent = "Analyzing...";


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: userText
            })

        });


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Prediction failed."
            );

        }


        // RECEIVE FLASK RESPONSE

        const emotion = data.result;

        const confidence = Number(data.confidence);

        const message = data.message;

        const severity = data.severity;

        const distribution = data.distribution || {};


        // DISPLAY RESULT

        document.getElementById("emotion").textContent = emotion;

        document.getElementById("confidence").textContent =
            confidence.toFixed(2) + "%";

        document.getElementById("supportMessage").textContent =
            "🌙 " + message;

        document.getElementById("severity").textContent =
            "⚡ Severity: " + severity;


        // SUICIDAL SAFETY MESSAGE

        if (emotion.toLowerCase().includes("suicidal")) {

            safetyMessage.classList.remove("hidden");

        }


        // SAVE HISTORY

        analysisHistory.unshift({
            emotion: emotion,
            confidence: confidence
        });

        analysisHistory = analysisHistory.slice(0, 5);

        updateHistory();


        // DISPLAY RESULT

        resultSection.classList.remove("hidden");


        // DRAW GRAPH

        createConfidenceChart(distribution);


        resultSection.scrollIntoView({
            behavior: "smooth",
            block: "start"
        });


    } catch (error) {

        errorMessage.textContent = "⚠️ " + error.message;

        errorMessage.classList.remove("hidden");

    } finally {

        loading.classList.add("hidden");

        analyzeBtn.disabled = false;

        analyzeBtn.textContent = "Analyze 🚀";

    }

}


// =========================================
// CONFIDENCE GRAPH
// =========================================

function createConfidenceChart(distribution) {

    const canvas = document.getElementById("confidenceChart");

    const ctx = canvas.getContext("2d");

    const labels = Object.keys(distribution);

    const values = Object.values(distribution).map(Number);


    if (confidenceChart !== null) {

        confidenceChart.destroy();

    }


    confidenceChart = new Chart(ctx, {

        type: "bar",

        data: {

            labels: labels,

            datasets: [{

                label: "ML Prediction Probabilities",

                data: values,

                backgroundColor: "#247bb3",

                borderColor: "#185c88",

                borderWidth: 1,

                barPercentage: 0.65,

                categoryPercentage: 0.7

            }]

        },

        options: {

            responsive: true,

            maintainAspectRatio: false,

            plugins: {

                legend: {
                    display: false
                },

                tooltip: {

                    callbacks: {

                        label: function(context) {

                            return context.parsed.y.toFixed(2) + "%";

                        }

                    }

                }

            },

            scales: {

                y: {

                    beginAtZero: true,

                    max: 100,

                    title: {

                        display: true,

                        text: "Confidence (%)",

                        font: {
                            size: 12,
                            weight: "bold"
                        }

                    },

                    ticks: {
                        font: {
                            size: 11
                        }
                    }

                },

                x: {

                    ticks: {

                        font: {
                            size: 11
                        },

                        maxRotation: 25,

                        minRotation: 0

                    }

                }

            }

        }

    });

}


// =========================================
// ANALYSIS HISTORY
// =========================================

function updateHistory() {

    const historyList = document.getElementById("historyList");

    historyList.innerHTML = "";


    analysisHistory.forEach((item, index) => {

        const li = document.createElement("li");

        li.textContent =
            `Analysis ${index + 1}: ${item.emotion} (${item.confidence.toFixed(2)}%)`;

        historyList.appendChild(li);

    });

}
