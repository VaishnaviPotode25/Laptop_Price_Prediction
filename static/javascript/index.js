// ==============================
// RUN AFTER DOM LOAD
// ==============================
document.addEventListener("DOMContentLoaded", function () {

    // ==============================
    // Smooth scroll to result
    // ==============================
    document.getElementById("predictForm").addEventListener("submit", function () {
        setTimeout(() => {
            const result = document.querySelector(".result");
            if (result) {
                result.scrollIntoView({ behavior: "smooth" });
            }
        }, 500);
    });

    // ==============================
    // Input focus glow
    // ==============================
    const inputs = document.querySelectorAll("input");

    inputs.forEach(input => {
        input.addEventListener("focus", () => {
            input.style.boxShadow = "0 0 10px #00aaff";
        });

        input.addEventListener("blur", () => {
            input.style.boxShadow = "none";
        });
    });

    // ==============================
    // Clear all fields
    // ==============================
    document.getElementById("clearBtn").addEventListener("click", function () {

        const confirmClear = confirm("Are you sure you want to clear all fields?");

        if (confirmClear) {
            inputs.forEach(input => input.value = "");

            const result = document.querySelector(".result");
            if (result) result.innerText = "";
        }
    });

    // ==============================
    // LIST HIGHLIGHT ROTATION ✅
    // ==============================
    const items = document.querySelectorAll(".carousel-list li");

    console.log("Items found:", items.length); // should be 5

    let currentIndex = 0;

    function rotateList() {
        items.forEach(item => item.classList.remove("active"));
        items[currentIndex].classList.add("active");
        currentIndex = (currentIndex + 1) % items.length;
    }

    // start rotation
    setInterval(rotateList, 2000);
});


// ==============================
// ANIMATED PRICE + CONFIDENCE
// ==============================

function animateResult(finalPrice, confidence) {

    let current = 0;
    const increment = finalPrice / 50;  // speed control

    const priceElement = document.getElementById("priceText");
    const confidenceElement = document.getElementById("confidenceText");

    const counter = setInterval(() => {
        current += increment;

        if (current >= finalPrice) {
            current = finalPrice;
            clearInterval(counter);
        }

        priceElement.innerText = "₹ " + Math.floor(current);
    }, 20);

    // Confidence display
    confidenceElement.innerText = `Model Confidence: ${confidence}%`;
}