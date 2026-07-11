const form = document.getElementById("emotionForm");
const button = document.getElementById("predictButton");

form.addEventListener("submit", function () {
    button.innerText = "Predicting...";
    button.disabled = true;
});
