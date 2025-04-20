function showVideo(videoSrc, exerciseName) {
    document.getElementById("homePage").classList.remove("active");
    document.getElementById("videoPage").classList.add("active");
    document.getElementById("exerciseVideo").src = `/static/${videoSrc}`;
    document.getElementById("exerciseTitle").innerText = exerciseName;
}

function showCameraAccess() {
    document.getElementById("videoPage").classList.remove("active");
    document.getElementById("cameraPage").classList.add("active");
}

function startExercise() {
    document.getElementById("cameraPage").classList.remove("active");
    document.getElementById("aiPage").classList.add("active");

    const exercise = document.getElementById("exerciseTitle").innerText.toLowerCase().replace(" ", "_");
    document.getElementById("aiFeed").src = `http://localhost:5000/video_feed/${exercise}`;

    fetch('http://localhost:5000/start_exercise', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ exercise: exercise })
    });
}

function goHome() {
    document.getElementById("videoPage").classList.remove("active");
    document.getElementById("cameraPage").classList.remove("active");
    document.getElementById("aiPage").classList.remove("active");
    document.getElementById("homePage").classList.add("active");

    document.getElementById("aiFeed").src = "";
}

function filterExercises() {
    let searchValue = document.getElementById("searchBox").value.toLowerCase();
    let buttons = document.getElementsByClassName("exercise-btn");

    for (let i = 0; i < buttons.length; i++) {
        let text = buttons[i].innerText.toLowerCase();
        buttons[i].style.display = text.includes(searchValue) ? "block" : "none";
    }
}