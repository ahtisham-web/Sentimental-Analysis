const colors = { positive: "#34d399", neutral: "#94a3b8", negative: "#f87171" };
const emojis = { positive: "😊", neutral: "😐", negative: "😞" };
const SEGMENTS = 24;

// ---------- Model accuracy badge ----------
async function loadModelInfo() {
  const badgeText = document.getElementById("modelBadgeText");
  try {
    const res = await fetch("/api/model-info");
    const data = await res.json();
    if (data.accuracy != null) {
      badgeText.textContent = `${data.model_name} · ${data.accuracy}% accuracy`;
    } else {
      badgeText.textContent = data.model_name || "model ready";
    }
  } catch (err) {
    badgeText.textContent = "model info unavailable";
  }
}
loadModelInfo();

// ---------- Segmented level-meter bar builder ----------
function buildMeterRow(key, value) {
  const filledCount = Math.round((value / 100) * SEGMENTS);
  let segsHTML = "";
  for (let i = 0; i < SEGMENTS; i++) {
    const isFilled = i < filledCount;
    segsHTML += `<div class="meter-seg${isFilled ? " filled" : ""}" style="--seg-color:${colors[key]}"></div>`;
  }
  return `
    <div class="meter-row">
      <div class="meter-tag">
        <span class="meter-dot" style="background:${colors[key]}"></span>
        ${key}
      </div>
      <div class="meter-track">${segsHTML}</div>
      <div class="meter-pct">${value.toFixed(1)}%</div>
    </div>
  `;
}

function renderMeterGroup(containerId, probabilities, isPercentAlready) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const order = ["positive", "neutral", "negative"];
  order.forEach((key) => {
    const raw = probabilities[key] ?? 0;
    const pct = isPercentAlready ? raw : raw * 100;
    container.innerHTML += buildMeterRow(key, pct);
  });
}

// ---------- Text sentiment ----------
const textarea = document.getElementById("inputText");
const btn = document.getElementById("analyzeBtn");
const btnText = document.getElementById("btnText");
const spinner = document.getElementById("spinner");
const resultBox = document.getElementById("resultBox");
const errorBox = document.getElementById("errorBox");

async function analyze() {
  const text = textarea.value.trim();
  errorBox.style.display = "none";
  if (!text) {
    errorBox.textContent = "Please enter some text first.";
    errorBox.style.display = "block";
    return;
  }

  setLoading(true);

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    if (!res.ok) throw new Error("Request failed");
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    errorBox.textContent = "Something went wrong: " + err.message;
    errorBox.style.display = "block";
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  btn.disabled = isLoading;
  btnText.textContent = isLoading ? "Analyzing..." : "Analyze Sentiment";
  spinner.hidden = !isLoading;
}

function renderResult(data) {
  document.getElementById("resultEmoji").textContent = data.emoji;
  document.getElementById("resultLabel").textContent = data.label;
  document.getElementById("resultLabel").style.color = colors[data.label] || "#fff";
  document.getElementById("resultConfidence").textContent =
    "confidence " + Math.round(data.confidence * 100) + "%";

  renderMeterGroup("bars", data.probabilities, false);
  resultBox.style.display = "block";
}

btn.addEventListener("click", analyze);
textarea.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) analyze();
});

// ---------- Tab switching ----------
function switchTab(tab) {
  const isText = tab === "text";
  document.getElementById("textCard").classList.toggle("hidden", !isText);
  document.getElementById("youtubeCard").classList.toggle("hidden", isText);
  document.getElementById("tabText").classList.toggle("active", isText);
  document.getElementById("tabYoutube").classList.toggle("active", !isText);
}

// ---------- YouTube comment analysis ----------
const ytUrl = document.getElementById("ytUrl");
const ytBtn = document.getElementById("ytAnalyzeBtn");
const ytBtnText = document.getElementById("ytBtnText");
const ytSpinner = document.getElementById("ytSpinner");
const ytResultBox = document.getElementById("ytResultBox");
const ytErrorBox = document.getElementById("ytErrorBox");

async function analyzeYoutube() {
  const url = ytUrl.value.trim();
  ytErrorBox.style.display = "none";
  if (!url) {
    ytErrorBox.textContent = "Please paste a YouTube video URL first.";
    ytErrorBox.style.display = "block";
    return;
  }

  ytBtn.disabled = true;
  ytBtnText.textContent = "Analyzing...";
  ytSpinner.hidden = false;
  ytResultBox.style.display = "none";

  try {
    const res = await fetch("/api/youtube/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_url: url, max_comments: 50 }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Request failed");
    renderYoutubeResult(data);
  } catch (err) {
    ytErrorBox.textContent = err.message;
    ytErrorBox.style.display = "block";
  } finally {
    ytBtn.disabled = false;
    ytBtnText.textContent = "Analyze Comments";
    ytSpinner.hidden = true;
  }
}

function renderYoutubeResult(data) {
  document.getElementById("ytSummary").innerHTML = `
    <span class="yt-count">${data.total_comments_analyzed} comments analyzed</span>
  `;

  renderMeterGroup("ytBars", data.sentiment_summary, true);

  const commentsContainer = document.getElementById("ytComments");
  commentsContainer.innerHTML = "";
  data.comments.forEach((c) => {
    const item = document.createElement("div");
    item.className = "comment-item";
    item.style.setProperty("--seg-color", colors[c.label]);
    item.innerHTML = `
      <div class="comment-author">
        <span>${c.author}</span>
        <span class="comment-tag">${c.emoji} ${c.label}</span>
      </div>
      <div class="comment-text">${c.text}</div>
    `;
    commentsContainer.appendChild(item);
  });

  ytResultBox.style.display = "block";
}

ytBtn.addEventListener("click", analyzeYoutube);
