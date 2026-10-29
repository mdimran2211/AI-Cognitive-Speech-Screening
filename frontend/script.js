// Dynamic API URL resolution (adapts to current port or URL params)
function getApiBase() {
    const params = new URLSearchParams(window.location.search);
    if (params.get("api")) return params.get("api").replace(/\/$/, "");
    if (params.get("port")) return `http://127.0.0.1:${params.get("port")}`;
    if (window.location.protocol.startsWith("http")) {
        if (window.location.port === "8000" || window.location.port === "8080" || !window.location.port) {
            return window.location.origin;
        }
    }
    return "http://127.0.0.1:8000";
}
const API_BASE = getApiBase();

// State
let currentFile = null;
let currentAnalysisData = null;
let isAnalyzing = false;
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let recordInterval = null;
let recordSeconds = 0;
let audioContext = null;
let analyserNode = null;
let micStream = null;
let speechRecognizer = null;
let batchFilesList = [];
let historyCache = [];

// DOM Elements
const audioFile = document.getElementById("audioFile");
const dropZone = document.getElementById("dropZone");
const selectedFile = document.getElementById("selectedFile");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const removeFile = document.getElementById("removeFile");
const qualityBadge = document.getElementById("qualityBadge");
const badgeTag = document.getElementById("badgeTag");
const badgeSnr = document.getElementById("badgeSnr");
const qualityMessage = document.getElementById("qualityMessage");
const qualitySuggestions = document.getElementById("qualitySuggestions");
const audioPlaybackPreview = document.getElementById("audioPlaybackPreview");
const mainAudioPlayer = document.getElementById("mainAudioPlayer");
const prepInfoText = document.getElementById("prepInfoText");
const analyzeButton = document.getElementById("analyzeButton");
const buttonText = document.getElementById("buttonText");
const buttonArrow = document.getElementById("buttonArrow");
const buttonIcon = document.getElementById("buttonIcon");
const formError = document.getElementById("formError");

// Results Elements
const emptyResults = document.getElementById("emptyResults");
const resultContent = document.getElementById("resultContent");
const predictionLabel = document.getElementById("predictionLabel");
const predictionDescription = document.getElementById("predictionDescription");
const resultMark = document.getElementById("resultMark");
const cnValue = document.getElementById("cnValue");
const adValue = document.getElementById("adValue");
const cnBar = document.getElementById("cnBar");
const adBar = document.getElementById("adBar");
const audioDuration = document.getElementById("audioDuration");
const speakingRatioText = document.getElementById("speakingRatioText");
const pitchVariationText = document.getElementById("pitchVariationText");
const pauseCountText = document.getElementById("pauseCountText");
const resultExplanation = document.getElementById("resultExplanation");
const rawPrediction = document.getElementById("rawPrediction");
const rawScores = document.getElementById("rawScores");
const recordIdText = document.getElementById("recordIdText");
const analyzeAnother = document.getElementById("analyzeAnother");
const downloadPdfBtn = document.getElementById("downloadPdfBtn");

// Notes & STT
const analysisNotesInput = document.getElementById("analysisNotesInput");
const saveNotesBtn = document.getElementById("saveNotesBtn");
const notesStatusText = document.getElementById("notesStatusText");
const triggerTranscribeBtn = document.getElementById("triggerTranscribeBtn");
const sttTranscriptText = document.getElementById("sttTranscriptText");
const sttConfidenceText = document.getElementById("sttConfidenceText");

// Canvases
const waveformCanvas = document.getElementById("waveformCanvas");
const pitchCanvas = document.getElementById("pitchCanvas");
const spectrogramCanvas = document.getElementById("spectrogramCanvas");
const timelineBar = document.getElementById("timelineBar");

// Topbar & Navigation
const systemStatusText = document.getElementById("systemStatusText");
const sidebarStatusDot = document.getElementById("sidebarStatusDot");
const breadcrumbPage = document.getElementById("breadcrumbPage");

// Mic Elements
const micVisualizerCanvas = document.getElementById("micVisualizerCanvas");
const recordTimer = document.getElementById("recordTimer");
const micStatusMsg = document.getElementById("micStatusMsg");
const startRecordBtn = document.getElementById("startRecordBtn");
const stopRecordBtn = document.getElementById("stopRecordBtn");
const reRecordBtn = document.getElementById("reRecordBtn");
const recordedPreview = document.getElementById("recordedPreview");
const recordedAudioPlayer = document.getElementById("recordedAudioPlayer");
const submitRecordingBtn = document.getElementById("submitRecordingBtn");
const liveSttContent = document.getElementById("liveSttContent");

// Batch Elements
const batchAudioFiles = document.getElementById("batchAudioFiles");
const selectedBatchList = document.getElementById("selectedBatchList");
const batchFilesUl = document.getElementById("batchFilesUl");
const runBatchBtn = document.getElementById("runBatchBtn");
const batchResultsPanel = document.getElementById("batchResultsPanel");
const batchTotalCount = document.getElementById("batchTotalCount");
const batchTypicalCount = document.getElementById("batchTypicalCount");
const batchChangeCount = document.getElementById("batchChangeCount");
const batchTableBody = document.getElementById("batchTableBody");
const exportBatchCsvBtn = document.getElementById("exportBatchCsvBtn");
let lastBatchData = null;

// History Elements
const historySearchInput = document.getElementById("historySearchInput");
const historySortSelect = document.getElementById("historySortSelect");
const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
const clearAllHistoryBtn = document.getElementById("clearAllHistoryBtn");
const historyTableBody = document.getElementById("historyTableBody");

// Compare Elements
const compareSelectA = document.getElementById("compareSelectA");
const compareSelectB = document.getElementById("compareSelectB");
const runCompareBtn = document.getElementById("runCompareBtn");
const compareResultsGrid = document.getElementById("compareResultsGrid");
const compareBodyA = document.getElementById("compareBodyA");
const compareBodyB = document.getElementById("compareBodyB");

// Settings Elements
const settingRetentionDays = document.getElementById("settingRetentionDays");
const settingSaveAudio = document.getElementById("settingSaveAudio");
const settingFontSize = document.getElementById("settingFontSize");
const saveSettingsBtn = document.getElementById("saveSettingsBtn");
const settingsStatusText = document.getElementById("settingsStatusText");
const settingsClearDataBtn = document.getElementById("settingsClearDataBtn");

// -------------------------------------------------------------
// 1. TAB NAVIGATION
// -------------------------------------------------------------
const navItems = document.querySelectorAll(".nav-item");
const tabPanes = document.querySelectorAll(".tab-pane");

const TAB_ALIASES = {
    "dashboard": "home",
    "analysis": "home",
    "check": "home",
    "overview": "home",
    "reports": "history",
    "report": "history",
    "batch-analysis": "batch",
    "evaluation": "model-eval",
    "eval": "model-eval",
    "metrics": "model-eval",
    "model": "model-eval",
    "datasets": "dataset",
    "status": "system",
    "diagnostics": "system",
    "faq": "help",
    "faqs": "help"
};

function resolveTab(rawTab) {
    if (!rawTab) return "home";
    const cleaned = String(rawTab).replace(/^#\/?/, "").trim().toLowerCase();
    const mapped = TAB_ALIASES[cleaned] || cleaned;
    if (document.getElementById(`pane-${mapped}`)) {
        return mapped;
    }
    return "home";
}

function switchTab(rawTabId, updateHash = true) {
    const tabId = resolveTab(rawTabId);

    tabPanes.forEach(pane => {
        const isMatch = pane.id === `pane-${tabId}`;
        pane.classList.toggle("hidden", !isMatch);
        pane.classList.toggle("active", isMatch);
    });

    navItems.forEach(item => {
        const itemTab = item.getAttribute("data-tab");
        item.classList.toggle("active", itemTab === tabId);
    });

    // Update breadcrumb
    const pageNames = {
        "home": "Check a Voice Recording",
        "record": "Microphone Check",
        "batch": "Check Multiple Files",
        "history": "Past Results",
        "compare": "Compare Results",
        "model-eval": "How Well AI Works",
        "dataset": "Explore Dataset",
        "settings": "Settings & Privacy",
        "system": "App Status",
        "help": "Help & FAQs"
    };
    const sectionNames = {
        "home": "Home",
        "record": "Screening",
        "batch": "Screening",
        "history": "Screening",
        "compare": "Screening",
        "model-eval": "Research & Tools",
        "dataset": "Research & Tools",
        "settings": "Preferences",
        "system": "Diagnostics",
        "help": "Support"
    };

    if (breadcrumbPage) breadcrumbPage.textContent = pageNames[tabId] || "Voice Screening";
    if (breadcrumbSection) breadcrumbSection.textContent = sectionNames[tabId] || "VoiceCheck";

    // Update URL hash without breaking history
    if (updateHash && window.location.hash !== `#${tabId}`) {
        if (window.history && window.history.pushState) {
            window.history.pushState(null, "", `#${tabId}`);
        } else {
            window.location.hash = `#${tabId}`;
        }
    }

    // Trigger tab-specific data loads
    if (tabId === "history") loadHistory();
    if (tabId === "compare") populateCompareSelectors();
    if (tabId === "model-eval") loadModelEval();
    if (tabId === "dataset") loadDatasetInfo();
    if (tabId === "settings") loadSettings();
    if (tabId === "system") checkSystemHealth();

    window.scrollTo({ top: 0, behavior: "smooth" });
}

navItems.forEach(item => {
    item.addEventListener("click", (e) => {
        e.preventDefault();
        const tabId = item.getAttribute("data-tab");
        switchTab(tabId);
    });
});

document.getElementById("brandLink")?.addEventListener("click", (e) => {
    e.preventDefault();
    switchTab("home");
});

window.addEventListener("hashchange", () => {
    const hash = window.location.hash.replace(/^#\/?/, "");
    if (hash) {
        switchTab(hash, false);
    }
});

// Run initial tab activation based on URL hash if present
const initialTabHash = window.location.hash ? window.location.hash.replace(/^#\/?/, "") : "";
if (initialTabHash) {
    switchTab(initialTabHash, false);
}

document.getElementById("chooseRecordingTopBtn")?.addEventListener("click", () => {
    switchTab("home");
    audioFile.click();
});
document.getElementById("recordMicTopBtn")?.addEventListener("click", () => switchTab("record"));
document.getElementById("getHelpTopBtn")?.addEventListener("click", () => {
    document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
});

// -------------------------------------------------------------
// 2. FILE HELPERS & QUALITY CHECK
// -------------------------------------------------------------
function showError(msg) {
    if (formError) {
        formError.textContent = msg;
        formError.classList.remove("hidden");
    }
}

function clearError() {
    if (formError) {
        formError.textContent = "";
        formError.classList.add("hidden");
    }
}

function formatFileSize(bytes) {
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}


// Reset results view so previous or stale results never linger
function resetResultsView() {
    currentAnalysisData = null;
    if (resultContent) resultContent.classList.add("hidden");
    if (emptyResults) emptyResults.classList.remove("hidden");
    const bio = document.getElementById("acousticBiomarkerCards");
    if (bio) {
        bio.classList.add("hidden");
        bio.innerHTML = "";
    }
    const ling = document.getElementById("linguisticFeaturesPanel");
    if (ling) ling.classList.add("hidden");

    if (waveformCanvas) {
        const ctx = waveformCanvas.getContext("2d");
        ctx.clearRect(0, 0, waveformCanvas.width, waveformCanvas.height);
    }
    if (pitchCanvas) {
        const ctx = pitchCanvas.getContext("2d");
        ctx.clearRect(0, 0, pitchCanvas.width, pitchCanvas.height);
    }
    if (spectrogramCanvas) {
        const ctx = spectrogramCanvas.getContext("2d");
        ctx.clearRect(0, 0, spectrogramCanvas.width, spectrogramCanvas.height);
    }
    if (timelineBar) timelineBar.innerHTML = "";
    if (sttTranscriptText) sttTranscriptText.textContent = "Click 'Transcribe Audio' to recognize words from this recording.";
    if (sttConfidenceText) sttConfidenceText.textContent = "";
    if (prepInfoText) prepInfoText.textContent = "";
    if (notesStatusText) notesStatusText.textContent = "";
    if (analysisNotesInput) analysisNotesInput.value = "";
}

async function handleFileSelect(file) {
    if (isAnalyzing || !file) return;
    clearError();
    resetResultsView();

    const allowed = [".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".webm", ".mp4", ".3gp"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
        showError("We could not open this file type. Please upload a WAV, MP3, M4A, AAC, OGG, FLAC, or WebM recording.");
        return;
    }

    if (file.size === 0) {
        showError("This file is empty. Please choose another voice recording.");
        return;
    }

    if (file.size > 25 * 1024 * 1024) {
        showError("This file is larger than 25 MB. Please choose a smaller recording.");
        return;
    }

    currentFile = file;
    fileName.textContent = file.name;
    fileMeta.textContent = `${formatFileSize(file.size)} · Getting recording ready...`;

    selectedFile.classList.remove("hidden");
    dropZone.classList.add("hidden");
    analyzeButton.disabled = false;
    buttonText.textContent = "Start Check";

    // Setup local audio preview
    const audioUrl = URL.createObjectURL(file);
    mainAudioPlayer.src = audioUrl;
    audioPlaybackPreview.classList.remove("hidden");

    // Fast Quality Check
    runFastQualityCheck(file);
}

async function runFastQualityCheck(file) {
    qualityBadge.classList.remove("hidden");
    badgeTag.textContent = "Checking quality...";
    badgeTag.className = "badge-tag checking";
    qualityMessage.textContent = "Evaluating clarity, background noise, and sound levels...";
    qualitySuggestions.innerHTML = "";

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch(`${API_BASE}/quality-check`, { method: "POST", body: formData });
        if (resp.ok) {
            const q = await resp.json();
            badgeTag.textContent = q.status;
            badgeTag.className = `badge-tag ${q.status.toLowerCase().replace(/[^a-z]/g, "")}`;
            badgeSnr.textContent = `SNR: ${q.snr_db} dB`;
            qualityMessage.textContent = q.message;
            qualitySuggestions.innerHTML = q.suggestions.map(s => `<span>&bull; ${s}</span>`).join(" ");
            fileMeta.textContent = `${formatFileSize(file.size)} · Ready to check (${q.duration_seconds}s)`;
        } else {
            qualityBadge.classList.add("hidden");
            fileMeta.textContent = `${formatFileSize(file.size)} · Ready to check`;
        }
    } catch {
        qualityBadge.classList.add("hidden");
        fileMeta.textContent = `${formatFileSize(file.size)} · Ready to check`;
    }
}

function clearCurrentFile() {
    currentFile = null;
    audioFile.value = "";
    selectedFile.classList.add("hidden");
    dropZone.classList.remove("hidden");
    qualityBadge.classList.add("hidden");
    audioPlaybackPreview.classList.add("hidden");
    mainAudioPlayer.src = "";
    analyzeButton.disabled = true;
    clearError();
    resetResultsView();
}

audioFile.addEventListener("click", () => {
    audioFile.value = "";
});
audioFile.addEventListener("change", () => {
    if (audioFile.files && audioFile.files[0]) handleFileSelect(audioFile.files[0]);
});

removeFile.addEventListener("click", clearCurrentFile);
analyzeAnother.addEventListener("click", () => {
    clearCurrentFile();
    document.getElementById("check-recording")?.scrollIntoView({ behavior: "smooth" });
});

// Drag & drop
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", e => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove("drag-over"); });
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFileSelect(e.dataTransfer.files[0]);
});

// -------------------------------------------------------------
// 3. RUN FULL ANALYSIS
// -------------------------------------------------------------
async function runAnalysis() {
    if (!currentFile || isAnalyzing) return;
    clearError();
    resetResultsView();
    isAnalyzing = true;

    analyzeButton.classList.add("is-loading");
    analyzeButton.disabled = true;
    buttonText.textContent = "Checking your recording…";
    buttonArrow.textContent = "↻";
    buttonIcon.textContent = "⏳";

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
        const resp = await fetch(`${API_BASE}/analyze-audio`, {
            method: "POST",
            body: formData
        });

        const data = await resp.json();
        if (!resp.ok) {
            throw new Error(data.detail || "Audio analysis failed.");
        }

        renderResults(data);
    } catch (err) {
        resetResultsView();
        if (err instanceof TypeError) {
            showError(`Could not connect to the analysis server (${API_BASE}). Please ensure FastAPI is running on port 8000.`);
        } else {
            showError(err.message || "An error occurred while checking the recording.");
        }
    } finally {
        isAnalyzing = false;
        analyzeButton.classList.remove("is-loading");
        analyzeButton.disabled = false;
        buttonText.textContent = "Start Check";
        buttonArrow.textContent = "→";
        buttonIcon.textContent = "▶";
    }
}

analyzeButton.addEventListener("click", runAnalysis);

// -------------------------------------------------------------
// 4. RENDER SCREENING RESULTS & SPEECH VISUALS
// -------------------------------------------------------------
function renderResults(data) {
    currentAnalysisData = data;
    const isTypical = (String(data.prediction || "").toUpperCase() === "CN");
    const probs = data.class_probabilities || { CN: 0.5, AD: 0.5 };
    const cn = Number(probs.CN || 0);
    const ad = Number(probs.AD || 0);

    predictionLabel.textContent = isTypical
        ? "Patterns Similar to Typical Voice Samples"
        : "Patterns Similar to Voice Samples with Cognitive Changes";

    predictionDescription.textContent = isTypical
        ? "The AI noticed acoustic features that align closely with speech patterns seen in our control study recordings."
        : "The AI noticed acoustic features that align closely with speech patterns seen in recordings from individuals with memory or thinking changes.";

    resultMark.textContent = isTypical ? "✓" : "!";
    resultMark.style.color = isTypical ? "var(--green)" : "var(--amber)";
    resultMark.style.background = isTypical ? "#1c3d31" : "#3d2f1c";
    resultMark.style.borderColor = isTypical ? "#2d5a47" : "#6b4d24";

    cnValue.textContent = `${(cn * 100).toFixed(1)}%`;
    adValue.textContent = `${(ad * 100).toFixed(1)}%`;
    cnBar.style.width = `${Math.max(0, Math.min(100, cn * 100))}%`;
    adBar.style.width = `${Math.max(0, Math.min(100, ad * 100))}%`;

    const dur = Number(data.duration_seconds || 0);
    audioDuration.textContent = `${dur.toFixed(1)} seconds`;

    const v = data.voice_details || {};
    speakingRatioText.textContent = `${Math.round((v.speaking_ratio || 0.8) * 100)}% of recording`;
    pitchVariationText.textContent = `${v.pitch_std_hz || 0} Hz std dev`;
    pauseCountText.textContent = `${v.pause_count || 0} pauses (${v.total_pause_seconds || 0}s)`;

    // Populate extended acoustic biomarker cards
    renderAcousticBiomarkers(v);

    // Populate linguistic features if transcription succeeded
    const ling = data.linguistic_features || null;
    const transcript = data.transcription || null;
    renderLinguisticSection(ling, transcript);

    if (rawPrediction) rawPrediction.textContent = data.prediction === "CN" ? "CN (Control / Typical Comparison)" : "AD (Cognitive Changes Study Group)";
    if (rawScores) rawScores.textContent = `Typical (CN): ${(cn * 100).toFixed(2)}% | Cognitive Changes (AD): ${(ad * 100).toFixed(2)}%`;
    if (recordIdText) recordIdText.textContent = `#${data.analysis_id || "-"}`;

    if (prepInfoText && data.preprocessing) {
        prepInfoText.textContent = `Preprocessed: 16 kHz Mono | Peak: ${data.preprocessing.normalized_peak} | ${data.preprocessing.filter}`;
    }

    // Update speech transcription panel directly from analysis output
    if (transcript && transcript.transcript && transcript.status === "ok") {
        sttTranscriptText.textContent = transcript.transcript;
        const conf = Math.round((transcript.confidence_score || 0) * 100);
        sttConfidenceText.textContent = `Confidence: ${conf}% (${transcript.engine || "STT"}) • ${transcript.notes || "Speech processed"}`;
    } else if (transcript && transcript.status === "unavailable") {
        sttTranscriptText.textContent = "[Speech recognition unavailable offline]";
        sttConfidenceText.textContent = transcript.notes || "Offline notice: Speech transcription requires local model or internet connectivity.";
    } else if (transcript && transcript.status === "unclear") {
        sttTranscriptText.textContent = "[Speech was too quiet, unclear, or absent]";
        sttConfidenceText.textContent = transcript.notes || "";
    } else {
        sttTranscriptText.textContent = "Click 'Transcribe Audio' to recognize words from this recording.";
        sttConfidenceText.textContent = "";
    }

    // Reset notes input
    analysisNotesInput.value = "";
    notesStatusText.textContent = "";

    // Draw speech visuals
    drawWaveform(data.speech_visuals?.waveform || []);
    drawPitchContour(data.speech_visuals?.pitch_contour || []);
    drawSpectrogram(data.speech_visuals?.spectrogram || []);
    drawPauseTimeline(data.speech_visuals?.pauses || [], dur);

    emptyResults.classList.add("hidden");
    resultContent.classList.remove("hidden");
    document.getElementById("results")?.scrollIntoView({ behavior: "smooth" });
}

// Visuals Draw Functions
function drawWaveform(points) {
    if (!waveformCanvas || !points.length) return;
    const ctx = waveformCanvas.getContext("2d");
    const w = waveformCanvas.width;
    const h = waveformCanvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#142030";
    ctx.fillRect(0, 0, w, h);

    const step = w / points.length;
    ctx.fillStyle = "#6397db";
    ctx.beginPath();
    points.forEach((val, i) => {
        const barH = val * (h - 10);
        const y = (h - barH) / 2;
        ctx.fillRect(i * step, y, Math.max(1, step - 1), barH);
    });
}

function drawPitchContour(points) {
    if (!pitchCanvas || !points.length) return;
    const ctx = pitchCanvas.getContext("2d");
    const w = pitchCanvas.width;
    const h = pitchCanvas.height;
    ctx.clearRect(0, 0, w, h);

    ctx.fillStyle = "#142030";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "#40546c";
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
    ctx.stroke();

    ctx.strokeStyle = "#4ade80";
    ctx.lineWidth = 2;
    ctx.beginPath();

    const minPitch = 60;
    const maxPitch = 350;
    let started = false;

    points.forEach((p, idx) => {
        const x = (idx / points.length) * w;
        if (p > minPitch) {
            const normP = Math.max(0, Math.min(1, (p - minPitch) / (maxPitch - minPitch)));
            const y = h - (normP * (h - 20) + 10);
            if (!started) { ctx.moveTo(x, y); started = true; }
            else ctx.lineTo(x, y);
        } else {
            started = false;
        }
    });
    ctx.stroke();
}

function drawSpectrogram(matrix) {
    if (!spectrogramCanvas || !matrix.length) return;
    const ctx = spectrogramCanvas.getContext("2d");
    const w = spectrogramCanvas.width;
    const h = spectrogramCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const rows = matrix.length;
    const cols = matrix[0].length;
    const cellW = w / cols;
    const cellH = h / rows;

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const intensity = matrix[rows - 1 - r][c]; // invert frequency axis
            const hue = 220 - (intensity * 180); // blue to amber
            const light = 15 + (intensity * 50);
            ctx.fillStyle = `hsl(${hue}, 85%, ${light}%)`;
            ctx.fillRect(c * cellW, r * cellH, cellW + 1, cellH + 1);
        }
    }
}

function drawPauseTimeline(pauses, totalDur) {
    if (!timelineBar || totalDur <= 0) return;
    timelineBar.innerHTML = "";

    // Speech background
    timelineBar.style.backgroundColor = "#2f69b3";

    pauses.forEach(p => {
        const leftPct = (p.start / totalDur) * 100;
        const widthPct = (p.duration / totalDur) * 100;
        const pauseSpan = document.createElement("span");
        pauseSpan.className = "timeline-pause-segment";
        pauseSpan.style.left = `${leftPct}%`;
        pauseSpan.style.width = `${Math.max(1.5, widthPct)}%`;
        pauseSpan.title = `Pause: ${p.duration}s at ${p.start}s`;
        timelineBar.appendChild(pauseSpan);
    });
}

// -------------------------------------------------------------
// 4b. ACOUSTIC BIOMARKERS & LINGUISTIC FEATURE CARDS
// -------------------------------------------------------------
function fmt(val, decimals) {
    if (val === undefined || val === null || val === "" || isNaN(Number(val))) return "—";
    return Number(val).toFixed(decimals);
}

function renderAcousticBiomarkers(v) {
    const container = document.getElementById("acousticBiomarkerCards");
    if (!container) return;

    const f0min = fmt(v.f0_min_hz, 0);
    const f0max = fmt(v.f0_max_hz, 0);
    const jitter = v.jitter_local !== undefined ? (Number(v.jitter_local) === 0 ? "<0.01%" : fmt(v.jitter_local * 100, 2) + "%") : "—";
    const shimmer = v.shimmer_local !== undefined ? fmt(v.shimmer_local * 100, 2) + "%" : "—";
    const hnr = v.hnr_mean_db !== undefined ? fmt(v.hnr_mean_db, 1) + " dB" : "—";
    const pauseMean = v.pause_duration_mean !== undefined ? fmt(v.pause_duration_mean, 2) + "s" : "—";
    const pauseRatio = v.pause_ratio !== undefined ? fmt(v.pause_ratio * 100, 1) + "%" : "—";

    container.innerHTML = `
        <div class="detail-card bio-card">
            <span class="detail-label" title="Lowest pitch frequency in the recording">Pitch Low (F0 Min)</span>
            <strong>${f0min === "—" ? f0min : f0min + " Hz"}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="Highest pitch frequency in the recording">Pitch High (F0 Max)</span>
            <strong>${f0max === "—" ? f0max : f0max + " Hz"}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="Jitter: small cycle-to-cycle pitch wobble. Low is normal.">Voice Steadiness (Jitter)</span>
            <strong>${jitter}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="Shimmer: small cycle-to-cycle loudness wobble. Low is normal.">Loudness Steadiness (Shimmer)</span>
            <strong>${shimmer}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="Harmonics-to-Noise Ratio: how clear vs. breathy the voice is. Higher is clearer.">Voice Clarity (HNR)</span>
            <strong>${hnr}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="Average length of each pause in the recording">Average Pause Length</span>
            <strong>${pauseMean}</strong>
        </div>
        <div class="detail-card bio-card">
            <span class="detail-label" title="What fraction of recording time is pauses">Pause Time (% of Total)</span>
            <strong>${pauseRatio}</strong>
        </div>
    `;
    container.classList.remove("hidden");
}

function renderLinguisticSection(ling, transcript) {
    const panel = document.getElementById("linguisticFeaturesPanel");
    if (!panel) return;
    if (!ling || !transcript || transcript.status !== "ok") {
        panel.classList.add("hidden");
        return;
    }
    const words = ling.total_words !== undefined ? ling.total_words : "—";
    const ttr = ling.type_token_ratio !== undefined ? fmt(ling.type_token_ratio, 3) : "—";
    const fillers = ling.filler_ratio !== undefined ? fmt(ling.filler_ratio * 100, 1) + "%" : "—";
    const wpm = ling.speaking_rate_wpm !== undefined ? fmt(ling.speaking_rate_wpm, 0) + " words/min" : "—";

    document.getElementById("lingWordCount").textContent = words;
    document.getElementById("lingTTR").textContent = ttr;
    document.getElementById("lingFillers").textContent = fillers;
    document.getElementById("lingWPM").textContent = wpm;
    panel.classList.remove("hidden");
}

// -------------------------------------------------------------
// 5. SPEECH-TO-TEXT / TRANSCRIBE
// -------------------------------------------------------------
triggerTranscribeBtn?.addEventListener("click", async () => {
    if (!currentFile) return;
    sttTranscriptText.textContent = "Transcribing words spoken in recording...";
    sttConfidenceText.textContent = "";
    triggerTranscribeBtn.disabled = true;

    const fd = new FormData();
    fd.append("file", currentFile);

    try {
        const resp = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: fd });
        const data = await resp.json();
        if (data.transcript) {
            sttTranscriptText.textContent = data.transcript;
            const conf = Math.round((data.confidence_score || 0) * 100);
            sttConfidenceText.textContent = `Confidence: ${conf}% (${data.engine || "STT"}) • ${data.notes || "Speech processed"}`;
        } else {
            sttTranscriptText.textContent = "[No words recognized in this audio]";
            sttConfidenceText.textContent = data.notes || "";
        }
    } catch (e) {
        sttTranscriptText.textContent = "[Speech recognition unavailable offline]";
        sttConfidenceText.textContent = "Offline notice: Speech transcription requires local Vosk/Whisper model or internet connectivity.";
    } finally {
        triggerTranscribeBtn.disabled = false;
    }
});

// -------------------------------------------------------------
// 6. RESEARCH NOTES & PDF DOWNLOAD
// -------------------------------------------------------------
saveNotesBtn?.addEventListener("click", async () => {
    if (!currentAnalysisData || !currentAnalysisData.analysis_id) return;
    const note = analysisNotesInput.value.trim();
    notesStatusText.textContent = "Saving...";

    const fd = new FormData();
    fd.append("notes", note);
    fd.append("feedback", "Screening completed");

    try {
        const resp = await fetch(`${API_BASE}/history/${currentAnalysisData.analysis_id}/notes`, {
            method: "PATCH",
            body: fd
        });
        if (resp.ok) {
            notesStatusText.textContent = "Saved to local database!";
            setTimeout(() => { notesStatusText.textContent = ""; }, 3000);
        }
    } catch {
        notesStatusText.textContent = "Could not save notes.";
    }
});

downloadPdfBtn?.addEventListener("click", () => {
    if (!currentAnalysisData || !currentAnalysisData.analysis_id) return;
    const downloadUrl = `${API_BASE}/history/${currentAnalysisData.analysis_id}/report`;
    window.open(downloadUrl, "_blank");
});

// -------------------------------------------------------------
// 7. MICROPHONE LIVE RECORDING
// -------------------------------------------------------------
startRecordBtn?.addEventListener("click", async () => {
    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        analyserNode = audioContext.createAnalyser();
        const source = audioContext.createMediaStreamSource(micStream);
        source.connect(analyserNode);
        analyserNode.fftSize = 64;

        recordedChunks = [];
        mediaRecorder = new MediaRecorder(micStream);
        mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
        mediaRecorder.onstop = () => {
            recordedBlob = new Blob(recordedChunks, { type: "audio/webm" });
            const recordedUrl = URL.createObjectURL(recordedBlob);
            recordedAudioPlayer.src = recordedUrl;
            recordedPreview.classList.remove("hidden");
            reRecordBtn.classList.remove("hidden");
            micStatusMsg.textContent = "Recording complete! Listen below or submit for screening.";
        };

        mediaRecorder.start();
        recordSeconds = 0;
        recordTimer.textContent = "00:00";
        recordInterval = setInterval(() => {
            recordSeconds++;
            const m = String(Math.floor(recordSeconds / 60)).padStart(2, "0");
            const s = String(recordSeconds % 60).padStart(2, "0");
            recordTimer.textContent = `${m}:${s}`;
        }, 1000);

        startRecordBtn.classList.add("hidden");
        stopRecordBtn.classList.remove("hidden");
        reRecordBtn.classList.add("hidden");
        recordedPreview.classList.add("hidden");
        micStatusMsg.textContent = "Recording in progress... speak naturally.";

        // Start live Web Speech API if supported
        startLiveSpeechRecognition();

        // Draw live mic volume wave
        drawMicVisualizer();

    } catch (err) {
        micStatusMsg.textContent = "Microphone access denied. Please grant microphone permission in your browser.";
    }
});

stopRecordBtn?.addEventListener("click", () => {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
    }
    if (micStream) micStream.getTracks().forEach(t => t.stop());
    clearInterval(recordInterval);
    if (speechRecognizer) speechRecognizer.stop();

    stopRecordBtn.classList.add("hidden");
    startRecordBtn.classList.remove("hidden");
});

reRecordBtn?.addEventListener("click", () => {
    recordedPreview.classList.add("hidden");
    reRecordBtn.classList.add("hidden");
    recordTimer.textContent = "00:00";
    micStatusMsg.textContent = "Ready to record again.";
    liveSttContent.textContent = "Spoken words will appear here...";
});

submitRecordingBtn?.addEventListener("click", async () => {
    if (!recordedBlob) return;
    const file = new File([recordedBlob], `microphone_check_${Date.now()}.webm`, { type: "audio/webm" });
    switchTab("home");
    await handleFileSelect(file);
    runAnalysis();
});

function drawMicVisualizer() {
    if (!micVisualizerCanvas || !analyserNode) return;
    const ctx = micVisualizerCanvas.getContext("2d");
    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);

    function render() {
        if (mediaRecorder && mediaRecorder.state === "recording") {
            requestAnimationFrame(render);
            analyserNode.getByteFrequencyData(dataArray);

            ctx.fillStyle = "#111b27";
            ctx.fillRect(0, 0, micVisualizerCanvas.width, micVisualizerCanvas.height);

            const barWidth = (micVisualizerCanvas.width / dataArray.length) * 1.5;
            let x = 0;
            for (let i = 0; i < dataArray.length; i++) {
                const barHeight = (dataArray[i] / 255) * micVisualizerCanvas.height;
                ctx.fillStyle = "#38bdf8";
                ctx.fillRect(x, micVisualizerCanvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }
    }
    render();
}

function startLiveSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = true;
    speechRecognizer.interimResults = true;

    speechRecognizer.onresult = event => {
        let transcript = "";
        for (let i = 0; i < event.results.length; i++) {
            transcript += event.results[i][0].transcript + " ";
        }
        liveSttContent.textContent = transcript || "Listening...";
    };

    speechRecognizer.onerror = () => {};
    try { speechRecognizer.start(); } catch {}
}

// -------------------------------------------------------------
// 8. BATCH MULTIPLE RECORDINGS ANALYSIS
// -------------------------------------------------------------
batchAudioFiles?.addEventListener("change", () => {
    batchFilesList = Array.from(batchAudioFiles.files).slice(0, 10);
    if (!batchFilesList.length) return;

    batchFilesUl.innerHTML = batchFilesList.map(f => `<li>🎵 <b>${f.name}</b> (${formatFileSize(f.size)})</li>`).join("");
    selectedBatchList.classList.remove("hidden");
    batchResultsPanel.classList.add("hidden");
});

runBatchBtn?.addEventListener("click", async () => {
    if (!batchFilesList.length) return;
    runBatchBtn.disabled = true;
    runBatchBtn.textContent = "Checking recordings... please wait...";

    const fd = new FormData();
    batchFilesList.forEach(file => fd.append("files", file));

    try {
        const resp = await fetch(`${API_BASE}/batch-analyze`, { method: "POST", body: fd });
        const data = await resp.json();
        lastBatchData = data;

        batchTotalCount.textContent = data.total_files;
        batchTypicalCount.textContent = data.typical_pattern_count;
        batchChangeCount.textContent = data.cognitive_change_pattern_count;

        batchTableBody.innerHTML = data.results.map(r => {
            const isTyp = (r.prediction === "CN");
            const badgeCls = isTyp ? "badge-cn" : (r.status === "Error" ? "badge-err" : "badge-ad");
            const resTxt = isTyp ? "Typical Voice Pattern" : (r.status === "Error" ? r.error : "Cognitive Changes Pattern");
            return `<tr>
                <td><b>${r.filename}</b></td>
                <td>${r.duration_seconds ? r.duration_seconds + "s" : "-"}</td>
                <td><span class="table-badge ${badgeCls}">${resTxt}</span></td>
                <td>${(r.cn_score !== undefined && r.cn_score !== null) ? (r.cn_score * 100).toFixed(1) + "%" : "-"}</td>
                <td>${(r.ad_score !== undefined && r.ad_score !== null) ? (r.ad_score * 100).toFixed(1) + "%" : "-"}</td>
                <td>${r.status}</td>
            </tr>`;
        }).join("");

        batchResultsPanel.classList.remove("hidden");
    } catch {
        alert("Batch check failed. Please check backend status.");
    } finally {
        runBatchBtn.disabled = false;
        runBatchBtn.textContent = "Start Checking All Recordings";
    }
});

exportBatchCsvBtn?.addEventListener("click", () => {
    if (!lastBatchData || !lastBatchData.results) return;
    let csv = "Filename,Duration_Sec,Screening_Result,Typical_Score,Changes_Score,Status\n";
    lastBatchData.results.forEach(r => {
        csv += `"${r.filename}",${r.duration_seconds || ""},"${r.prediction || ""}",${r.cn_score || ""},${r.ad_score || ""},"${r.status}"\n`;
    });
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `VoiceCheck_Batch_Results_${Date.now()}.csv`;
    a.click();
});

// -------------------------------------------------------------
// 9. HISTORY TAB
// -------------------------------------------------------------
async function loadHistory() {
    const search = historySearchInput?.value || "";
    const sort = historySortSelect?.value || "date_desc";

    historyTableBody.innerHTML = `<tr><td colspan="7" class="text-center">Loading past results...</td></tr>`;

    try {
        const resp = await fetch(`${API_BASE}/history?search=${encodeURIComponent(search)}&sort=${sort}`);
        const data = await resp.json();
        historyCache = data.analyses || [];

        if (!historyCache.length) {
            historyTableBody.innerHTML = `<tr><td colspan="7" class="text-center">No past results found. Check a voice recording to save results.</td></tr>`;
            return;
        }

        historyTableBody.innerHTML = historyCache.map(row => {
            const isTyp = (row.prediction === "CN");
            const badgeCls = isTyp ? "badge-cn" : "badge-ad";
            const resTxt = isTyp ? "Typical Voice Pattern" : "Cognitive Changes Pattern";
            const dateStr = row.created_at.replace("T", " ").substring(0, 16);

            return `<tr>
                <td>${dateStr}</td>
                <td><b>${row.filename}</b></td>
                <td>${row.duration_seconds}s</td>
                <td><span class="table-badge ${badgeCls}">${resTxt}</span></td>
                <td>Typical: ${(row.cn_score * 100).toFixed(0)}% | Changes: ${(row.ad_score * 100).toFixed(0)}%</td>
                <td><small>${row.notes || "None"}</small></td>
                <td>
                    <button class="btn-icon" title="Download PDF Report" onclick="downloadReportById(${row.id})">📥</button>
                    <button class="btn-icon danger" title="Delete result" onclick="deleteHistoryItem(${row.id})">🗑️</button>
                </td>
            </tr>`;
        }).join("");

    } catch {
        historyTableBody.innerHTML = `<tr><td colspan="7" class="text-center text-danger">Could not load history from server.</td></tr>`;
    }
}

historySearchInput?.addEventListener("input", debounce(loadHistory, 300));
historySortSelect?.addEventListener("change", loadHistory);
refreshHistoryBtn?.addEventListener("click", loadHistory);

window.downloadReportById = (id) => {
    window.open(`${API_BASE}/history/${id}/report`, "_blank");
};

window.deleteHistoryItem = async (id) => {
    if (!confirm("Are you sure you want to delete this screening result?")) return;
    try {
        const resp = await fetch(`${API_BASE}/history/${id}`, { method: "DELETE" });
        if (resp.ok) loadHistory();
    } catch {
        alert("Failed to delete record.");
    }
};

clearAllHistoryBtn?.addEventListener("click", async () => {
    if (!confirm("Are you sure you want to permanently delete ALL past screening results? This cannot be undone.")) return;
    try {
        const resp = await fetch(`${API_BASE}/history`, { method: "DELETE" });
        if (resp.ok) loadHistory();
    } catch {
        alert("Failed to clear history.");
    }
});

// -------------------------------------------------------------
// 10. COMPARE TWO RESULTS
// -------------------------------------------------------------
async function populateCompareSelectors() {
    try {
        const resp = await fetch(`${API_BASE}/history`);
        const data = await resp.json();
        const items = data.analyses || [];

        const optionsHtml = items.map(item => {
            const dateStr = item.created_at.substring(0, 10);
            return `<option value="${item.id}">#${item.id} - ${item.filename} (${dateStr}, ${item.prediction})</option>`;
        }).join("");

        compareSelectA.innerHTML = optionsHtml;
        compareSelectB.innerHTML = optionsHtml;

        if (items.length >= 2) {
            compareSelectB.selectedIndex = 1;
        }
    } catch {}
}

runCompareBtn?.addEventListener("click", async () => {
    const idA = compareSelectA.value;
    const idB = compareSelectB.value;
    if (!idA || !idB) return;

    try {
        const [respA, respB] = await Promise.all([
            fetch(`${API_BASE}/history/${idA}`).then(r => r.json()),
            fetch(`${API_BASE}/history/${idB}`).then(r => r.json())
        ]);

        compareBodyA.innerHTML = renderCompareCard(respA);
        compareBodyB.innerHTML = renderCompareCard(respB);
        compareResultsGrid.classList.remove("hidden");
    } catch {
        alert("Could not load comparative data.");
    }
});

function renderCompareCard(d) {
    const isTyp = (d.prediction === "CN");
    const pattern = isTyp ? "Typical Voice Pattern" : "Cognitive Changes Pattern";
    return `
        <div class="compare-item"><span>File Name:</span> <strong>${d.filename}</strong></div>
        <div class="compare-item"><span>Screening Pattern:</span> <strong>${pattern}</strong></div>
        <div class="compare-item"><span>AI Typical Score:</span> <strong>${(d.cn_score * 100).toFixed(1)}%</strong></div>
        <div class="compare-item"><span>AI Changes Score:</span> <strong>${(d.ad_score * 100).toFixed(1)}%</strong></div>
        <div class="compare-item"><span>Duration:</span> <strong>${d.duration_seconds}s</strong></div>
        <div class="compare-item"><span>Quality Status:</span> <strong>${d.quality_status}</strong></div>
        <div class="compare-item"><span>Notes:</span> <strong>${d.notes || "None"}</strong></div>
    `;
}

// -------------------------------------------------------------
// 11. MODEL EVALUATION & DATASET EXPLORER
// -------------------------------------------------------------
// Map raw feature names → readable labels
const FEATURE_LABELS = {
    "f0_max": "Pitch (Highest, Hz)",
    "f0_min": "Pitch (Lowest, Hz)",
    "shimmer_local": "Loudness Steadiness (Shimmer)",
    "jitter_local": "Pitch Steadiness (Jitter)",
    "hnr_mean": "Voice Clarity (HNR)",
    "zero_crossing_mean": "Voice Texture (Zero Crossings)",
    "zero_crossing_std": "Voice Texture Variation",
    "voiced_unvoiced_ratio": "Voiced vs. Unvoiced Ratio",
    "pause_count": "Number of Pauses",
    "pause_duration_mean": "Average Pause Length",
    "pause_ratio": "Total Pause Time Ratio",
    "quiet_ratio": "Silence Fraction",
    "quiet_duration_seconds": "Total Silence (seconds)",
    "duration_seconds": "Recording Length",
    "rms_mean": "Average Volume (RMS)",
    "rms_std": "Volume Variation",
    "pitch_mean": "Average Pitch",
    "pitch_std": "Pitch Variation",
    "mfcc_1_mean": "Vocal Tone (MFCC 1 mean)",
    "mfcc_1_std": "Vocal Tone Variation (MFCC 1 std)",
    "mfcc_2_mean": "Vocal Colour (MFCC 2 mean)",
    "mfcc_2_std": "Vocal Colour Variation (MFCC 2 std)",
    "mfcc_3_mean": "Resonance Pattern (MFCC 3 mean)",
    "mfcc_3_std": "Resonance Variation (MFCC 3 std)",
    "mfcc_4_mean": "Frequency Shape (MFCC 4 mean)",
    "mfcc_4_std": "Frequency Shape Variation (MFCC 4 std)",
    "mfcc_5_mean": "Sound Quality (MFCC 5 mean)",
    "mfcc_5_std": "Sound Quality Variation (MFCC 5 std)",
    "mfcc_6_mean": "Spectral Feature 6",
    "mfcc_6_std": "Spectral Feature 6 Variation",
    "mfcc_7_mean": "Spectral Feature 7",
    "mfcc_7_std": "Spectral Feature 7 Variation",
    "mfcc_8_mean": "Spectral Feature 8",
    "mfcc_8_std": "Spectral Feature 8 Variation",
    "mfcc_9_mean": "Spectral Feature 9",
    "mfcc_9_std": "Spectral Feature 9 Variation",
    "mfcc_10_mean": "Spectral Feature 10",
    "mfcc_10_std": "Spectral Feature 10 Variation",
    "mfcc_11_mean": "Spectral Feature 11",
    "mfcc_11_std": "Spectral Feature 11 Variation",
    "mfcc_12_mean": "Spectral Feature 12",
    "mfcc_12_std": "Spectral Feature 12 Variation",
    "mfcc_13_mean": "Spectral Feature 13",
    "mfcc_13_std": "Spectral Feature 13 Variation",
    "parselmouth_available": "Voice Analysis Available"
};

function featureLabel(raw) {
    return FEATURE_LABELS[raw] || raw.replace(/_/g, " ").replace(/\w/g, c => c.toUpperCase());
}

async function loadModelEval() {
    try {
        const resp = await fetch(`${API_BASE}/model-evaluation`);
        const evalData = await resp.json();

        // Primary metric cards
        document.getElementById("evalBalancedAcc").textContent = `${(evalData.balanced_accuracy * 100).toFixed(1)}%`;
        document.getElementById("evalSensitivity").textContent = `${(evalData.sensitivity_recall_ad * 100).toFixed(1)}%`;
        document.getElementById("evalSpecificity").textContent = `${(evalData.specificity_recall_cn * 100).toFixed(1)}%`;

        // Selected model label
        const selModelEl = document.getElementById("evalSelectedModel");
        if (selModelEl) selModelEl.textContent = evalData.selected_model || "Logistic Regression";

        // Held-out confusion matrix (live from API)
        const cm = evalData.confusion_matrix || {};
        const tn = cm.true_negatives_cn !== undefined ? cm.true_negatives_cn : "—";
        const fp = cm.false_positives_ad !== undefined ? cm.false_positives_ad : "—";
        const fn_ = cm.false_negatives_cn !== undefined ? cm.false_negatives_cn : "—";
        const tp = cm.true_positives_ad !== undefined ? cm.true_positives_ad : "—";
        const cmTN = document.getElementById("cmTN"); if (cmTN) cmTN.textContent = tn;
        const cmFP = document.getElementById("cmFP"); if (cmFP) cmFP.textContent = fp;
        const cmFN = document.getElementById("cmFN"); if (cmFN) cmFN.textContent = fn_;
        const cmTP = document.getElementById("cmTP"); if (cmTP) cmTP.textContent = tp;

        // Additional held-out metrics
        const precision = evalData.precision !== undefined ? `${(evalData.precision * 100).toFixed(1)}%` : "—";
        const f1 = evalData.f1_score !== undefined ? `${(evalData.f1_score * 100).toFixed(1)}%` : "—";
        const rocAuc = evalData.roc_auc !== undefined ? `${(evalData.roc_auc * 100).toFixed(1)}%` : "—";
        const precEl = document.getElementById("evalPrecision"); if (precEl) precEl.textContent = precision;
        const f1El = document.getElementById("evalF1"); if (f1El) f1El.textContent = f1;
        const aucEl = document.getElementById("evalROCAUC"); if (aucEl) aucEl.textContent = rocAuc;

        // Model comparison table
        const modelsData = evalData.models_compared || [];
        const compTableBody = document.getElementById("modelCompTableBody");
        if (compTableBody && modelsData.length) {
            compTableBody.innerHTML = modelsData.map((m, idx) => {
                const isSelected = idx === 0;
                const rowClass = isSelected ? "selected-model-row" : "";
                const badge = isSelected ? ' <span class="badge-selected">Selected</span>' : "";
                return `<tr class="${rowClass}">
                    <td><strong>${m.name}${badge}</strong></td>
                    <td>${m.balanced_accuracy || "—"}</td>
                    <td>${m.f1_score || "—"}</td>
                    <td>${m.roc_auc || "—"}</td>
                </tr>`;
            }).join("");
            const compTable = document.getElementById("modelCompTablePanel");
            if (compTable) compTable.classList.remove("hidden");
        }

        // Feature importance
        const respFi = await fetch(`${API_BASE}/model-insights`);
        const fiData = await respFi.json();
        const fiList = document.getElementById("featureImportanceList");
        if (fiList && fiData.top_features && fiData.top_features.length) {
            // Scale bars relative to top feature
            const maxImp = fiData.top_features[0].importance || 0.1;
            fiList.innerHTML = fiData.top_features
                .filter(f => f.feature !== "parselmouth_available")
                .map(f => {
                    const widthPct = Math.min(100, (f.importance / maxImp) * 100);
                    const label = featureLabel(f.feature);
                    return `
                        <div class="fi-row" title="${f.feature}">
                            <span class="fi-name">${label}</span>
                            <div class="fi-track"><div class="fi-bar" style="width: ${widthPct}%"></div></div>
                            <span class="fi-val">${(f.importance * 100).toFixed(2)}%</span>
                        </div>
                    `;
                }).join("");
        }
    } catch(e) {
        console.warn("loadModelEval error:", e);
    }
}

async function loadDatasetInfo() {
    try {
        const resp = await fetch(`${API_BASE}/dataset-info`);
        const d = await resp.json();
        document.getElementById("datasetTotalFiles").textContent = d.total_recordings;
        document.getElementById("datasetControlFiles").textContent = d.control_group_count;
        document.getElementById("datasetAdFiles").textContent = d.cognitive_change_count;
        document.getElementById("datasetAvgDuration").textContent = `${d.average_duration_seconds}s`;
    } catch {}
}

// -------------------------------------------------------------
// 12. SETTINGS & PRIVACY
// -------------------------------------------------------------
async function loadSettings() {
    try {
        const resp = await fetch(`${API_BASE}/settings`);
        const s = await resp.json();
        if (settingRetentionDays) settingRetentionDays.value = s.retention_days || "30";
        if (settingSaveAudio) settingSaveAudio.checked = (s.save_audio_recordings === "true");
        if (settingFontSize) {
            settingFontSize.value = s.font_size || "medium";
            applyFontSize(s.font_size || "medium");
        }
    } catch {}
}

saveSettingsBtn?.addEventListener("click", async () => {
    const payload = {
        retention_days: settingRetentionDays.value,
        save_audio_recordings: settingSaveAudio.checked ? "true" : "false",
        font_size: settingFontSize.value
    };

    settingsStatusText.textContent = "Saving...";
    try {
        const resp = await fetch(`${API_BASE}/settings`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (resp.ok) {
            applyFontSize(payload.font_size);
            settingsStatusText.textContent = "Preferences saved!";
            setTimeout(() => { settingsStatusText.textContent = ""; }, 3000);
        }
    } catch {
        settingsStatusText.textContent = "Error saving settings.";
    }
});

settingsClearDataBtn?.addEventListener("click", async () => {
    if (!confirm("This will permanently remove all saved analysis records from the database. Proceed?")) return;
    try {
        const resp = await fetch(`${API_BASE}/history`, { method: "DELETE" });
        if (resp.ok) alert("All past records have been deleted.");
    } catch {
        alert("Failed to delete records.");
    }
});

function applyFontSize(size) {
    document.body.classList.remove("font-small", "font-medium", "font-large");
    document.body.classList.add(`font-${size}`);
}

// -------------------------------------------------------------
// 13. SYSTEM HEALTH CHECK
// -------------------------------------------------------------
async function checkSystemHealth() {
    try {
        const resp = await fetch(`${API_BASE}/system-health`);
        const data = await resp.json();
        if (resp.ok) {
            if (systemStatusText) systemStatusText.textContent = "Ready to check";
            if (sidebarStatusDot) sidebarStatusDot.style.backgroundColor = "#66c7a0";
            document.getElementById("sysBackendStatus").textContent = "Connected (Port 8000)";
            document.getElementById("sysModelStatus").textContent = `Active (${data.model_path})`;
            document.getElementById("sysPyavStatus").textContent = `Active (v${data.pyav_version})`;
            document.getElementById("sysDbStatus").textContent = `Active (${data.database_records} saved analyses)`;
            document.getElementById("sysCodecsList").textContent = data.supported_codecs.join(", ").toUpperCase();
        }
    } catch {
        if (systemStatusText) systemStatusText.textContent = "Server offline";
        if (sidebarStatusDot) sidebarStatusDot.style.backgroundColor = "#ef4444";
        document.getElementById("sysBackendStatus").textContent = "Offline (Please start FastAPI)";
    }
}

function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// Initial check on load
checkSystemHealth();
setInterval(checkSystemHealth, 30000);
