import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import io
from datetime import datetime
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_pdf_report(analysis: Dict[str, Any]) -> bytes:
    """
    Generates a complete, professional downloadable PDF report using reportlab.
    Includes analysis metadata, screening result, AI scores, acoustic biomarkers,
    linguistic metrics (if available), speech visuals chart, and health cautions.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#102a45")
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#556980")
    )

    h2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1b395a"),
        spaceBefore=8,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#223344")
    )

    caution_style = ParagraphStyle(
        "CautionText",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#8a3800")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("Voice Check &mdash; Screening Summary Report", title_style))
    elements.append(Paragraph(
        "AI-Based Speech Pattern Exploration &bull; Research Prototype &bull; Not a Medical Diagnosis",
        subtitle_style
    ))
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2b5278"), spaceAfter=10))

    # 2. Recording Metadata Table
    pred = str(analysis.get("prediction", "")).upper()
    is_typical = (pred == "CN")
    duration = float(analysis.get("duration_seconds", 0.0))
    created_at = analysis.get("created_at", datetime.now().isoformat())
    filename = analysis.get("filename", "audio_recording")
    cn_score = float(analysis.get("cn_score", 0.0))
    ad_score = float(analysis.get("ad_score", 0.0))

    meta_data = [
        [
            Paragraph("<b>File Checked:</b>", body_style),
            Paragraph(str(filename), body_style),
            Paragraph("<b>Recording Length:</b>", body_style),
            Paragraph(f"{duration:.1f} seconds", body_style),
        ],
        [
            Paragraph("<b>Date & Time:</b>", body_style),
            Paragraph(str(created_at).replace("T", " ")[:19], body_style),
            Paragraph("<b>Audio Quality:</b>", body_style),
            Paragraph(str(analysis.get("quality_status", "Clear")), body_style),
        ]
    ]

    meta_table = Table(meta_data, colWidths=[90, 180, 110, 160])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0f4f8")),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0dbe5")),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 10))

    # 3. Screening Result Banner
    result_title = "Patterns Similar to Typical Voice Samples" if is_typical else "Patterns Similar to Voice Samples with Cognitive Changes"
    result_bg = colors.HexColor("#eef7f2") if is_typical else colors.HexColor("#fff8ee")
    result_border = colors.HexColor("#389e6e") if is_typical else colors.HexColor("#d9822b")

    result_box = [
        [Paragraph(f"<b>Screening Result:</b> {result_title}", ParagraphStyle(
            "ResTitle", fontName="Helvetica-Bold", fontSize=12, leading=16,
            textColor=colors.HexColor("#1b5e3b") if is_typical else colors.HexColor("#8a4700")
        ))],
        [Paragraph(
            "The AI model examined acoustic patterns (pitch variation, pauses, steadiness, and sound features) and compared them with research study recordings.",
            body_style
        )]
    ]
    res_table = Table(result_box, colWidths=[540])
    res_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), result_bg),
        ("BOX", (0, 0), (-1, -1), 1.5, result_border),
        ("PADDING", (0, 0), (-1, -1), 7),
    ]))
    elements.append(res_table)
    elements.append(Spacer(1, 10))

    # 4. AI Result Scores Table
    elements.append(Paragraph("AI Result Scores", h2_style))
    score_data = [
        [Paragraph("<b>Study Group Pattern</b>", body_style), Paragraph("<b>Similarity Score</b>", body_style), Paragraph("<b>Explanation</b>", body_style)],
        [
            Paragraph("Typical voice pattern", body_style),
            Paragraph(f"<b>{cn_score * 100:.1f}%</b>", body_style),
            Paragraph("Resemblance to speech samples from participants with typical cognitive health.", body_style)
        ],
        [
            Paragraph("Pattern with cognitive changes", body_style),
            Paragraph(f"<b>{ad_score * 100:.1f}%</b>", body_style),
            Paragraph("Resemblance to speech samples from participants with memory or thinking changes.", body_style)
        ],
    ]
    score_table = Table(score_data, colWidths=[150, 95, 295])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b4a6f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0dbe5")),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(score_table)
    elements.append(Spacer(1, 10))

    # 5. Acoustic Biomarkers & Voice Details
    features = analysis.get("features", {})
    if features:
        elements.append(Paragraph("Voice Details &amp; Acoustic Biomarkers", h2_style))

        def _fmt(val, dec=1, suffix=""):
            if val is None or str(val) == "":
                return "&mdash;"
            try:
                num = float(val)
                return f"{num:.{dec}f}{suffix}"
            except (ValueError, TypeError):
                return str(val)

        f0_min = _fmt(features.get("f0_min"), 0, " Hz")
        f0_max = _fmt(features.get("f0_max"), 0, " Hz")

        jit = features.get("jitter_local")
        jitter_str = "&mdash;"
        if jit is not None:
            try:
                j_val = float(jit)
                jitter_str = "&lt;0.01%" if j_val == 0 else f"{j_val * 100:.2f}%"
            except (ValueError, TypeError):
                jitter_str = str(jit)

        shim = features.get("shimmer_local")
        shimmer_str = "&mdash;"
        if shim is not None:
            try:
                shimmer_str = f"{float(shim) * 100:.2f}%"
            except (ValueError, TypeError):
                shimmer_str = str(shim)

        hnr = _fmt(features.get("hnr_mean"), 1, " dB")
        pause_cnt = str(features.get("pause_count", "&mdash;"))
        pause_dur = _fmt(features.get("pause_duration_mean"), 2, "s")
        pause_pct = _fmt((float(features["pause_ratio"]) * 100) if "pause_ratio" in features else None, 1, "%")
        voiced_ratio = _fmt(features.get("voiced_unvoiced_ratio"), 2)

        bio_data = [
            [
                Paragraph("<b>Biomarker</b>", body_style),
                Paragraph("<b>Measured</b>", body_style),
                Paragraph("<b>Biomarker</b>", body_style),
                Paragraph("<b>Measured</b>", body_style),
            ],
            [
                Paragraph("Pitch Low (F0 Min)", body_style),
                Paragraph(f0_min, body_style),
                Paragraph("Pitch High (F0 Max)", body_style),
                Paragraph(f0_max, body_style),
            ],
            [
                Paragraph("Pitch Steadiness (Jitter)", body_style),
                Paragraph(jitter_str, body_style),
                Paragraph("Loudness Steadiness (Shimmer)", body_style),
                Paragraph(shimmer_str, body_style),
            ],
            [
                Paragraph("Voice Clarity (HNR)", body_style),
                Paragraph(hnr, body_style),
                Paragraph("Voiced / Unvoiced Ratio", body_style),
                Paragraph(voiced_ratio, body_style),
            ],
            [
                Paragraph("Audible Pauses", body_style),
                Paragraph(pause_cnt, body_style),
                Paragraph("Average Pause Length", body_style),
                Paragraph(f"{pause_dur} ({pause_pct} of audio)", body_style),
            ]
        ]

        bio_table = Table(bio_data, colWidths=[150, 120, 150, 120])
        bio_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#37557a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0dbe5")),
            ("PADDING", (0, 0), (-1, -1), 4.5),
        ]))
        elements.append(bio_table)
        elements.append(Spacer(1, 10))

    # 6. Linguistic & Spoken Words Analysis (if present)
    ling = features.get("_linguistic") or analysis.get("linguistic_features")
    stt = features.get("_transcription") or analysis.get("transcription")
    if ling and isinstance(ling, dict) and ling.get("total_words", 0) > 0:
        elements.append(Paragraph("Words Spoken &amp; Speech Rhythm", h2_style))
        transcript_text = stt.get("transcript", "") if isinstance(stt, dict) else ""
        if transcript_text:
            elements.append(Paragraph(f"<b>Recognized Words:</b> &ldquo;{transcript_text}&rdquo;", body_style))
            elements.append(Spacer(1, 4))

        ling_data = [
            [
                Paragraph("<b>Total Words Spoken</b>", body_style),
                Paragraph(str(ling.get("total_words", "&mdash;")), body_style),
                Paragraph("<b>Vocabulary Variety (TTR)</b>", body_style),
                Paragraph(f"{ling.get('type_token_ratio', 0):.3f}", body_style),
            ],
            [
                Paragraph("<b>Filler Word Ratio</b>", body_style),
                Paragraph(f"{ling.get('filler_ratio', 0) * 100:.1f}%", body_style),
                Paragraph("<b>Speaking Speed</b>", body_style),
                Paragraph(f"{ling.get('speaking_rate_wpm', 0):.0f} WPM", body_style),
            ]
        ]
        ling_table = Table(ling_data, colWidths=[150, 120, 150, 120])
        ling_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f9f5")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cce5d4")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(ling_table)
        elements.append(Spacer(1, 10))

    # 7. Speech Visuals Chart (rendered via matplotlib into PNG)
    visuals = analysis.get("speech_visuals", {})
    waveform = visuals.get("waveform", [])
    pitch_contour = visuals.get("pitch_contour", [])

    if waveform or pitch_contour:
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 2.2), dpi=150)
            fig.patch.set_facecolor("#f8fafc")

            if waveform:
                ax1.plot(waveform, color="#2f69b3", lw=1.1)
                ax1.fill_between(range(len(waveform)), waveform, color="#8bbcff", alpha=0.3)
                ax1.set_title("Voice Energy Envelope (Waveform)", fontsize=7.5, color="#1e293b", pad=2)
                ax1.set_ylim(0, 1.05)
                ax1.tick_params(labelsize=6)
                ax1.set_facecolor("#ffffff")

            if pitch_contour:
                valid_pitch = [p if p > 0 else np.nan for p in pitch_contour]
                ax2.plot(valid_pitch, color="#059669", lw=1.3, marker="o", markersize=1.8)
                ax2.set_title("Pitch Variation Contour (Hz)", fontsize=7.5, color="#1e293b", pad=2)
                ax2.set_ylim(50, 350)
                ax2.tick_params(labelsize=6)
                ax2.set_facecolor("#ffffff")

            plt.tight_layout()
            img_buf = io.BytesIO()
            plt.savefig(img_buf, format="png", bbox_inches="tight")
            plt.close(fig)
            img_buf.seek(0)

            chart_img = Image(img_buf, width=540, height=165)
            elements.append(Paragraph("Speech Visuals (Recorded Acoustic Data)", h2_style))
            elements.append(chart_img)
            elements.append(Spacer(1, 8))
        except Exception:
            pass

    # 8. Important Health Caution
    caution_content = [
        [Paragraph("<b>IMPORTANT HEALTH NOTICE &amp; LIMITATIONS</b>", ParagraphStyle(
            "CautionH", fontName="Helvetica-Bold", fontSize=8.5, textColor=colors.HexColor("#8a3800")
        ))],
        [Paragraph(
            "This screening tool is an exploratory research system based on acoustic analysis of speech recordings. "
            "It cannot diagnose, confirm, or rule out Alzheimer's disease, dementia, or any other medical condition. "
            "Everyday factors such as microphone quality, fatigue, background noise, or emotional state can influence voice features. "
            "If you or a family member have questions or concerns about memory, thinking, or overall health, please consult a qualified doctor or healthcare professional.",
            caution_style
        )]
    ]
    caution_table = Table(caution_content, colWidths=[540])
    caution_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fff9f2")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f0b875")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(caution_table)

    doc.build(elements)
    return buffer.getvalue()
