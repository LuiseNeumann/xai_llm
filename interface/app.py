"""Streamlit-Oberfläche für interaktive Co-12-Prüfungen."""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

import streamlit as st

from interface.assessment import METRIC_INFO, interpret_metric, metric_state
from interface.pipeline_service import (
    AnalysisConfig,
    AnalysisRequest,
    LocalModelService,
    analyze_artifacts,
    analyze_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
COLOR_MAP = {
    "green": "#2f8f68",
    "yellow": "#d69b2d",
    "red": "#c94f4f",
}
STATE_CLASS = {
    "unauffällig": "metric-good",
    "prüfen": "metric-warn",
    "kritisch": "metric-bad",
    "nicht messbar": "metric-missing",
    "diagnostischer Wert": "metric-neutral",
}


st.set_page_config(
    page_title="Co-12 Prüflabor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172234;
        --muted: #657083;
        --paper: #f5f2ea;
        --panel: #ffffff;
        --line: #d7d2c7;
        --accent: #184e77;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #e8e3d8; border-right: 1px solid var(--line); }
    h1, h2, h3 { font-family: Georgia, "Times New Roman", serif; color: var(--ink); }
    h1 { letter-spacing: -0.035em; }
    .eyebrow {
        color: var(--accent); font-size: .75rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase; margin-bottom: .35rem;
    }
    .lead { color: var(--muted); max-width: 760px; font-size: 1.03rem; line-height: 1.6; }
    .audit-card {
        background: var(--panel); border: 1px solid var(--line); border-radius: 4px;
        padding: 1rem 1.15rem; min-height: 138px; box-shadow: 0 2px 0 rgba(23,34,52,.04);
    }
    .audit-card .label { color: var(--muted); font-size: .76rem; letter-spacing: .08em; text-transform: uppercase; }
    .audit-card .value { color: var(--ink); font-family: Georgia, serif; font-size: 1.35rem; margin: .35rem 0; }
    .signal { width: .72rem; height: .72rem; border-radius: 50%; display: inline-block; margin-right: .45rem; }
    .metric-good { color: #247553; font-weight: 700; }
    .metric-warn { color: #9a6815; font-weight: 700; }
    .metric-bad { color: #a83939; font-weight: 700; }
    .metric-missing { color: #737b88; font-weight: 700; }
    .metric-neutral { color: #315f80; font-weight: 700; }
    .answer-box {
        background: #fff; border-left: 4px solid var(--accent); border-top: 1px solid var(--line);
        border-right: 1px solid var(--line); border-bottom: 1px solid var(--line);
        padding: 1.1rem 1.25rem; margin: .5rem 0 1.25rem 0;
    }
    .step {
        display: grid; grid-template-columns: 2rem 1fr; gap: .55rem; padding: .62rem 0;
        border-bottom: 1px solid #e5e0d7;
    }
    .step-number { color: var(--accent); font-family: ui-monospace, monospace; font-weight: 700; }
    [data-testid="stMetricValue"] { font-family: Georgia, serif; }
    @media (max-width: 700px) {
        .audit-card { min-height: auto; margin-bottom: .5rem; }
        .lead { font-size: .96rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model_service(
    model_name: str,
    max_new_tokens: int,
    device: str,
) -> LocalModelService:
    """Lädt ein Modell einmalig und verwendet es über mehrere Anfragen hinweg."""
    return LocalModelService(model_name, max_new_tokens=max_new_tokens, device=device)


def _split_lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip()]


def _split_terms(value: str) -> list[str]:
    return [term.strip() for term in value.split(",") if term.strip()]


def _read_uploaded_jsonl(uploaded_file: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    text = uploaded_file.getvalue().decode("utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Ungültiges JSON in Zeile {line_number}") from exc
    return records


def _assessment_cards(assessment: dict[str, Any]) -> None:
    columns = st.columns(3)
    color = COLOR_MAP.get(assessment.get("color", "yellow"), COLOR_MAP["yellow"])
    with columns[0]:
        st.markdown(
            f"""
            <div class="audit-card">
              <div class="label">Prüfstatus</div>
              <div class="value"><span class="signal" style="background:{color}"></span>{assessment['label']}</div>
              <div>{assessment['message']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[1]:
        st.markdown(
            f"""
            <div class="audit-card">
              <div class="label">Evidenz</div>
              <div class="value">{assessment['evidence_level'].capitalize()}</div>
              <div>{assessment['available_metrics']} von {assessment['expected_metrics']} Eigenschaften messbar</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with columns[2]:
        coverage = 100 * float(assessment.get("evidence_coverage", 0.0))
        st.markdown(
            f"""
            <div class="audit-card">
              <div class="label">Abdeckung</div>
              <div class="value">{coverage:.0f} %</div>
              <div>Fehlende Evidenz wird nicht als positives Ergebnis gewertet.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _metric_details(metrics: list[dict[str, Any]], key_prefix: str) -> None:
    for index, result in enumerate(metrics):
        name = str(result.get("name", ""))
        info = METRIC_INFO.get(name, {"title": name, "description": ""})
        state = metric_state(result)
        with st.expander(f"{info['title']} · {state}", expanded=state == "kritisch"):
            st.markdown(f"**Was wird geprüft?** {info['description']}")
            st.markdown(
                f"<span class='{STATE_CLASS.get(state, 'metric-neutral')}'>{interpret_metric(result)}</span>",
                unsafe_allow_html=True,
            )
            score = result.get("score")
            if score is not None and result.get("higher_is_better", True) and 0 <= float(score) <= 1:
                st.progress(float(score))
            direction = "höher ist besser" if result.get("higher_is_better", True) else "kleiner ist kompakter/einfacher"
            st.caption(f"Messrichtung: {direction}. Status: {result.get('status', 'unbekannt')}.")
            st.json(result.get("details", {}), expanded=False)


def _warnings(assessment: dict[str, Any]) -> None:
    warnings = assessment.get("warnings", [])
    recommendations = assessment.get("recommendations", [])
    if warnings:
        st.subheader("Warum eine Prüfung sinnvoll sein kann")
        for warning in warnings:
            st.markdown(f"- {warning}")
    if recommendations:
        st.subheader("Empfohlene nächste Prüfungen")
        for recommendation in recommendations:
            st.markdown(f"- {recommendation}")


def _structure_diagnostics(structure: dict[str, Any]) -> None:
    note = structure.get("note")
    if note:
        st.info(note)
    artifact = structure.get("artifact")
    if not artifact:
        return
    effects = artifact.get("node_effects", {})
    if effects:
        st.subheader("Layer mit dem stärksten Patching-Effekt")
        rows = [
            {"Komponente": name, "Effekt auf die Logit-Differenz": value}
            for name, value in sorted(effects.items(), key=lambda item: abs(item[1]), reverse=True)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    limitations = artifact.get("metadata", {}).get("limitations", [])
    if limitations:
        with st.expander("Grenzen der Strukturanalyse"):
            for limitation in limitations:
                st.markdown(f"- {limitation}")


def render_live_report(report: dict[str, Any]) -> None:
    _assessment_cards(report["overall_assessment"])
    st.markdown("## Modellantwort")
    answer = escape(str(report.get("answer") or "Keine Antwort extrahiert."))
    st.markdown(f"<div class='answer-box'>{answer}</div>", unsafe_allow_html=True)

    steps = report.get("steps", [])
    if steps:
        st.markdown("### Ausgegebene Begründung")
        for index, step in enumerate(steps, start=1):
            safe_step = escape(str(step))
            st.markdown(
                f"<div class='step'><div class='step-number'>{index:02d}</div><div>{safe_step}</div></div>",
                unsafe_allow_html=True,
            )

    _warnings(report["overall_assessment"])
    tab_names = ["Chain of Thought"]
    if report.get("structure"):
        tab_names.append("Innere Modellstruktur")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        _assessment_cards(report["cot"]["assessment"])
        _metric_details(report["cot"]["metrics"], "cot")
    if report.get("structure"):
        with tabs[1]:
            assessment = report["structure"].get("assessment")
            if assessment:
                _assessment_cards(assessment)
            _structure_diagnostics(report["structure"])
            _metric_details(report["structure"]["metrics"], "structure")

    st.download_button(
        "Vollständigen Bericht als JSON herunterladen",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="co12_bericht.json",
        mime="application/json",
        use_container_width=True,
    )


def render_artifact_report(report: dict[str, Any]) -> None:
    _assessment_cards(report["overall_assessment"])
    tabs = st.tabs(["Chain of Thought", "Innere Modellstruktur"])
    with tabs[0]:
        if report.get("cot"):
            _assessment_cards(report["cot"]["assessment"])
            _metric_details(report["cot"]["metrics"], "upload-cot")
        else:
            st.info("Keine CoT-Artefakte hochgeladen.")
    with tabs[1]:
        if report.get("structure"):
            _assessment_cards(report["structure"]["assessment"])
            _metric_details(report["structure"]["metrics"], "upload-structure")
        else:
            st.info("Keine Strukturartefakte hochgeladen.")
    st.download_button(
        "Auswertung als JSON herunterladen",
        data=json.dumps(report, ensure_ascii=False, indent=2),
        file_name="co12_artefaktbericht.json",
        mime="application/json",
        use_container_width=True,
    )


def direct_analysis_tab() -> None:
    st.markdown('<div class="eyebrow">Interaktive Einzelfallprüfung</div>', unsafe_allow_html=True)
    st.title("Prompt prüfen, Evidenz sichtbar machen")
    st.markdown(
        '<div class="lead">Das Modell beantwortet deinen Prompt und die Pipeline prüft, welche Co-12-Eigenschaften mit den verfügbaren Daten tatsächlich messbar sind. Die Ampel ist eine Prüfempfehlung, kein Wahrheitszertifikat.</div>',
        unsafe_allow_html=True,
    )

    prompt = st.text_area(
        "Prompt",
        height=170,
        placeholder="Beispiel: Erkläre Schritt für Schritt, warum ...",
    )
    with st.expander("Referenzen und kontrollierte Gegenfakten", expanded=False):
        st.caption("Diese Angaben erhöhen die Evidenzabdeckung und ermöglichen zusätzliche Co-12-Prüfungen.")
        expected_answer = st.text_input("Erwartete Antwort", help="Optional: bekannte korrekte Endantwort")
        alternative_answer = st.text_input("Alternativantwort", help="Für die Logit-Differenz der Strukturanalyse")
        corrupted_prompt = st.text_area(
            "Kontrafaktischer Prompt",
            height=100,
            help="Minimal veränderte Eingabe, für die sich die relevante Antwort ändern sollte",
        )
        gold_steps_text = st.text_area("Referenzschritte, ein Schritt pro Zeile", height=110)
        gold_concepts_text = st.text_input("Erwartete Konzepte, kommagetrennt")
        constraint_columns = st.columns(3)
        with constraint_columns[0]:
            max_steps_enabled = st.checkbox("Maximale Schrittzahl vorgeben")
            max_steps = st.number_input("Maximale Schritte", min_value=1, max_value=30, value=5, disabled=not max_steps_enabled)
        with constraint_columns[1]:
            required_terms_text = st.text_input("Erforderliche Begriffe")
        with constraint_columns[2]:
            forbidden_terms_text = st.text_input("Verbotene Begriffe")

    if st.button("Antwort erzeugen und prüfen", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("Bitte zuerst einen Prompt eingeben.")
            return
        request = AnalysisRequest(
            prompt=prompt.strip(),
            expected_answer=expected_answer.strip(),
            alternative_answer=alternative_answer.strip(),
            corrupted_prompt=corrupted_prompt.strip(),
            gold_steps=_split_lines(gold_steps_text),
            gold_concepts=_split_terms(gold_concepts_text),
            max_steps=int(max_steps) if max_steps_enabled else None,
            required_terms=_split_terms(required_terms_text),
            forbidden_terms=_split_terms(forbidden_terms_text),
        )
        config = AnalysisConfig(
            mode=st.session_state["analysemodus"],
            samples=st.session_state["samples"],
            temperature=st.session_state["temperature"],
            seed=st.session_state["seed"],
            run_structure=st.session_state["struktur"],
            structure_top_k=st.session_state["top_k"],
        )
        progress_bar = st.progress(0.0)
        progress_text = st.empty()

        def update_progress(value: float, message: str) -> None:
            progress_bar.progress(min(max(value, 0.0), 1.0))
            progress_text.caption(message)

        try:
            update_progress(0.01, "Modell wird geladen")
            service = load_model_service(
                st.session_state["model_name"],
                st.session_state["max_new_tokens"],
                st.session_state["device"],
            )
            report = analyze_prompt(
                service,
                request,
                config,
                model_name=st.session_state["model_name"],
                progress=update_progress,
            )
            st.session_state["live_report"] = report
        except Exception as exc:
            st.exception(exc)
        finally:
            progress_bar.empty()
            progress_text.empty()

    if report := st.session_state.get("live_report"):
        render_live_report(report)


def artifact_tab() -> None:
    st.markdown('<div class="eyebrow">Reproduzierbare Offline-Auswertung</div>', unsafe_allow_html=True)
    st.title("Gesammelte Artefakte prüfen")
    st.markdown(
        "Bereits erzeugte JSONL-Artefakte können ohne Modelllauf ausgewertet werden. "
        "Die Beispieldateien unter `data/` zeigen das erwartete Format."
    )
    columns = st.columns(2)
    with columns[0]:
        cot_file = st.file_uploader("CoT-Artefakte", type=["jsonl"], key="cot-upload")
    with columns[1]:
        structure_file = st.file_uploader("Strukturartefakte", type=["jsonl"], key="structure-upload")
    if st.button("Artefakte auswerten", use_container_width=True):
        if not cot_file and not structure_file:
            st.warning("Bitte mindestens eine JSONL-Datei auswählen.")
            return
        try:
            cot_records = _read_uploaded_jsonl(cot_file) if cot_file else None
            structure_records = _read_uploaded_jsonl(structure_file) if structure_file else None
            st.session_state["artifact_report"] = analyze_artifacts(cot_records, structure_records)
        except (UnicodeDecodeError, ValueError) as exc:
            st.error(str(exc))
    if report := st.session_state.get("artifact_report"):
        render_artifact_report(report)


def methods_tab() -> None:
    st.markdown('<div class="eyebrow">Methodische Grundlage</div>', unsafe_allow_html=True)
    content = (ROOT / "CO12_METHODEN.md").read_text(encoding="utf-8")
    st.markdown(content)


with st.sidebar:
    st.markdown("## Versuchsaufbau")
    st.text_input("Hugging-Face-Modell", value="Qwen/Qwen2.5-1.5B-Instruct", key="model_name")
    st.selectbox("Gerät", options=["auto", "cpu", "cuda"], key="device")
    selected_mode = st.radio(
        "Analysemodus",
        options=["Schnell", "Vollständig"],
        help="Der vollständige Modus erzeugt mehr Varianten und benötigt deutlich mehr Zeit.",
    )
    st.session_state["analysemodus"] = selected_mode.casefold()
    default_samples = 3 if selected_mode == "Schnell" else 6
    st.slider("Reasoning-Samples", min_value=2, max_value=12, value=default_samples, key="samples")
    st.slider("Temperatur", min_value=0.1, max_value=1.5, value=0.7, step=0.1, key="temperature")
    st.number_input("Zufalls-Seed", min_value=0, value=42, key="seed")
    st.number_input("Maximale neue Tokens", min_value=64, max_value=2048, value=384, step=64, key="max_new_tokens")
    st.divider()
    st.checkbox(
        "Innere Modellstruktur analysieren",
        value=False,
        key="struktur",
        help="Benötigt Alternativantwort und kontrafaktischen Prompt; führt viele zusätzliche Forward-Pässe aus.",
    )
    st.slider("Anzahl relevanter Layer", min_value=1, max_value=20, value=5, key="top_k", disabled=not st.session_state.get("struktur", False))
    st.caption("Lokale Verarbeitung: Prompts werden nicht an einen externen API-Anbieter gesendet.")


direct_tab, upload_tab, documentation_tab = st.tabs(
    ["Prompt analysieren", "Artefakte auswerten", "Methoden und Quellen"]
)
with direct_tab:
    direct_analysis_tab()
with upload_tab:
    artifact_tab()
with documentation_tab:
    methods_tab()
