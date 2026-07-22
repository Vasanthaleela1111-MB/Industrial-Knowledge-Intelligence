"""Flask frontend for IndustrialMind AI.

Run with: python frontend/flask_app.py
The FastAPI service must be running at API_BASE_URL (default http://localhost:8000).
"""

import os
from functools import wraps

import requests
from flask import Flask, flash, redirect, render_template, request, url_for


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
ALLOWED_EXTENSIONS = {
    "pdf", "docx", "pptx", "xlsx", "csv", "png", "jpg", "jpeg", "bmp",
    "tiff", "eml", "html", "md", "txt",
}

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "change-this-in-production")
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_UPLOAD_BYTES", 250 * 1024 * 1024))


def api_get(path):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=(5, 120))
    response.raise_for_status()
    return response.json()


def api_post(path, payload):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=(10, 300))
    response.raise_for_status()
    return response.json()


def safe_api(default=None):
    """Show a useful UI error when the FastAPI service cannot be reached."""
    def decorator(view):
        @wraps(view)
        def wrapped(*args, **kwargs):
            try:
                return view(*args, **kwargs)
            except requests.RequestException as exc:
                flash(f"Could not contact the IndustrialMind API at {API_BASE_URL}: {exc}", "error")
                return render_template("base.html", health={}, page_title="IndustrialMind AI"), 503
        return wrapped
    return decorator


def layout_context(**context):
    try:
        health = api_get("/")
    except requests.RequestException:
        health = {}
    return {"health": health, **context}


@app.get("/")
@safe_api()
def documentation():
    return render_template("documentation.html", **layout_context(page_title="Project Documentation", active="documentation"))


# @app.get("/intelligence")
# @safe_api()
# def intelligence():
#     metrics = api_get("/metrics")
#     documents = api_get("/documents")
#     # graph = api_get("/knowledge-graph")
#     selected = ",".join()

#     graph = api_get(
#         f"/knowledge-graph?selected_documents={selected}"
#     )
#     return render_template(
#         "intelligence.html",
#         **layout_context(page_title="Industrial Intelligence", metrics=metrics, documents=documents, graph=graph, active="intelligence"),
#     )

@app.get("/intelligence")
@safe_api()
def intelligence():

    metrics = api_get("/metrics")

    documents = api_get("/documents")

    graph = api_get("/knowledge-graph")

    return render_template(
        "intelligence.html",
        **layout_context(
            page_title="Industrial Intelligence",
            metrics=metrics,
            documents=documents,
            graph=graph,
            active="intelligence",
        ),
    )


@app.post("/documents/upload")
@safe_api()
def upload_documents():
    files = request.files.getlist("files")
    uploaded, failures = [], []
    for file in files:
        if not file or not file.filename:
            continue
        extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if extension not in ALLOWED_EXTENSIONS:
            failures.append(f"{file.filename}: unsupported file type")
            continue
        try:
            response = requests.post(
                f"{API_BASE_URL}/documents/upload",
                files={"file": (file.filename, file.stream, file.mimetype)},
                timeout=(10, 1800),
            )
            response.raise_for_status()
            uploaded.append(file.filename)
        except requests.RequestException as exc:
            failures.append(f"{file.filename}: {exc}")
    if uploaded:
        flash(f"Successfully processed {len(uploaded)} document(s): {', '.join(uploaded)}", "success")
    for failure in failures:
        flash(failure, "error")
    return redirect(url_for("intelligence") + "#workspace")

@app.post("/delete-documents")
@safe_api()
def delete_documents():

    selected_documents = request.form.getlist("selected_documents")

    if not selected_documents:
        flash("Select at least one document.", "error")
        return redirect(url_for("intelligence") + "#assistant")

    api_post(
        "/delete-documents",
        {
            "document_ids": selected_documents
        }
    )

    flash(
        f"Deleted {len(selected_documents)} document(s).",
        "success"
    )

    return redirect(url_for("intelligence") + "#assistant")


@app.post("/clear")
@safe_api()
def clear_knowledge_base():
    api_post("/clear", {})
    flash("Knowledge base cleared successfully.", "success")
    return redirect(url_for("intelligence") + "#workspace")


@app.post("/ask")
@safe_api()
def ask():
    question = request.form.get("question", "").strip()
    selected_documents = request.form.getlist("selected_documents")
    top_k = int(request.form.get("top_k", 5))
    if not question or not selected_documents:
        flash("Enter a question and select at least one document.", "error")
        return redirect(url_for("intelligence") + "#assistant")
    answer = api_post("/ask", {"question": question, "selected_documents": selected_documents, "top_k": top_k})
    # metrics, documents, graph = api_get("/metrics"), api_get("/documents"), api_get("/knowledge-graph")
    metrics = api_get("/metrics")

    documents = api_get("/documents")

    selected = ",".join(selected_documents)

    graph = api_get(
        f"/knowledge-graph?selected_documents={selected}"
    )
    # Surface graph-linked entities when a text chunk has no direct NER labels.
    if not answer.get("entities"):
        entity_labels = [node.get("label", node.get("id")) for node in graph.get("nodes", [])]
        answer["entities"] = [label for label in entity_labels if label][:12]
    return render_template("intelligence.html", **layout_context(
        page_title="Industrial Intelligence", metrics=metrics, documents=documents, graph=graph, active="intelligence", active_tab="assistant", answer=answer, question=question, selected_documents=selected_documents, top_k=top_k,
    ))


@app.post("/maintenance")
@safe_api()
def maintenance():
    equipment_tag = request.form.get("equipment_tag", "").strip()
    if not equipment_tag:
        flash("Enter an equipment tag.", "error")
        return redirect(url_for("intelligence") + "#asset")
    maintenance_result = api_post("/maintenance", {"equipment_tag": equipment_tag})
    lessons = api_get("/lessons")
    metrics, documents, graph = api_get("/metrics"), api_get("/documents"), api_get("/knowledge-graph")
    # Surface graph-linked entities when a text chunk has no direct NER labels.
    return render_template("intelligence.html", **layout_context(
        page_title="Industrial Intelligence", metrics=metrics, documents=documents, graph=graph,
        active="intelligence", active_tab="asset", maintenance=maintenance_result, lessons=lessons, equipment_tag=equipment_tag,
    ))


@app.post("/compliance")
@safe_api()
def compliance():
    standard = request.form.get("standard", "Factory Act, OISD, PESO, ISO 9001").strip()
    compliance_result = api_post("/compliance", {"standard": standard})
    metrics, documents, graph = api_get("/metrics"), api_get("/documents"), api_get("/knowledge-graph")
    # Surface graph-linked entities when a text chunk has no direct NER labels.
    return render_template("intelligence.html", **layout_context(
        page_title="Industrial Intelligence", metrics=metrics, documents=documents, graph=graph,
        active="intelligence", active_tab="compliance", compliance=compliance_result, standard=standard,
    ))


@app.get("/creator")
def creator():
    return render_template("creator.html", **layout_context(page_title="Creator", active="creator"))


@app.errorhandler(413)
def file_too_large(_error):
    flash("The upload is larger than the configured maximum size.", "error")
    return redirect(url_for("intelligence") + "#workspace")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("FLASK_DEBUG") == "1")
