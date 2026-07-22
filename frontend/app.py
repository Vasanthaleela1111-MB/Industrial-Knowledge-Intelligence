import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

import Services as svc

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "industrialmind-dev-secret")


# ============================================================
# HOME / HEADER (health check + navigation)
# ============================================================
@app.route("/")
def index():
    return redirect(url_for("project_documentation"))


@app.route("/documentation")
def project_documentation():
    health, last_error = svc.get_health()
    if health is None:
        return render_template("error.html", error=str(last_error))
    return render_template("documentation.html", health=health, active="documentation")


# ============================================================
# INDUSTRIAL INTELLIGENCE
# ============================================================
@app.route("/intelligence")
def intelligence():
    health, last_error = svc.get_health()
    if health is None:
        return render_template("error.html", error=str(last_error))

    metrics = svc.get_metrics()
    documents = svc.get_documents()

    return render_template(
        "intelligence.html",
        health=health,
        metrics=metrics,
        documents=documents,
        active="intelligence",
    )


@app.route("/intelligence/clear-kb", methods=["POST"])
def clear_knowledge_base():
    svc.api_post("/clear", {})
    svc.clear_data_cache()
    flash("Knowledge base cleared successfully.", "success")
    return redirect(url_for("intelligence", tab="workspace"))


@app.route("/intelligence/upload", methods=["POST"])
def upload_documents():
    files = request.files.getlist("files")
    files = [f for f in files if f and f.filename]

    total = len(files)
    failed = []

    for f in files:
        try:
            svc.upload_document(f)
        except Exception as exc:
            failed.append(f.filename)
            flash(f"{f.filename}: {exc}", "error")

    if failed:
        flash(f"Failed files: {', '.join(failed)}", "warning")

    svc.clear_data_cache()
    if total:
        flash(f"{total} document(s) successfully processed.", "success")
        flash("Document ingestion completed.", "info")

    return redirect(url_for("intelligence", tab="workspace"))


@app.route("/intelligence/ask", methods=["POST"])
def ask_ai():
    question = request.form.get("question", "").strip()
    top_k = int(request.form.get("top_k", 5))
    selected_documents = request.form.getlist("selected_documents")

    health, last_error = svc.get_health()
    metrics = svc.get_metrics()
    documents = svc.get_documents()

    response = None
    graph = None
    error = None

    if question and selected_documents:
        try:
            response = svc.api_post(
                "/ask",
                {
                    "question": question,
                    "top_k": top_k,
                    "selected_documents": selected_documents,
                },
            )
            # graph = svc.api_get("/knowledge-graph")
            selected = ",".join(selected_documents)

            graph = svc.api_get(
                f"/knowledge-graph?selected_documents={selected}"
            )
        except Exception as exc:
            error = str(exc)

    return render_template(
        "intelligence.html",
        health=health,
        metrics=metrics,
        documents=documents,
        active="intelligence",
        active_tab="assistant",
        question=question,
        top_k=top_k,
        selected_documents=selected_documents,
        response=response,
        graph=graph,
        ask_error=error,
    )


@app.route("/intelligence/asset-analyze", methods=["POST"])
def asset_analyze():
    equipment_tag = request.form.get("equipment_tag", "").strip()

    health, last_error = svc.get_health()
    metrics = svc.get_metrics()
    documents = svc.get_documents()

    maintenance = None
    lessons = None
    error = None

    if equipment_tag:
        try:
            maintenance = svc.api_post("/maintenance", {"equipment_tag": equipment_tag})
            lessons = svc.api_get("/lessons")
        except Exception as exc:
            error = str(exc)

    return render_template(
        "intelligence.html",
        health=health,
        metrics=metrics,
        documents=documents,
        active="intelligence",
        active_tab="asset",
        equipment_tag=equipment_tag,
        maintenance=maintenance,
        lessons=lessons,
        asset_error=error,
    )


@app.route("/intelligence/compliance-assess", methods=["POST"])
def compliance_assess():
    standard = request.form.get(
        "ISO 9001"
    ).strip()

    health, last_error = svc.get_health()
    metrics = svc.get_metrics()
    documents = svc.get_documents()

    compliance = None
    error = None

    try:
        compliance = svc.api_post("/compliance", {"standard": standard})
    except Exception as exc:
        error = str(exc)

    return render_template(
        "intelligence.html",
        health=health,
        metrics=metrics,
        documents=documents,
        active="intelligence",
        active_tab="compliance",
        standard=standard,
        compliance=compliance,
        compliance_error=error,
    )


# ============================================================
# CREATOR
# ============================================================
@app.route("/creator")
def creator():
    health, last_error = svc.get_health()
    return render_template("creator.html", health=health, active="creator")


# ============================================================
# JSON API passthroughs (optional, useful for AJAX/testing)
# ============================================================
@app.route("/api/metrics")
def api_metrics():
    return jsonify(svc.get_metrics())


@app.route("/api/documents")
def api_documents():
    return jsonify(svc.get_documents())


@app.route("/api/knowledge-graph")
def api_knowledge_graph():
    return jsonify(svc.api_get("/knowledge-graph"))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
