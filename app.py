import csv
import io
import json
import logging
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from utils.database import (
    fetch_average_confidence,
    fetch_category_counts,
    fetch_recent_results,
    fetch_severity_counts,
    fetch_total_results,
    initialize_database,
    save_detection_result,
    save_detection_results,
    update_review_status,
)
from utils.model_utils import MAX_SEQUENCE_LENGTH, load_artifacts
from utils.preprocess import clean_text
from utils.severity import (
    apply_rule_based_override,
    build_suggestion,
    calculate_fuzzy_severity,
    map_category_to_severity,
)

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key="cyberbullying-secret-key")
app.mount("/static", StaticFiles(directory=Path(__file__).resolve().parent / "static"), name="static")

templates = Jinja2Templates(directory=Path(__file__).resolve().parent / "templates")

METRICS_PATH = Path(__file__).resolve().parent / "models" / "performance_metrics.json"
LOGGER = logging.getLogger(__name__)
_ARTIFACTS = None

try:
    _ARTIFACTS = load_artifacts()
except FileNotFoundError:
    LOGGER.warning("Model artifacts are not available yet; predictions will fail until training is complete.")


# ---------------------------------------------------------------------------
# Flash-message helpers (uses Starlette session middleware)
# ---------------------------------------------------------------------------

def flash(request: Request, message: str, category: str = "info") -> None:
    """Store a flash message in the session, mimicking Flask's flash()."""
    if "_flashes" not in request.session:
        request.session["_flashes"] = []
    request.session["_flashes"].append({"category": category, "message": message})


def get_flashed_messages(request: Request) -> list[dict]:
    """Pop all flash messages from the session."""
    return request.session.pop("_flashes", [])


# ---------------------------------------------------------------------------
# Template helper – avoids the "unhashable type: dict" Jinja2 cache bug
# ---------------------------------------------------------------------------

def render(request: Request, template_name: str, context: dict) -> HTMLResponse:
    """Render a Jinja2 template, working around Starlette/Jinja2 cache-key issues."""
    template = templates.env.get_template(template_name)
    context["request"] = request
    html = template.render(context)
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_performance_metrics():
    if not METRICS_PATH.exists():
        return None
    with METRICS_PATH.open(encoding="utf-8") as metrics_file:
        return json.load(metrics_file)


def predict_messages(messages):
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = load_artifacts()
    model, tokenizer, labels, vectorizer, classifier, lstm_weight = _ARTIFACTS
    cleaned_messages = [clean_text(message) for message in messages]
    if model is None:
        probabilities = classifier.predict_proba(vectorizer.transform(cleaned_messages))
    else:
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        sequences = pad_sequences(
            tokenizer.texts_to_sequences(cleaned_messages),
            maxlen=MAX_SEQUENCE_LENGTH,
            padding="post",
            truncating="post",
        )
        probabilities = model.predict(sequences, verbose=0)
        if vectorizer is not None and classifier is not None:
            companion_probabilities = classifier.predict_proba(vectorizer.transform(cleaned_messages))
            probabilities = lstm_weight * probabilities + (1 - lstm_weight) * companion_probabilities

    results = []
    for message, message_probabilities in zip(messages, probabilities):
        prediction = labels[int(message_probabilities.argmax())]
        confidence = float(message_probabilities.max())
        severity, _ = calculate_fuzzy_severity(message, prediction, confidence)
        if severity == "Safe":
            severity = map_category_to_severity(prediction)
        prediction, severity, overridden = apply_rule_based_override(message, prediction, severity)
        category_description = {
            "high": "hate-speech-related language",
            "medium": "offensive language",
            "not_cyberbullying": "no cyberbullying detected",
        }.get(prediction, f"{prediction} content")
        results.append({
            "message": message,
            "predicted_category": prediction,
            "severity": severity,
            "confidence": confidence,
            "suggestion": build_suggestion(severity),
            "explanation": f"Detected {category_description}; assigned {severity} severity at {confidence:.0%} confidence.",
            "overridden": overridden,
        })
    return results


# ---------------------------------------------------------------------------
# Startup event – initialize the database once
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup_event():
    initialize_database()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    severity_counts = fetch_severity_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    overview_cards = [
        {"label": "Total analyses", "value": total_messages},
        {"label": "Safe", "value": severity_counts.get("Safe", 0)},
        {"label": "High risk", "value": severity_counts.get("High", 0)},
        {"label": "Average confidence", "value": f"{average_confidence * 100:.1f}%"},
    ]
    return render(request, "index.html", {
        "overview_cards": overview_cards,
        "total_messages": total_messages,
        "severity_counts": severity_counts,
        "average_confidence": average_confidence,
        "flashed_messages": get_flashed_messages(request),
        "result": None,
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, message: str = Form("")):
    message = message.strip()
    if not message:
        flash(request, "Please enter a message to analyze.", "error")
        return RedirectResponse(url="/", status_code=303)

    try:
        result = predict_messages([message])[0]
    except FileNotFoundError:
        flash(request, "Train the model first so the app can make predictions.", "error")
        return RedirectResponse(url="/", status_code=303)

    save_detection_result(
        result["message"], result["predicted_category"], result["severity"],
        result["confidence"], result["suggestion"],
    )

    severity_counts = fetch_severity_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    overview_cards = [
        {"label": "Total analyses", "value": total_messages},
        {"label": "Safe", "value": severity_counts.get("Safe", 0)},
        {"label": "High risk", "value": severity_counts.get("High", 0)},
        {"label": "Average confidence", "value": f"{average_confidence * 100:.1f}%"},
    ]

    return render(request, "index.html", {
        "overview_cards": overview_cards,
        "total_messages": total_messages,
        "severity_counts": severity_counts,
        "average_confidence": average_confidence,
        "flashed_messages": [],
        "result": {
            "message": message,
            "predicted_category": result["predicted_category"],
            "severity": result["severity"],
            "confidence": round(result["confidence"] * 100, 2),
            "suggestion": result["suggestion"],
            "explanation": result["explanation"],
            "overridden": result["overridden"],
        },
    })


@app.get("/bulk", response_class=HTMLResponse)
async def bulk_analysis_get(request: Request):
    return render(request, "bulk.html", {
        "results": None,
        "summary": None,
        "flashed_messages": get_flashed_messages(request),
    })


@app.post("/bulk", response_class=HTMLResponse)
async def bulk_analysis_post(request: Request, file: UploadFile = File(None)):
    results = None
    summary = None
    error_message = None

    if file is None or not file.filename:
        error_message = "Choose a CSV file to analyze."
    elif not file.filename.lower().endswith((".csv", ".zip", ".txt")):
        error_message = "Bulk analysis accepts TXT, CSV, or ZIP files containing a CSV."
    else:
        try:
            uploaded_bytes = await file.read()
            filename = file.filename.lower()
            if filename.endswith(".txt"):
                messages = [
                    line.strip()
                    for line in uploaded_bytes.decode("utf-8-sig").splitlines()
                    if line.strip()
                ]
                rows = [{"message": message} for message in messages]
                skipped_rows = 0
            else:
                if filename.endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(uploaded_bytes)) as archive:
                        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
                        if not csv_names:
                            raise ValueError("ZIP file does not contain a CSV file.")
                        content = archive.read(csv_names[0]).decode("utf-8-sig")
                else:
                    content = uploaded_bytes.decode("utf-8-sig")
                rows = list(csv.DictReader(io.StringIO(content)))
                message_column = next(
                    (column for column in ("message", "text", "tweet", "tweet_text", "comment_text", "content") if rows and column in rows[0]),
                    None,
                )
                if message_column is None and rows:
                    raise ValueError("CSV must contain a message, text, tweet, tweet_text, comment_text, or content column.")
                messages = [row.get(message_column, "").strip() for row in rows]
                skipped_rows = sum(not message for message in messages)
                messages = [message for message in messages if message]
            if not messages:
                raise ValueError("The selected message column does not contain any non-empty messages.")
            if len(messages) > 200000:
                raise ValueError("Please upload at most 200,000 messages per batch.")
            results = []
            for start in range(0, len(messages), 5000):
                batch_results = predict_messages(messages[start:start + 5000])
                results.extend(batch_results)
                save_detection_results(batch_results)
            summary = {
                "total": len(results),
                "input_rows": len(rows),
                "skipped_rows": skipped_rows,
                "safe": sum(result["severity"] == "Safe" for result in results),
                "low": sum(result["severity"] == "Low" for result in results),
                "medium": sum(result["severity"] == "Medium" for result in results),
                "high": sum(result["severity"] == "High" for result in results),
            }
        except (UnicodeDecodeError, ValueError, zipfile.BadZipFile) as error:
            error_message = str(error)
        except FileNotFoundError:
            error_message = "Train the model first so the app can make predictions."
        except Exception:
            LOGGER.exception("Bulk analysis failed")
            error_message = "Bulk analysis could not be completed. Check the file format and try a smaller batch."

    flashed = get_flashed_messages(request)
    if error_message:
        flashed.append({"category": "error", "message": error_message})

    return render(request, "bulk.html", {
        "results": results,
        "summary": summary,
        "flashed_messages": flashed,
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(
    request: Request,
    severity: str = Query("All"),
    search: str = Query(""),
    review_status: str = Query("All"),
):
    search = search.strip()
    results = fetch_recent_results(
        limit=20,
        severity=severity,
        search=search or None,
        review_status=review_status,
    )
    severity_counts = fetch_severity_counts()
    category_counts = fetch_category_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    severity_chart_labels = ["Safe", "Low", "Medium", "High"]
    severity_chart_values = [severity_counts.get(label, 0) for label in severity_chart_labels]
    category_chart_labels = list(category_counts.keys())
    category_chart_values = list(category_counts.values())
    total_messages = sum(severity_counts.values())
    return render(request, "dashboard.html", {
        "results": results,
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "total_messages": total_messages,
        "average_confidence": average_confidence,
        "severity_chart_labels": severity_chart_labels,
        "severity_chart_values": severity_chart_values,
        "category_chart_labels": category_chart_labels,
        "category_chart_values": category_chart_values,
        "current_severity": severity,
        "current_search": search,
        "current_review_status": review_status,
        "performance_metrics": load_performance_metrics(),
    })


@app.post("/review/{result_id}")
async def review_result(request: Request, result_id: int, review_status: str = Form("Pending Review")):
    update_review_status(result_id, review_status)
    referer = request.headers.get("referer", "/dashboard")
    return RedirectResponse(url=referer, status_code=303)


@app.get("/export-csv")
async def export_csv(severity: str = Query("All"), search: str = Query("")):
    search = search.strip()
    results = fetch_recent_results(limit=1000, severity=severity, search=search or None)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "message", "predicted_category", "severity", "confidence", "suggestion", "review_status", "created_at"])

    for row in results:
        writer.writerow([
            row["id"],
            row["message"],
            row["predicted_category"],
            row["severity"],
            f"{row['confidence']:.4f}",
            row["suggestion"],
            row["review_status"],
            row["created_at"],
        ])

    csv_bytes = buffer.getvalue().encode("utf-8")
    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cyberbullying_results.csv"},
    )


@app.get("/report", response_class=HTMLResponse)
async def report(request: Request):
    severity_counts = fetch_severity_counts()
    category_counts = fetch_category_counts()
    total_messages = fetch_total_results()
    average_confidence = fetch_average_confidence()
    recent_results = fetch_recent_results(limit=5)

    highest_severity = max(severity_counts.items(), key=lambda item: item[1])[0] if severity_counts else "No data"
    top_category = max(category_counts.items(), key=lambda item: item[1])[0] if category_counts else "No data"

    return render(request, "report.html", {
        "total_messages": total_messages,
        "average_confidence": average_confidence,
        "severity_counts": severity_counts,
        "category_counts": category_counts,
        "recent_results": recent_results,
        "highest_severity": highest_severity,
        "top_category": top_category,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=5000)
