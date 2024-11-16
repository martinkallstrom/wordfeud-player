"""FastAPI application for Wordfeud board recognition."""

import os
import asyncio
from typing import Dict, Optional
from pathlib import Path
from tempfile import NamedTemporaryFile
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from main import WordfeudPlayer

# Initialize FastAPI app
app = FastAPI(
    title="Wordfeusk Board Analyzer",
    description="API for analyzing Wordfeud board screenshots",
    version="1.0.0",
)

# Create static and templates directories if they don't exist
STATIC_DIR = Path("static")
TEMPLATES_DIR = Path("templates")
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize the Wordfeud player
player = WordfeudPlayer()

# Create a dictionary to store background tasks
tasks: Dict[str, Dict] = {}


class AnalysisResponse(BaseModel):
    """Response model for board analysis."""

    task_id: str
    status: str
    result: Optional[Dict] = None
    error: Optional[str] = None


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


async def process_image(task_id: str, temp_file_path: str):
    """Process the image in the background."""
    try:
        # Analyze the board image
        result = player.analyze_board_image(temp_file_path)

        # Convert numpy types to Python native types
        result = convert_numpy_types(result)

        # Save visualization to static directory if present
        if "visualization" in result:
            vis_path = STATIC_DIR / f"visualization_{task_id}.png"
            result["visualization"].save(vis_path)
            result["visualization_url"] = f"/static/visualization_{task_id}.png"
            del result["visualization"]  # Remove PIL image from result

        # Update task status
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["result"] = result

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_file_path)
        except:
            pass


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_board(
    background_tasks: BackgroundTasks, file: UploadFile = File(...)
) -> Dict:
    """
    Analyze a Wordfeud board screenshot.

    Args:
        file: Uploaded image file

    Returns:
        Dict containing task ID and initial status
    """
    # Generate unique task ID
    task_id = f"task_{len(tasks) + 1}"

    try:
        # Create temporary file
        with NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Initialize task
        tasks[task_id] = {"status": "processing", "result": None, "error": None}

        # Start background processing
        background_tasks.add_task(process_image, task_id, temp_file_path)

        return {"task_id": task_id, "status": "processing"}

    except Exception as e:
        if task_id in tasks:
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["error"] = str(e)
        return {"task_id": task_id, "status": "failed", "error": str(e)}


@app.get("/status/{task_id}", response_model=AnalysisResponse)
async def get_status(task_id: str) -> Dict:
    """
    Get the status of an analysis task.

    Args:
        task_id: Task ID to check

    Returns:
        Dict containing task status and result if completed
    """
    if task_id not in tasks:
        return {"task_id": task_id, "status": "not_found"}

    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "result": task.get("result"),
        "error": task.get("error"),
    }


@app.get("/", response_class=HTMLResponse)
async def upload_form(request: Request):
    """Serve the upload form."""
    return templates.TemplateResponse("upload.html", {"request": request})
