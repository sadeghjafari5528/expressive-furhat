
from pathlib import Path
import mimetypes

import pandas as pd
from flask import Flask, abort, send_file


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "ground_truth.csv"

app = Flask(__name__)


def load_ground_truth() -> pd.DataFrame:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing CSV file: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    required_columns = {"id", "text"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns in ground_truth.csv: {missing}")

    df = df.copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype(int)
    return df


GROUND_TRUTH_DF = load_ground_truth()


@app.get("/")
def health() -> str:
    return "Web service is running. Use /<id> to download a file."


@app.get("/<int:item_id>")
def download_file(item_id: int):
    row = GROUND_TRUTH_DF.loc[GROUND_TRUTH_DF["id"] == item_id]
    if row.empty:
        abort(404, description="ID not found")

    file_ref = str(row.iloc[0]["text"]).strip()
    file_path = (BASE_DIR / file_ref).resolve()

    try:
        file_path.relative_to(BASE_DIR.resolve())
    except ValueError:
        abort(403, description="Invalid file path")

    if not file_path.is_file():
        abort(404, description="File not found")

    guessed_mime, _ = mimetypes.guess_type(str(file_path))
    return send_file(file_path, as_attachment=False, mimetype=guessed_mime)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

