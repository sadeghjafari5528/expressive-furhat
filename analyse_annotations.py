import pandas as pd


ANNOTATOR_COLUMNS = [
	"Annotator1_Response",
	"Annotator2_Response",
	"Annotator3_Response",
    "Annotator4_Response",
    "Annotator5_Response",
]

GROUND_TRUTH_PATH = "ground_truth.csv"
PROLIFIC_DATA_PATH = "prolific_result.csv"

EMOTIONS_7 = [
	"disappointment",
	"disgust",
	"confusion",
	"neutral",
	"contentment",
	"joy",
	"surprise",
]

ground_truth_df = pd.read_csv(GROUND_TRUTH_PATH)
prolific_df = pd.read_csv(PROLIFIC_DATA_PATH)

video_ids = list(range(56))
attention_test_ids = [56]
text_ids = list(range(57, 65))

print("Ground truth columns:", ground_truth_df.columns)
print("Prolific data columns:", prolific_df.columns)


def normalize_label(x):
	if pd.isna(x):
		return None
	label = str(x).strip().lower()
	return label if label else None


def fleiss_kappa(counts_df, n_raters):
	# counts_df: rows are items, columns are categories, values are counts per category per item
	N = len(counts_df)
	if N == 0:
		return float("nan")

	P_i = (counts_df.pow(2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
	P_bar = P_i.mean()

	p_j = counts_df.sum(axis=0) / (N * n_raters)
	P_e_bar = (p_j.pow(2)).sum()

	denominator = 1 - P_e_bar
	if denominator == 0:
		return float("nan")

	return (P_bar - P_e_bar) / denominator


# Filter prolific rows for text_ids
prolific_df = prolific_df.copy()
prolific_df["id"] = pd.to_numeric(prolific_df["id"], errors="coerce")
text_df = prolific_df[prolific_df["id"].isin(text_ids)].copy()

# Normalize annotator responses
for col in ANNOTATOR_COLUMNS:
	text_df[col] = text_df[col].apply(normalize_label)

# Build category list and per-item counts for IAA
all_labels = pd.unique(text_df[ANNOTATOR_COLUMNS].values.ravel("K"))
categories = sorted([c for c in all_labels if c is not None])

if len(text_df) == 0 or len(categories) == 0:
	print("No valid text_id samples or annotations found in prolific data.")
else:
	counts_rows = []
	for _, row in text_df[ANNOTATOR_COLUMNS].iterrows():
		vc = row.value_counts(dropna=True)
		counts_rows.append([int(vc.get(cat, 0)) for cat in categories])

	counts_df = pd.DataFrame(counts_rows, columns=categories)
	n_raters = len(ANNOTATOR_COLUMNS)

	# IAA metrics
	row_majority = counts_df.max(axis=1)
	row_agreement = row_majority / n_raters
	mean_agreement = row_agreement.mean()
	kappa = fleiss_kappa(counts_df, n_raters)

	print("\n=== IAA for text_ids in prolific data ===")
	print(f"Number of text samples: {len(text_df)}")
	print(f"Mean per-item agreement: {mean_agreement:.4f}")
	print(f"Fleiss' kappa: {kappa:.4f}")

	# Per-sample emotion dictionary (always shows all 7 emotions)
	print("\n=== Per-sample emotion counts (text_ids) ===")
	for _, sample in text_df.iterrows():
		sample_counts = (
			sample[ANNOTATOR_COLUMNS]
			.value_counts(dropna=True)
			.reindex(EMOTIONS_7, fill_value=0)
			.astype(int)
			.to_dict()
		)
		print(f"id={int(sample['id'])}: {sample_counts}")

