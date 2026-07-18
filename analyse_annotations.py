import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ANNOTATOR_COLUMNS = [
	"Annotator1_Response",
	"Annotator2_Response",
	"Annotator3_Response",
    "Annotator4_Response",
    "Annotator5_Response",
]

GROUND_TRUTH_PATH = "data/ground_truth.csv"
PROLIFIC_DATA_PATH = "data/prolific_result.csv"

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

# Draw text emotion distributions as circles (pie charts)
text_ids_to_plot = sorted(text_df["id"].dropna().astype(int).unique())[:7]
fig, axes = plt.subplots(1, len(text_ids_to_plot), figsize=(2.6 * len(text_ids_to_plot), 3.2))
if len(text_ids_to_plot) == 1:
    axes = [axes]

colors_text = {label: plt.cm.tab10(i % 10) for i, label in enumerate(EMOTIONS_7)}

for idx, tid in enumerate(text_ids_to_plot):
    ax = axes[idx]
    sample_row = text_df[text_df["id"] == tid].iloc[0]
    dist = (
        sample_row[ANNOTATOR_COLUMNS]
        .value_counts(dropna=True)
        .reindex(EMOTIONS_7, fill_value=0)
        .astype(int)
    )
    values = dist.tolist()

    if sum(values) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=8)
    else:
        ax.pie(
            values,
            colors=[colors_text[label] for label in EMOTIONS_7],
            startangle=90,
            counterclock=False,
            wedgeprops={'linewidth': 0.3, 'edgecolor': 'white'}
        )

    ax.set_aspect('equal')
    ax.set_title(f"story {idx + 1}", fontsize=12, y=0.93)
    ax.set_xticks([])
    ax.set_yticks([])

legend_handles_text = [Patch(facecolor=colors_text[label], label=label) for label in EMOTIONS_7]
fig.legend(
    handles=legend_handles_text,
    loc='lower center',
    ncol=len(EMOTIONS_7),
    fontsize=16,
    frameon=False
)
plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.savefig("results/text_emotion.png", dpi=300)
plt.close(fig)


# Calculate IAA for video_ids
video_df = prolific_df[prolific_df["id"].isin(video_ids)].copy()
for col in ANNOTATOR_COLUMNS:
    video_df[col] = video_df[col].apply(normalize_label)

all_labels_video = pd.unique(video_df[ANNOTATOR_COLUMNS].values.ravel("K"))
categories_video = sorted([c for c in all_labels_video if c is not None])

if len(video_df) == 0 or len(categories_video) == 0:
	print("No valid video_id samples or annotations found in prolific data.")
else:
	counts_rows_video = []
	for _, row in video_df[ANNOTATOR_COLUMNS].iterrows():
		vc = row.value_counts(dropna=True)
		counts_rows_video.append([int(vc.get(cat, 0)) for cat in categories_video])

	counts_df_video = pd.DataFrame(counts_rows_video, columns=categories_video)
	n_raters_video = len(ANNOTATOR_COLUMNS)

	# IAA metrics
	row_majority_video = counts_df_video.max(axis=1)
	row_agreement_video = row_majority_video / n_raters_video
	mean_agreement_video = row_agreement_video.mean()
	kappa_video = fleiss_kappa(counts_df_video, n_raters_video)

	print("\n=== IAA for video_ids in prolific data ===")
	print(f"Number of video samples: {len(video_df)}")
	print(f"Mean per-item agreement: {mean_agreement_video:.4f}")
	print(f"Fleiss' kappa: {kappa_video:.4f}")


# Calculate Emotion Matrix for video_ids
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

STORIES = [
	'story 1\n disappointment',
	'story 2\n disgust',
	'story 3\n confusion',
	'story 4\n neutral',
	'story 5\n contentment',
	'story 6\n joy',
	'story 7\n surprise',
]

EMOTIONS = [
    'disappointment',
    'disgust',
    'confusion',
    'neutral',
    'contentment',
    'joy',
    'surprise',
    'default',
]

DISTRIBUTION_LABELS = EMOTIONS_7

emotion_matrix = pd.DataFrame(index=EMOTIONS, columns=STORIES, dtype=object)

for col_idx, story in enumerate(STORIES):
	for row_idx, emotion in enumerate(EMOTIONS):
		sample_id = col_idx * len(EMOTIONS) + row_idx
		row = video_df[video_df['id'] == sample_id]

		if row.empty:
			counts = {label: 0 for label in DISTRIBUTION_LABELS}
		else:
			counts = (
				row.iloc[0][ANNOTATOR_COLUMNS]
				.value_counts(dropna=True)
				.reindex(DISTRIBUTION_LABELS, fill_value=0)
				.astype(int)
				.to_dict()
			)

		emotion_matrix.at[emotion, story] = counts

# Draw as matrix of circles (pie charts)
fig, axes = plt.subplots(
	len(EMOTIONS),
	len(STORIES),
	figsize=(2.2 * len(STORIES), 2.2 * len(EMOTIONS))
)

if len(EMOTIONS) == 1 and len(STORIES) == 1:
	axes = [[axes]]
elif len(EMOTIONS) == 1:
	axes = [list(axes)]
elif len(STORIES) == 1:
	axes = [[ax] for ax in axes]

colors = {label: plt.cm.tab10(i % 10) for i, label in enumerate(DISTRIBUTION_LABELS)}

for i, emotion in enumerate(EMOTIONS):
	for j, story in enumerate(STORIES):
		ax = axes[i][j]
		dist = emotion_matrix.at[emotion, story]
		values = [dist[label] for label in DISTRIBUTION_LABELS]

		if sum(values) == 0:
			ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=7)
		else:
			ax.pie(
				values,
				colors=[colors[label] for label in DISTRIBUTION_LABELS],
				startangle=90,
				counterclock=False,
				wedgeprops={'linewidth': 0.3, 'edgecolor': 'white'}
			)

		ax.set_aspect('equal')
		ax.set_xticks([])
		ax.set_yticks([])

		if i == 0:
			ax.set_title(story, fontsize=12)
		if j == 0:
			ax.set_ylabel(emotion, fontsize=12)

legend_handles = [Patch(facecolor=colors[label], label=label) for label in DISTRIBUTION_LABELS]
fig.legend(
	handles=legend_handles,
	loc='lower center',
	ncol=len(DISTRIBUTION_LABELS),
	fontsize=16,
	frameon=False
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("results/emotion_matrix.png", dpi=300)