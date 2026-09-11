# Mental Health App — Dataset & Model: Status + Review

## 1. What I could / couldn't fetch

My code-execution sandbox doesn't have network access, so I can't `curl`/`wget` files
straight into it. I *can* fetch individual web pages one at a time and pull back their
text — enough to verify a link is real and see what's actually on the other end, but
not to bulk-download large files for training. Practical takeaway: **you download the
dataset file yourself and upload it here** (like you did with the zip) — then I can
load, clean, and train on it directly. That's the fastest reliable path.

What I verified by fetching the pages:

- **MindSET** — the link is *only the paper* (ACL Anthology / LREC 2026). The abstract
  confirms it's a real, large (13M+ post) benchmark, but the page has no dataset
  download link. You'd need to check the authors' GitHub/HuggingFace (not linked from
  this page) — worth emailing the authors if it's not public yet. Don't treat this one
  as "ready to download."
- **Preprocessed Conversational Mental Health Dataset (Zenodo)** — confirmed real and
  open. Direct file: `Classification_dataset.csv` (3.4 MB, CC-BY-4.0). I previewed the
  actual rows. Important flag: **this is raw, unfiltered first-person Reddit content**
  about anxiety/depression, and a meaningful share of it touches on self-harm and
  suicidal ideation. It's a legitimate research dataset, but this is not clean
  "labeled sentiment" data — it's real people's crisis writing. That has real
  implications for how (and whether) you use it — see §3.
- **MiniRecommendation (Hugging Face)** — confirmed real, small (400 rows), viewable
  in the dataset viewer. Good fit for the recommendation stage, but it's clearly
  synthetic/templated (repeated phrasing patterns), so treat it as a scaffold to
  extend, not a finished corpus.

I didn't re-verify every remaining link (Neurai-VN, VoiceDiaryMood, BRIDGE, Nepal
ARMS, DoseMate) — happy to if you want, but the two "restricted" ones will need a
formal data use agreement either way, so there's nothing to download there yet.

## 2. What I actually trained

Your uploaded zip already had a full pipeline (`generate_dataset.py` → `dataset.csv`
→ `train_model.py` → `model.pkl`/`vectorizer.pkl`) for a 6-class emotion classifier
(anger, anxiety, depression, happy, neutral, stress) meant to back the Streamlit demo
in `app.py`. I re-ran it as-is:

- 480 rows, 80 per class, TF-IDF (uni+bigrams) → compared Logistic Regression /
  Linear SVM / Naive Bayes with 5-fold CV.
- **Result: 100% accuracy, every model, every class.**

That number is a red flag, not a win. The dataset is generated from a small set of
sentence templates per class (`generate_dataset.py`), so train and test rows share
near-identical phrasing — the model is matching templates, not learning to
generalize to how real people actually write. It will look perfect in this repo and
fail the moment a real user types something that doesn't fit the template shape.

## 3. Where I'd push back on the plan

- **Don't wire a classifier like this straight into "risk detection."** The stage-1
  dataset (Zenodo) contains real self-harm/suicidal content; a demo classifier that's
  never seen that distribution will misclassify exactly the messages that matter most.
  Risk detection in a real product needs: a model trained and validated on data that
  actually includes at-risk language, a human-reviewed escalation path (not just an
  in-app message), and sign-off from a clinician — not just good test-set accuracy.
- **Treat "100% accuracy" as a bug to fix, not a milestone.** Before trusting any
  version of this model, test it on sentences you write yourself in plain language,
  not sampled from the same templates that generated the training data.
- **MindSET may not be usable yet.** Don't block the roadmap on it — confirm the real
  download path before planning around its 13M posts.
- **License/attribution housekeeping**: Zenodo dataset is CC-BY-4.0 (cite Diwakar &
  Raj), MiniRecommendation is on Hugging Face (check its card for terms before
  commercial use — recommendation datasets are sometimes research-only even when
  downloadable).

## 4. Suggested next step

Download `Classification_dataset.csv` from Zenodo yourself and upload it here — I'll
clean it, build a proper train/val/test split with realistic (non-templated) examples
mixed in, and retrain so the accuracy number actually means something. Same offer for
the MiniRecommendation parquet if you want the recommendation-stage model extended
beyond its current 400 rows.
