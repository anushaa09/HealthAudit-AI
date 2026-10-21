# HealthAudit-AI

**Intelligent Healthcare Claims Auditing and Decision Support System**

HealthAudit-AI predicts whether a healthcare claim should be approved or denied, explains the decision in plain language, suggests what could change a denial, and records a tamper-evident proof of every decision on the Ethereum Sepolia test network.

**Authors:** Nandita Gaur, Anusha Sarkar

> This is a university decision-support prototype. It is not a production medical or insurance adjudication system.

## What it does

1. **Predicts** Approve / Deny from engineered beneficiary-level features (XGBoost classifier).
2. **Explains** the decision with SHAP, showing the top five contributing factors.
3. **Suggests recourse** with DiCE counterfactuals: for a denied claim, how much lower the length of the hospital stay or the inpatient / outpatient cost would have to be for it to be approved. Only those three things are ever changed, and only downwards. Age, sex, location, coverage, existing conditions, insurer-side amounts and claim counts are held fixed. Each suggestion is rounded to tidy figures and re-checked against the trained model before it is shown.
4. **Audits** every decision: stores it in Supabase, hashes it (SHA-256), and anchors the hash in a Solidity smart contract on Sepolia. The anchored hash can be verified later.

## Architecture

| Layer | Technology | Hosting |
| --- | --- | --- |
| Frontend | React, Vite | Vercel |
| API | FastAPI, Uvicorn | Render |
| ML | XGBoost pipeline, SHAP, DiCE | (inside the API) |
| Audit store | Supabase | Supabase |
| Integrity | SHA-256 + Solidity contract | Ethereum Sepolia |

Decision flow: frontend -> `POST /predict` -> prediction + SHAP -> hash -> Supabase insert -> Sepolia anchor -> transaction details saved -> response. A denied claim then calls `POST /recourse`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| POST | `/predict` | Prediction, SHAP factors, audit record and blockchain anchor |
| POST | `/recourse` | Counterfactual suggestions for a denied claim |
| POST | `/anchor` | Anchor a decision hash on Sepolia |
| GET | `/anchor/{decision_id}` | Read an anchored hash back from the chain |
| GET | `/health` | Liveness check |

Interactive docs are served at `/docs`.

## Run locally

Use **Python 3.12** (versions in `requirements.txt` are pinned to match the saved model files).

```bash
python3.12 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --reload     # http://127.0.0.1:8000/docs
```

Frontend:

```bash
cd frontend
npm install
npm run dev                       # set VITE_API_URL=http://127.0.0.1:8000 in frontend/.env.local to use a local API
```

Secrets are read from `.env` (project root: `SUPABASE_URL`, `SUPABASE_SECRET_KEY`) and `blockchain/.env` (`SEPOLIA_RPC_URL`, `PRIVATE_KEY`, `CONTRACT_ADDRESS`, `CONTRACT_ABI`). Both files are git-ignored. Never commit them.

## Deployment

- **Render (API):** build `pip install -r requirements.txt`, start `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- **Vercel (frontend):** root directory `frontend`, framework Vite.

## Recourse reference data

`/recourse` needs a small background dataset (`data/processed/dice_background.parquet`) so DiCE knows the allowed ranges for each feature. The full training split is large and git-ignored, so this small file is committed instead.

- With the real training split: `python -m src.explainability.build_dice_background`
- Without it (fallback): `python -m src.explainability.build_dice_background --from-model` builds a **model-derived reference set** from the trained model's own stored statistics. It is not real patient data, the file is stamped as such, and the API logs a warning when it is used. Suggestions are still checked against the real trained model.

## Limitations

- The model was trained on **proxy labels** produced by weak supervision (see `src/preprocessing/label_generator.py`). Its high scores partly reflect the model recovering the labelling rules, not real-world claim-decision accuracy. See `outputs/models/xgboost_claim_classifier_metadata.json`.
- The reported confidence is the distance of the predicted probability from 0.5, scaled to 0-1. It is not a calibrated probability.
- Blockchain anchoring proves a record was not altered after it was anchored. It does not prove the prediction is correct or fair.
- Fairness, calibration and security review are still to be done before any real-world use.
