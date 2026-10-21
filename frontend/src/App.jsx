import { useState } from 'react';
import axios from 'axios';
import './App.css';

// Defaults to the deployed backend. For local testing, set VITE_API_URL in
// frontend/.env.local (e.g. VITE_API_URL=http://127.0.0.1:8000).
const API_BASE = import.meta.env.VITE_API_URL || 'https://healthaudit-ai.onrender.com';

// Turns an axios error into a readable string. The backend can return
// `detail` as a string, a validation-error list, or an object (for example
// when the audit record was saved but blockchain anchoring failed).
function getErrorMessage(err) {
  const detail = err.response?.data?.detail;
  if (Array.isArray(detail)) {
    return detail[0]?.msg || 'Some of the claim details could not be validated.';
  }
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') {
    const reference = detail.decision_id ? ` Reference: ${detail.decision_id}` : '';
    return `${detail.message || 'The claim service reported a problem.'}${reference}`;
  }
  if (err.response) return 'The claim service reported a problem. Please try again.';
  return 'We could not reach the claim service. Check that the backend is running.';
}

const CONDITIONS = [
  { key: 'alzheimer', label: "Alzheimer's / Dementia", severity: 2 },
  { key: 'chf', label: 'Heart Failure', severity: 4 },
  { key: 'kidney', label: 'Kidney Disease', severity: 4 },
  { key: 'cancer', label: 'Cancer', severity: 5 },
  { key: 'copd', label: 'COPD (lung disease)', severity: 3 },
  { key: 'depression', label: 'Depression', severity: 1 },
  { key: 'diabetes', label: 'Diabetes', severity: 2 },
  { key: 'ischemic', label: 'Heart Disease', severity: 4 },
  { key: 'osteoporosis', label: 'Osteoporosis', severity: 1 },
  { key: 'arthritis', label: 'Arthritis', severity: 1 },
  { key: 'stroke', label: 'Stroke History', severity: 4 },
];

const initialForm = {
  patientId: '',
  claimId: '',
  name: '',
  age: '',
  sex: '1',
  conditions: {},
  document: null,
  extractedCost: null,
  extractedDays: null,
  manualCost: '',
  manualDays: '',
  needsManualFallback: false,
};

function signedLog(x) {
  if (x === 0) return 0;
  return Math.sign(x) * Math.log1p(Math.abs(x));
}

async function extractFromPdf(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await window.pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    let fullText = '';
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      fullText += content.items.map((it) => it.str).join(' ') + ' ';
    }

    const dollarMatches = [...fullText.matchAll(/\$\s?([0-9][0-9,]*(?:\.[0-9]{2})?)/g)]
      .map((m) => parseFloat(m[1].replace(/,/g, '')))
      .filter((n) => !isNaN(n));
    const maxCost = dollarMatches.length > 0 ? Math.max(...dollarMatches) : null;

    const dayMatches = [...fullText.matchAll(/(\d+)\s*(day|days)\b/gi)]
      .map((m) => parseInt(m[1], 10))
      .filter((n) => !isNaN(n) && n < 365);
    const maxDays = dayMatches.length > 0 ? Math.max(...dayMatches) : null;

    return { cost: maxCost, days: maxDays };
  } catch (err) {
    console.error('PDF extraction failed:', err);
    return { cost: null, days: null };
  }
}

function estimateClaimFeatures(form) {
  const age = Number(form.age) || 50;
  const checkedConditions = CONDITIONS.filter((c) => form.conditions[c.key]);
  const chronicConditionCount = checkedConditions.length;
  const severitySum = checkedConditions.reduce((sum, c) => sum + c.severity, 0);
  const ageFactor = age > 70 ? 1.4 : age > 50 ? 1.1 : 0.8;

  const totalCost = form.extractedCost ?? (Number(form.manualCost) ||
    (500 + severitySum * 900) * ageFactor);

  const seriousCount = checkedConditions.filter((c) => c.severity >= 4).length;
  const inpatientDays = form.extractedDays ?? (Number(form.manualDays) ||
    (seriousCount > 0 ? seriousCount * 3 * ageFactor : 0));

  const inpatientVisits = inpatientDays > 0 ? Math.max(1, Math.round(seriousCount)) : 0;
  const outpatientVisits = Math.max(1, chronicConditionCount);
  const inpatientDiagnoses = inpatientVisits > 0 ? Math.max(1, Math.round(severitySum / 2)) : 0;
  const outpatientDiagnoses = Math.max(1, chronicConditionCount);

  const inpatientPayment = inpatientVisits > 0 ? totalCost * 0.7 : 0;
  const outpatientPayment = inpatientVisits > 0 ? totalCost * 0.3 : totalCost;
  const totalClaims = Math.max(inpatientVisits + outpatientVisits, 1);

  const b = (key) => (form.conditions[key] ? 1 : 0);

  return {
    BENE_ESRD_IND: 0,
    BENE_HI_CVRAGE_TOT_MONS: 12,
    BENE_SMI_CVRAGE_TOT_MONS: 12,
    BENE_HMO_CVRAGE_TOT_MONS: 0,
    PLAN_CVRG_MOS_NUM: 12,
    age: age,
    is_deceased: 0,
    SP_ALZHDMTA: b('alzheimer'),
    SP_CHF: b('chf'),
    SP_CHRNKIDN: b('kidney'),
    SP_CNCR: b('cancer'),
    SP_COPD: b('copd'),
    SP_DEPRESSN: b('depression'),
    SP_DIABETES: b('diabetes'),
    SP_ISCHMCHT: b('ischemic'),
    SP_OSTEOPRS: b('osteoporosis'),
    SP_RA_OA: b('arthritis'),
    SP_STRKETIA: b('stroke'),
    chronic_condition_count: chronicConditionCount,
    MEDREIMB_IP: inpatientPayment,
    BENRES_IP: 0,
    PPPYMT_IP: 0,
    MEDREIMB_OP: outpatientPayment,
    BENRES_OP: 0,
    PPPYMT_OP: 0,
    MEDREIMB_CAR: 0,
    BENRES_CAR: 0,
    PPPYMT_CAR: 0,
    total_inpatient_claims: inpatientVisits,
    distinct_inpatient_diagnosis_codes: inpatientDiagnoses,
    total_outpatient_claims: outpatientVisits,
    distinct_outpatient_diagnosis_codes: outpatientDiagnoses,
    total_inpatient_payment_amt_log: signedLog(inpatientPayment),
    total_outpatient_payment_amt_log: signedLog(outpatientPayment),
    total_inpatient_primary_payer_amt_log: 0,
    total_outpatient_primary_payer_amt_log: 0,
    total_inpatient_days_log: signedLog(inpatientDays),
    avg_inpatient_payment_amt_log: signedLog(inpatientPayment / Math.max(inpatientVisits, 1)),
    avg_outpatient_payment_amt_log: signedLog(outpatientPayment / Math.max(outpatientVisits, 1)),
    cost_per_claim: totalCost / totalClaims,
    BENE_SEX_IDENT_CD: form.sex,
    BENE_RACE_CD: '1',
    SP_STATE_CODE: '34',
    BENE_COUNTY_CD: '230',
  };
}

function readableFeatureName(feature) {
  const nameMap = {
    numeric__total_outpatient_claims: 'Number of outpatient visits',
    numeric__total_inpatient_payment_amt_log: 'Hospital care cost',
    numeric__total_inpatient_days_log: 'Length of hospital stay',
    numeric__total_outpatient_payment_amt_log: 'Outpatient care cost',
    numeric__distinct_inpatient_diagnosis_codes: 'Number of health issues found during hospital stay',
    numeric__distinct_outpatient_diagnosis_codes: 'Number of health issues found during outpatient visits',
    numeric__avg_inpatient_payment_amt_log: 'Average hospital care cost',
    numeric__avg_outpatient_payment_amt_log: 'Average outpatient care cost',
    numeric__chronic_condition_count: 'Number of ongoing health conditions',
    numeric__cost_per_claim: 'Overall cost of care',
    numeric__age: 'Age',
  };
  // SHAP factors arrive as "numeric__<name>"; recourse suggestions arrive as "<name>".
  const prefixed = feature.startsWith('numeric__') ? feature : `numeric__${feature}`;
  if (nameMap[prefixed]) return nameMap[prefixed];
  const extraNames = {
    total_inpatient_claims: 'Number of hospital stays',
    MEDREIMB_IP: 'Hospital care cost',
    MEDREIMB_OP: 'Outpatient care cost',
    total_inpatient_primary_payer_amt_log: 'Hospital costs paid by other insurance',
    total_outpatient_primary_payer_amt_log: 'Outpatient costs paid by other insurance',
  };
  if (extraNames[feature]) return extraNames[feature];
  return feature.replace(/^numeric__/, '').replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

// Recourse values come back in the model's units (log-scaled dollars and days).
// Convert them to something a person can read.
function formatRecourseValue(feature, value) {
  const name = feature.replace(/^numeric__/, '');
  const n = Number(value);
  if (Number.isNaN(n)) return String(value);
  const dollars = (x) => `$${Math.round(x).toLocaleString('en-US')}`;
  if (name.endsWith('_days_log')) {
    const days = Math.round(Math.expm1(n));
    return `${days} day${days === 1 ? '' : 's'}`;
  }
  if (name.endsWith('_amt_log')) return dollars(Math.expm1(n));
  if (name === 'cost_per_claim' || /^(MEDREIMB|BENRES|PPPYMT)_/.test(name)) return dollars(n);
  if (name.startsWith('total_') || name.startsWith('distinct_')) return String(Math.round(n));
  return String(Math.round(n * 10) / 10);
}

function buildPlainReason(result) {
  const isDeny = result.predicted_label_name === 'Deny';
  const factors = (result.top_factors || []).slice(0, 3);

  if (factors.length === 0) {
    return isDeny
      ? 'This claim was flagged as unusual based on the overall pattern of the submitted information.'
      : 'This claim matched the typical pattern of routine, low-risk claims.';
  }

  const clauses = factors.map((f) => {
    const name = readableFeatureName(f.feature).toLowerCase();
    return `the ${name} was ${f.value > 0 ? 'higher' : 'lower'} than typical`;
  });

  let clauseText;
  if (clauses.length === 1) clauseText = clauses[0];
  else if (clauses.length === 2) clauseText = `${clauses[0]}, and ${clauses[1]}`;
  else clauseText = `${clauses.slice(0, -1).join(', ')}, and ${clauses[clauses.length - 1]}`;

  const verdict = isDeny
    ? 'Together, these suggested a higher-risk or unusually costly claim.'
    : 'Together, these matched a routine, low-risk claim.';

  return `This claim was ${isDeny ? 'denied' : 'approved'} mainly because ${clauseText}. ${verdict}`;
}

function buildPlainRecourse(recourse) {
  if (!recourse) return '';
  if (recourse.error) return 'Suggestions for this claim are temporarily unavailable.';
  if (recourse.applicable === false) return 'No changes are applicable to this claim.';
  const cfs = recourse.counterfactuals || [];
  if (cfs.length === 0) return 'The system did not return specific suggestions for this claim.';
  const parts = cfs.map((cf, i) => {
    const changes = (cf.changes || [])
      .map((c) => {
        const verb = Number(c.to_value) < Number(c.from_value) ? 'reduce' : 'change';
        return `${verb} the ${readableFeatureName(c.feature).toLowerCase()} from ${formatRecourseValue(c.feature, c.from_value)} to ${formatRecourseValue(c.feature, c.to_value)}`;
      })
      .join(' and ');
    return `Option ${i + 1}: ${changes}.`;
  });
  return `This claim would likely have been approved with any one of these changes. ${parts.join(' ')}`;
}

function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [recourse, setRecourse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [anchorInfo, setAnchorInfo] = useState(null);
  const [anchoring, setAnchoring] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const set = (key, value) => setForm((f) => ({ ...f, [key]: value }));

  const toggleCondition = (key) => {
    setForm((f) => ({ ...f, conditions: { ...f.conditions, [key]: !f.conditions[key] } }));
  };

  const processFile = async (file) => {
    if (!file) return;
    set('document', file);

    if (file.type === 'application/pdf' && window.pdfjsLib) {
      window.pdfjsLib.GlobalWorkerOptions.workerSrc =
        'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
      const { cost, days } = await extractFromPdf(file);
      setForm((f) => ({
        ...f,
        extractedCost: cost,
        extractedDays: days,
        needsManualFallback: cost === null && days === null,
      }));
    } else {
      setForm((f) => ({ ...f, needsManualFallback: true }));
    }
  };

  const handleFileChange = (e) => processFile(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragActive(false);
    processFile(e.dataTransfer.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setRecourse(null);
    setAnchorInfo(null);
    try {
      const features = estimateClaimFeatures(form);
      features.patient_id = form.patientId?.trim() || null;
      features.claim_id = form.claimId?.trim() || null;

      const res = await axios.post(`${API_BASE}/predict`, features);
      setResult(res.data);

      if (res.data.predicted_label_name === 'Deny') {
        try {
          const recourseRes = await axios.post(`${API_BASE}/recourse`, features);
          setRecourse(recourseRes.data);
        } catch (rErr) {
          setRecourse({ error: true });
        }
      }
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

 
const handleAnchor = async () => {
  if (!result?.transaction_hash) return;

  setAnchoring(true);
  try {
    setAnchorInfo({
      transaction_hash: result.transaction_hash,
      explorer_url: `https://sepolia.etherscan.io/tx/${result.transaction_hash}`,
      block_number: result.block_number,
    });
  } catch (err) {
    setAnchorInfo({ error: true });
  } finally {
    setAnchoring(false);
  }
};


  return (
    <div className="page">
      <header className="masthead">
        <span className="wordmark">HealthAudit</span>
        <p className="tagline">A clear, explainable look at how your claim is reviewed</p>
      </header>

      <form className="claim-form" onSubmit={handleSubmit}>
        <section className="form-section">
          <h2>About you</h2><div className="field-row">
  <label>
    Patient ID
    <input
      type="text"
      value={form.patientId}
      onChange={(e) => set('patientId', e.target.value)}
      placeholder="e.g. PAT-1001"
      required
    />
  </label>

  <label>
    Claim ID
    <input
      type="text"
      value={form.claimId}
      onChange={(e) => set('claimId', e.target.value)}
      placeholder="e.g. CLM-2026-001"
      required
    />
  </label>
</div>
          <div className="field-row">
            <label>
              Full name
              <input type="text" value={form.name} onChange={(e) => set('name', e.target.value)} required />
            </label>
            <label className="field-narrow">
              Age
              <input type="number" min="0" max="120" value={form.age} onChange={(e) => set('age', e.target.value)} required />
            </label>
            <label className="field-narrow">
              Sex
              <select value={form.sex} onChange={(e) => set('sex', e.target.value)}>
                <option value="1">Male</option>
                <option value="2">Female</option>
              </select>
            </label>
          </div>
        </section>

        <section className="form-section">
          <h2>Do you have any of these conditions?</h2>
          <p className="section-hint">Select all that apply.</p>
          <div className="pill-grid">
            {CONDITIONS.map((c) => (
              <label key={c.key} className={`pill ${form.conditions[c.key] ? 'pill-active' : ''}`}>
                <input
                  type="checkbox"
                  checked={!!form.conditions[c.key]}
                  onChange={() => toggleCondition(c.key)}
                />
                {c.label}
              </label>
            ))}
          </div>
        </section>

        <section className="form-section">
          <h2>Supporting document</h2>
          <p className="section-hint">A medical bill, discharge summary, or prescription — we'll read the figures for you.</p>

          <label
            className={`dropzone ${dragActive ? 'dropzone-active' : ''} ${form.document ? 'dropzone-filled' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
          >
            <input type="file" accept="application/pdf,image/*" onChange={handleFileChange} hidden />
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6">
              <path d="M12 3v12m0-12 4 4m-4-4-4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span>{form.document ? form.document.name : 'Drag a file here, or click to choose one'}</span>
          </label>

          {form.document && !form.needsManualFallback && (form.extractedCost || form.extractedDays) && (
            <p className="extracted-note">
              Found in your document —
              {form.extractedCost ? ` cost of about $${form.extractedCost}` : ''}
              {form.extractedCost && form.extractedDays ? ',' : ''}
              {form.extractedDays ? ` ${form.extractedDays} day(s) hospitalized` : ''}
            </p>
          )}

          {form.needsManualFallback && (
            <div className="fallback-panel">
              <p className="section-hint">We couldn't read exact figures from your document. A couple of quick estimates:</p>
              <div className="field-row">
                <label>
                  Roughly, your total medical bill ($)
                  <input type="number" min="0" value={form.manualCost} onChange={(e) => set('manualCost', e.target.value)} />
                </label>
                <label>
                  Days hospitalized, if any
                  <input type="number" min="0" value={form.manualDays} onChange={(e) => set('manualDays', e.target.value)} />
                </label>
              </div>
            </div>
          )}
        </section>

        <button type="submit" className="btn-primary" disabled={loading}>
          {loading ? 'Reviewing your claim…' : 'Review my claim'}
        </button>
      </form>

      {error && <p className="error-text">{String(error)}</p>}

      {result && (
        <section className={`result-panel ${result.predicted_label_name === 'Deny' ? 'result-deny' : 'result-approve'}`}>
          <div className="result-head">
            <span className="status-pill">{result.predicted_label_name === 'Deny' ? 'Denied' : 'Approved'}</span>
            <p className="routing-note">
              {result.routing === 'Final Decision'
                ? `Final decision — ${(result.confidence * 100).toFixed(0)}% confidence.`
                : `Sent to a human reviewer — only ${(result.confidence * 100).toFixed(0)}% confident, below the threshold for an automatic decision.`}
            </p>
          </div>

          <div className="result-block">
            <h3>Why this decision was made</h3>
            <p>{buildPlainReason(result)}</p>
          </div>

          {result.predicted_label_name === 'Deny' && (
            <div className="result-block">
              <h3>What could change this outcome</h3>
              {recourse ? <p>{buildPlainRecourse(recourse)}</p> : <p className="muted">Loading suggestions…</p>}
            </div>
          )}

          
          {result.decision_hash && (
            <div className="result-block">
              <h3>Blockchain verification</h3>

              <p className="hash">
                Decision ID
                <br />
                <code>{result.decision_id}</code>
              </p>

              <p className="hash">
                SHA-256 decision hash
                <br />
                <code>{result.decision_hash}</code>
              </p>

              {result.transaction_hash && (
                <>
                  <p className="hash">
                    Sepolia transaction
                    <br />
                    <code>{result.transaction_hash}</code>
                  </p>

                  <p>
                    Block number: {result.block_number ?? 'Unavailable'}
                  </p>

                  <a
                    href={`https://sepolia.etherscan.io/tx/${result.transaction_hash}`}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Verify transaction on Etherscan ↗
                  </a>
                </>
              )}
            </div>
          )}

        </section>
      )}
    </div>
  );
}

export default App;