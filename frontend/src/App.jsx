import { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

const FIELD_GROUPS = {
  'Demographics': [
    ['age', 'Age', 'number', 72],
    ['BENE_SEX_IDENT_CD', 'Sex (1=Male, 2=Female)', 'text', '1'],
    ['BENE_RACE_CD', 'Race Code', 'text', '1'],
    ['BENE_COUNTY_CD', 'County Code', 'text', '230'],
    ['SP_STATE_CODE', 'State Code', 'text', '34'],
    ['is_deceased', 'Deceased', 'checkbox', 0],
    ['BENE_ESRD_IND', 'ESRD Indicator', 'checkbox', 0],
  ],
  'Coverage': [
    ['BENE_HI_CVRAGE_TOT_MONS', 'Part A Coverage (months)', 'number', 12],
    ['BENE_SMI_CVRAGE_TOT_MONS', 'Part B Coverage (months)', 'number', 12],
    ['BENE_HMO_CVRAGE_TOT_MONS', 'HMO Coverage (months)', 'number', 0],
    ['PLAN_CVRG_MOS_NUM', 'Plan Coverage (months)', 'number', 0],
  ],
  'Chronic Conditions': [
    ['chronic_condition_count', 'Total Chronic Condition Count', 'number', 4],
    ['SP_ALZHDMTA', "Alzheimer's/Dementia", 'checkbox', 0],
    ['SP_CHF', 'Congestive Heart Failure', 'checkbox', 1],
    ['SP_CHRNKIDN', 'Chronic Kidney Disease', 'checkbox', 0],
    ['SP_CNCR', 'Cancer', 'checkbox', 0],
    ['SP_COPD', 'COPD', 'checkbox', 0],
    ['SP_DEPRESSN', 'Depression', 'checkbox', 1],
    ['SP_DIABETES', 'Diabetes', 'checkbox', 1],
    ['SP_ISCHMCHT', 'Ischemic Heart Disease', 'checkbox', 1],
    ['SP_OSTEOPRS', 'Osteoporosis', 'checkbox', 0],
    ['SP_RA_OA', 'Rheumatoid/Osteoarthritis', 'checkbox', 0],
    ['SP_STRKETIA', 'Stroke/TIA', 'checkbox', 0],
  ],
  'Financials': [
    ['BENRES_CAR', 'Beneficiary Responsibility (Carrier)', 'number', 125],
    ['BENRES_IP', 'Beneficiary Responsibility (Inpatient)', 'number', 1000],
    ['BENRES_OP', 'Beneficiary Responsibility (Outpatient)', 'number', 200],
    ['MEDREIMB_CAR', 'Medicare Reimbursement (Carrier)', 'number', 500],
    ['MEDREIMB_IP', 'Medicare Reimbursement (Inpatient)', 'number', 4000],
    ['MEDREIMB_OP', 'Medicare Reimbursement (Outpatient)', 'number', 800],
    ['PPPYMT_CAR', 'Primary Payer Payment (Carrier)', 'number', 0],
    ['PPPYMT_IP', 'Primary Payer Payment (Inpatient)', 'number', 0],
    ['PPPYMT_OP', 'Primary Payer Payment (Outpatient)', 'number', 0],
    ['cost_per_claim', 'Cost Per Claim', 'number', 1350],
  ],
  'Claims & Utilization': [
    ['total_inpatient_claims', 'Total Inpatient Claims', 'number', 1],
    ['total_outpatient_claims', 'Total Outpatient Claims', 'number', 2],
    ['distinct_inpatient_diagnosis_codes', 'Distinct Inpatient Diagnosis Codes', 'number', 3],
    ['distinct_outpatient_diagnosis_codes', 'Distinct Outpatient Diagnosis Codes', 'number', 2],
    ['total_inpatient_days_log', 'Total Inpatient Days (log)', 'number', 1.79],
    ['total_inpatient_payment_amt_log', 'Total Inpatient Payment Amt (log)', 'number', 8.29],
    ['avg_inpatient_payment_amt_log', 'Avg Inpatient Payment Amt (log)', 'number', 8.29],
    ['total_inpatient_primary_payer_amt_log', 'Total Inpatient Primary Payer Amt (log)', 'number', 0],
    ['total_outpatient_payment_amt_log', 'Total Outpatient Payment Amt (log)', 'number', 4.11],
    ['avg_outpatient_payment_amt_log', 'Avg Outpatient Payment Amt (log)', 'number', 3.4],
    ['total_outpatient_primary_payer_amt_log', 'Total Outpatient Primary Payer Amt (log)', 'number', 0],
  ],
};

const initialState = {};
Object.values(FIELD_GROUPS).flat().forEach(([key, , type, def]) => {
  initialState[key] = def;
});

function readableFeatureName(feature) {
  const nameMap = {
    numeric__total_outpatient_claims: 'Total outpatient claims',
    numeric__total_inpatient_payment_amt_log: 'Total inpatient payment amount',
    numeric__total_inpatient_days_log: 'Total inpatient days',
    numeric__total_outpatient_payment_amt_log: 'Total outpatient payment amount',
    numeric__distinct_inpatient_diagnosis_codes: 'Number of distinct inpatient diagnoses',
    numeric__distinct_outpatient_diagnosis_codes: 'Number of distinct outpatient diagnoses',
    numeric__avg_inpatient_payment_amt_log: 'Average inpatient payment amount',
    numeric__avg_outpatient_payment_amt_log: 'Average outpatient payment amount',
    numeric__chronic_condition_count: 'Number of chronic conditions',
    numeric__cost_per_claim: 'Cost per claim',
    numeric__age: 'Patient age',
  };
  if (nameMap[feature]) return nameMap[feature];
  return feature
    .replace(/^numeric__/, '')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function describeDirection(value) {
  return value > 0 ? 'higher than typical' : 'lower than typical';
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
    const direction = describeDirection(f.value);
    return `${name} was ${direction} for this patient`;
  });

  let clauseText;
  if (clauses.length === 1) {
    clauseText = clauses[0];
  } else if (clauses.length === 2) {
    clauseText = `${clauses[0]}, and ${clauses[1]}`;
  } else {
    clauseText = `${clauses.slice(0, -1).join(', ')}, and ${clauses[clauses.length - 1]}`;
  }

  const verdict = isDeny
    ? 'These factors together suggested a higher risk of an anomalous or unusually costly claim, so it was flagged for denial.'
    : 'These factors together were consistent with a routine, low-risk claim, so it was approved.';

  return `This claim was ${isDeny ? 'denied' : 'approved'} mainly because ${clauseText}. ${verdict}`;
}

function buildPlainRecourse(recourse) {
  if (!recourse) return '';
  if (recourse.error) {
    return 'Suggestions for this claim are temporarily unavailable.';
  }
  if (recourse.applicable === false) {
    return 'No changes are applicable to this claim.';
  }
  const suggestions = recourse.counterfactuals || [];
  if (suggestions.length === 0) {
    return 'The system did not return specific suggestions for this claim.';
  }
  return `Based on similar cases, this claim could likely be approved if: ${suggestions
    .map((s) => JSON.stringify(s))
    .join('; ')}.`;
}

function App() {
  const [formData, setFormData] = useState(initialState);
  const [result, setResult] = useState(null);
  const [recourse, setRecourse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (key, type, value) => {
    if (type === 'checkbox') {
      setFormData({ ...formData, [key]: value ? 1 : 0 });
    } else if (type === 'number') {
      setFormData({ ...formData, [key]: value === '' ? '' : Number(value) });
    } else {
      setFormData({ ...formData, [key]: value });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);
    setRecourse(null);
    try {
      const res = await axios.post(`${API_BASE}/predict`, formData);
      setResult(res.data);

      if (res.data.predicted_label_name === 'Deny') {
        try {
          const recourseRes = await axios.post(`${API_BASE}/recourse`, formData);
          setRecourse(recourseRes.data);
        } catch (rErr) {
          console.error('Recourse fetch failed:', rErr);
          setRecourse({ error: true });
        }
      }
    } catch (err) {
      setError(
        err.response?.data?.detail?.[0]?.msg ||
        err.response?.data?.detail ||
        'Prediction failed. Is the backend running?'
      );
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1>HealthAudit-AI — Claim Review</h1>
      <p className="subtitle">Explainable AI system for healthcare claim decision support</p>

      <form onSubmit={handleSubmit}>
        {Object.entries(FIELD_GROUPS).map(([groupName, fields]) => (
          <fieldset key={groupName}>
            <legend>{groupName}</legend>
            <div className="field-grid">
              {fields.map(([key, label, type]) => (
                <label key={key} className={type === 'checkbox' ? 'checkbox-label' : ''}>
                  {label}
                  {type === 'checkbox' ? (
                    <input
                      type="checkbox"
                      checked={!!formData[key]}
                      onChange={(e) => handleChange(key, type, e.target.checked)}
                    />
                  ) : (
                    <input
                      type={type === 'number' ? 'number' : 'text'}
                      step="any"
                      value={formData[key]}
                      onChange={(e) => handleChange(key, type, e.target.value)}
                      required
                    />
                  )}
                </label>
              ))}
            </div>
          </fieldset>
        ))}

        <button type="submit" disabled={loading}>
          {loading ? 'Analyzing...' : 'Submit Claim'}
        </button>
      </form>

      {error && <p className="error">{String(error)}</p>}

      {result && (
        <div className="result">
          <h2 className={result.predicted_label_name === 'Deny' ? 'label-deny' : 'label-approve'}>
            Claim {result.predicted_label_name === 'Deny' ? 'Denied' : 'Approved'}
          </h2>

          <p className="routing-note">
            {result.routing === 'Final Decision'
              ? `This is a final, automated decision (${(result.confidence * 100).toFixed(0)}% model confidence).`
              : `This decision is not final — it has been sent to a human reviewer because the model's confidence was only ${(result.confidence * 100).toFixed(0)}%, below the threshold needed for an automatic decision.`}
          </p>

          <div className="reason-summary">
            <h3>Why this decision was made</h3>
            <p>{buildPlainReason(result)}</p>
          </div>

          {result.predicted_label_name === 'Deny' && (
            <div className="recourse-box">
              <h3>What could change this outcome?</h3>
              {recourse ? (
                <p>{buildPlainRecourse(recourse)}</p>
              ) : (
                <p className="recourse-loading">Loading suggestions...</p>
              )}
            </div>
          )}

          {result.record_hash && (
            <p className="hash">Decision Hash: <code>{result.record_hash}</code></p>
          )}
        </div>
      )}
    </div>
  );
}

export default App;