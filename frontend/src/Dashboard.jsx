import { useState } from "react"
import mockCase from "./data/mockCase.json"

const RISK_STYLES = {
  LOW: {
    badge: "bg-emerald-100 text-emerald-800",
    ring: "border-emerald-200",
    iconBg: "bg-emerald-500",
    text: "text-emerald-700",
    tint: "bg-emerald-50"
  },
  MEDIUM: {
    badge: "bg-yellow-100 text-yellow-800",
    ring: "border-yellow-200",
    iconBg: "bg-yellow-400",
    text: "text-yellow-700",
    tint: "bg-yellow-50"
  },
  HIGH: {
    badge: "bg-red-100 text-red-800",
    ring: "border-red-200",
    iconBg: "bg-red-600",
    text: "text-red-700",
    tint: "bg-red-50"
  }
}

function formatLabel(key) {
  return key.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") return "—"
  if (typeof value === "boolean") return value ? "Yes" : "No"
  if (Array.isArray(value)) return value.length > 0 ? value.join(", ") : "None"
  return String(value)
}

function formatCurrency(value) {
  if (value === null || value === undefined || isNaN(Number(value))) return "—"
  return `Rs. ${Number(value).toLocaleString("en-IN")}`
}

/* Defensive coercion — extraction should return real booleans, but tolerate
   a stray "YES"/"NO" string so truthy-checks (e.g. avocation flags) don't
   misfire on non-empty-but-negative text. */
function isTrue(value) {
  if (typeof value === "boolean") return value
  if (typeof value === "string") return /^(yes|true)$/i.test(value.trim())
  return false
}

const AVOCATION_FLAGS = [
  "scuba_diving", "mountaineering", "private_aviation",
  "parachuting", "hang_gliding", "hot_air_ballooning", "motorsports"
]

function initials(name) {
  if (!name) return "?"
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map(w => w[0])
    .join("")
    .toUpperCase()
}

/* ── PROFILE TAB — hero + risk donut, viz KPI cards, icon data-grids ──── */

function parseNum(value) {
  if (value === null || value === undefined) return null
  if (typeof value === "number") return value
  const match = String(value).replace(/,/g, "").match(/-?\d+(\.\d+)?/)
  return match ? parseFloat(match[0]) : null
}

/* Semi-circular speedometer gauge — colored bands + a needle at the actual value */
function polarToCartesian(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function describeArc(cx, cy, r, startAngle, endAngle) {
  const start = polarToCartesian(cx, cy, r, endAngle)
  const end = polarToCartesian(cx, cy, r, startAngle)
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1"
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 0 ${end.x} ${end.y}`
}

function ArcGauge({ value, min, max, bands }) {
  const cx = 100, cy = 92, r = 78
  const clamped = Math.min(max, Math.max(min, value))
  const valueToAngle = v => -90 + ((v - min) / (max - min)) * 180
  const needleTip = polarToCartesian(cx, cy, r - 24, valueToAngle(clamped))
  const gapDeg = 1.5

  return (
    <div className="mt-3 w-full flex justify-center">
      <svg viewBox="0 0 200 108" className="w-full max-w-[190px]">
        {bands.map((b, i) => {
          const prevUpto = i === 0 ? min : bands[i - 1].upto
          let a0 = valueToAngle(prevUpto)
          let a1 = valueToAngle(b.upto)
          if (i !== 0) a0 += gapDeg
          if (i !== bands.length - 1) a1 -= gapDeg
          return (
            <path key={i} d={describeArc(cx, cy, r, a0, a1)}
              stroke={b.color} strokeWidth="14" fill="none" />
          )
        })}
        <line x1={cx} y1={cy} x2={needleTip.x} y2={needleTip.y}
          stroke="#1f2937" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="5" fill="#1f2937" />
        <text x={cx - r} y={cy + 14} fontSize="10" fill="#9ca3af" textAnchor="middle">{min}</text>
        <text x={cx + r} y={cy + 14} fontSize="10" fill="#9ca3af" textAnchor="middle">{max}</text>
      </svg>
    </div>
  )
}

/* Full-circle gauge — colored bands run all the way around, needle at the actual value */
function polarFull(cx, cy, r, angleDeg) {
  const rad = ((angleDeg - 90) * Math.PI) / 180
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) }
}

function describeFullArc(cx, cy, r, startAngle, endAngle) {
  const start = polarFull(cx, cy, r, startAngle)
  const end = polarFull(cx, cy, r, endAngle)
  const largeArcFlag = endAngle - startAngle <= 180 ? "0" : "1"
  return `M ${start.x} ${start.y} A ${r} ${r} 0 ${largeArcFlag} 1 ${end.x} ${end.y}`
}

function FullCircleGauge({ value, min, max, bands }) {
  const cx = 50, cy = 50, r = 38
  const clamped = Math.min(max, Math.max(min, value))
  const valueToAngle = v => ((v - min) / (max - min)) * 360
  const needleTip = polarFull(cx, cy, r - 8, valueToAngle(clamped))
  const gapDeg = 2

  return (
    <div className="mt-3 w-full flex justify-center">
      <svg viewBox="0 0 100 100" className="w-28 h-28">
        {bands.map((b, i) => {
          const prevUpto = i === 0 ? min : bands[i - 1].upto
          let a0 = valueToAngle(prevUpto)
          let a1 = valueToAngle(b.upto)
          if (i !== 0) a0 += gapDeg
          if (i !== bands.length - 1) a1 -= gapDeg
          return (
            <path key={i} d={describeFullArc(cx, cy, r, a0, a1)}
              stroke={b.color} strokeWidth="10" fill="none" strokeLinecap="round" />
          )
        })}
        <line x1={cx} y1={cy} x2={needleTip.x} y2={needleTip.y}
          stroke="#1f2937" strokeWidth="2.5" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="4" fill="#1f2937" />
      </svg>
    </div>
  )
}

/* A simple stacked-coins motif — a wealth symbol, not a line icon */
function CoinStack() {
  return (
    <div className="mt-3 flex justify-center">
      <svg viewBox="0 0 100 70" className="w-24 h-16">
        <ellipse cx="50" cy="54" rx="36" ry="11" fill="#fde68a" stroke="#c98500" strokeWidth="2" />
        <ellipse cx="50" cy="42" rx="36" ry="11" fill="#fbbf24" stroke="#c98500" strokeWidth="2" />
        <ellipse cx="50" cy="30" rx="36" ry="11" fill="#f2b705" stroke="#c98500" strokeWidth="2" />
        <text x="50" y="34.5" textAnchor="middle" fontSize="15" fontWeight="bold" fill="#7c4a03">Rs</text>
      </svg>
    </div>
  )
}

function ProgressBar({ pct, color }) {
  return (
    <div className="mt-3 w-full h-2.5 rounded-full bg-gray-100 overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${Math.min(100, Math.max(0, pct))}%`, backgroundColor: color }} />
    </div>
  )
}

function RingStat({ pct, color, caption }) {
  const r = 24
  const c = 2 * Math.PI * r
  const clamped = Math.min(100, Math.max(0, pct))
  const dash = (clamped / 100) * c
  return (
    <div className="flex items-center justify-center gap-3 mt-3 w-full">
      <div className="relative w-14 h-14 flex-shrink-0">
        <svg viewBox="0 0 60 60" className="w-full h-full -rotate-90">
          <circle cx="30" cy="30" r={r} fill="none" stroke="#eef0f2" strokeWidth="7" />
          <circle cx="30" cy="30" r={r} fill="none" stroke={color} strokeWidth="7" strokeLinecap="round"
            strokeDasharray={`${dash} ${c - dash}`} />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-[11px] font-bold text-gray-900">
          {Math.round(clamped)}%
        </div>
      </div>
      {caption && <span className="text-xs text-gray-400 leading-snug">{caption}</span>}
    </div>
  )
}

function VizCard({ label, value, children }) {
  return (
    <div className="bg-white rounded-2xl border shadow-sm p-5 flex flex-col items-center text-center">
      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">{label}</p>
      <p className="text-xl font-bold text-gray-900 mt-0.5">{value}</p>
      {children}
    </div>
  )
}

/* Compact risk-distribution donut — real counts from the classified features */
function HeaderRiskDonut({ counts }) {
  const total = counts.LOW + counts.MEDIUM + counts.HIGH
  const radius = 26
  const circumference = 2 * Math.PI * radius
  const gap = total > 1 ? 5 : 0
  const segments = [
    { key: "HIGH", value: counts.HIGH, color: "#dc2626" },
    { key: "MEDIUM", value: counts.MEDIUM, color: "#facc15" },
    { key: "LOW", value: counts.LOW, color: "#10b981" }
  ].filter(s => s.value > 0)

  let offset = 0
  const arcs = segments.map(seg => {
    const raw = total > 0 ? (seg.value / total) * circumference : 0
    const dash = Math.max(raw - gap, 0)
    const el = { ...seg, dash, offset }
    offset += raw
    return el
  })

  const overall = counts.HIGH > 0 ? "HIGH" : counts.MEDIUM > 0 ? "MEDIUM" : "LOW"
  const style = RISK_STYLES[overall]

  return (
    <div className="flex items-center gap-4">
      <div className="relative w-16 h-16 flex-shrink-0">
        <svg viewBox="0 0 64 64" className="w-full h-full -rotate-90">
          <circle cx="32" cy="32" r={radius} fill="none" stroke="#eef0f2" strokeWidth="8" />
          {arcs.map(a => (
            <circle key={a.key} cx="32" cy="32" r={radius} fill="none"
              stroke={a.color} strokeWidth="8" strokeLinecap="round"
              strokeDasharray={`${a.dash} ${circumference - a.dash}`}
              strokeDashoffset={-a.offset} />
          ))}
        </svg>
        <div className="absolute inset-0 flex items-center justify-center text-sm font-bold text-gray-900">
          {total}
        </div>
      </div>
      <div className="flex flex-col gap-1.5">
        <span className={`self-start text-xs font-bold px-2.5 py-1 rounded-full ${style.badge}`}>
          {overall} RISK
        </span>
        <div className="flex gap-2.5 text-[11px] text-gray-500">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-600" />{counts.HIGH} High</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-400" />{counts.MEDIUM} Med</span>
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500" />{counts.LOW} Low</span>
        </div>
      </div>
    </div>
  )
}

function HeroHeader({ personal, counts }) {
  return (
    <div className="bg-white rounded-2xl border shadow-sm p-6 flex items-center justify-between flex-wrap gap-4">
      <div className="flex items-center gap-5">
        <div className="w-16 h-16 rounded-2xl bg-blue-900 text-white flex items-center justify-center text-xl font-bold flex-shrink-0">
          {initials(personal.full_name)}
        </div>
        <div>
          <h2 className="text-lg font-bold text-gray-900">{personal.full_name}</h2>
          <p className="text-sm text-gray-500 mt-1">
            {personal.marital_status} &nbsp;·&nbsp; {personal.gender} &nbsp;·&nbsp; {personal.age} yrs &nbsp;·&nbsp; {personal.nationality}
          </p>
          <p className="text-sm text-gray-500 mt-0.5">
            {personal.city}, {personal.state} &nbsp;·&nbsp; PIN {personal.pincode}
          </p>
        </div>
      </div>
      <HeaderRiskDonut counts={counts} />
    </div>
  )
}

function DataRow({ label, value }) {
  return (
    <div className="flex flex-col py-2 px-3 rounded-lg bg-gray-50/70">
      <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wide">{label}</span>
      <span className="text-sm text-gray-800 font-medium mt-0.5">{formatValue(value)}</span>
    </div>
  )
}

function DataGridCard({ title, themeKey, fields }) {
  const theme = SECTION_THEME[themeKey] ?? SECTION_THEME.Financial
  return (
    <div className={`bg-white rounded-2xl border border-l-4 ${theme.border} shadow-sm p-6`}>
      <h3 className={`font-bold text-base mb-4 pb-3 border-b ${theme.text}`}>
        {title}
      </h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
        {fields.map(([label, value]) => (
          <DataRow key={label} label={label} value={value} />
        ))}
      </div>
    </div>
  )
}

function ProfileTab({ profile, counts }) {
  const { personal, occupation, avocations, lifestyle, financial_summary, policy, health_declaration, family_history, driving_record } = profile

  const financialFields = [
    ["Annual Income", financial_summary.annual_income_display],
    ["Monthly Net Salary", formatCurrency(financial_summary.monthly_net_salary)],
    ["Total Monthly EMI", formatCurrency(financial_summary.total_monthly_emi)],
    ["CIBIL Score", financial_summary.cibil_score],
    ["Income Coverage Ratio", financial_summary.income_coverage_ratio],
    ["EMI to Income Ratio", financial_summary.emi_to_income_ratio],
    ["Dependents", personal.number_of_dependents]
  ]

  const sumAssured = financial_summary.sum_assured || 0
  const existingCoverage = policy.existing_coverage_amount || 0
  const existingPct = sumAssured > 0 ? (existingCoverage / sumAssured) * 100 : 0
  const incomeRatioNum = parseNum(financial_summary.income_coverage_ratio)
  const cibilNum = parseNum(financial_summary.cibil_score)
  const emiRatioNum = parseNum(financial_summary.emi_to_income_ratio)

  const medicalFields = [
    ["Blood Pressure", health_declaration.blood_pressure],
    ["BMI", health_declaration.bmi],
    ["Cholesterol & Lipid Profile", `Total ${formatValue(health_declaration.total_cholesterol)}, ratio ${formatValue(health_declaration.cholesterol_ratio)}`],
    ["Liver Function", `AST ${formatValue(health_declaration.liver_ast)}, ALT ${formatValue(health_declaration.liver_alt)}, Bilirubin ${formatValue(health_declaration.liver_bilirubin)}`],
    ["Thyroid", `TSH ${formatValue(health_declaration.thyroid_tsh)}`],
    ["Kidney Function", `Creatinine ${formatValue(health_declaration.kidney_creatinine)}, eGFR ${formatValue(health_declaration.kidney_egfr)}`],
    ["Diabetes", health_declaration.diabetes ? `${health_declaration.diabetes_type} (HbA1c ${health_declaration.hbA1c})` : "No"],
    ["Hypertension", health_declaration.hypertension],
    ["Heart Disease", health_declaration.heart_disease],
    [
      "Family Heart Disease",
      (family_history.father_heart_disease || family_history.siblings_heart_disease)
        ? `Yes (Father MI at ${family_history.father_age_at_death}; sibling affected)`
        : "No"
    ]
  ]

  const lifestyleFields = [
    ["Occupation", occupation.designation],
    ["Employer", occupation.employer],
    ["Nature of Work", occupation.nature_of_work],
    ["Smoker", lifestyle.smoker],
    ["Alcohol Use", `${lifestyle.alcohol_use} (${lifestyle.alcohol_frequency_per_week}x/week)`],
    ["Avocations", AVOCATION_FLAGS.filter(k => isTrue(avocations[k])).map(formatLabel)],
    ["Drunk Driving (12mo)", lifestyle.drunk_driving_incidents_last_12_months],
    ["Moving Violations (3yr)", driving_record.moving_violations_last_3_years]
  ]

  return (
    <div className="flex flex-col gap-5">
      <HeroHeader personal={personal} counts={counts} />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        <VizCard label="Sum Assured"
          value={formatCurrency(financial_summary.sum_assured)}>
          <CoinStack />
        </VizCard>

        <VizCard label="Income Coverage Ratio"
          value={formatValue(financial_summary.income_coverage_ratio)}>
          {incomeRatioNum !== null
            ? <FullCircleGauge value={incomeRatioNum} min={0} max={30}
                bands={[{ upto: 20, color: "#10b981" }, { upto: 25, color: "#facc15" }, { upto: 30, color: "#dc2626" }]} />
            : <p className="text-xs text-gray-400 mt-3">No data available</p>}
        </VizCard>

        <VizCard label="CIBIL Score"
          value={formatValue(financial_summary.cibil_score)}>
          {cibilNum !== null
            ? <ArcGauge value={cibilNum} min={300} max={900}
                bands={[{ upto: 650, color: "#dc2626" }, { upto: 750, color: "#facc15" }, { upto: 900, color: "#10b981" }]} />
            : <p className="text-xs text-gray-400 mt-3">No credit report uploaded</p>}
        </VizCard>

        <VizCard label="EMI to Income Ratio"
          value={formatValue(financial_summary.emi_to_income_ratio)}>
          {emiRatioNum !== null
            ? <FullCircleGauge value={emiRatioNum} min={0} max={100}
                bands={[{ upto: 40, color: "#10b981" }, { upto: 55, color: "#facc15" }, { upto: 100, color: "#dc2626" }]} />
            : <p className="text-xs text-gray-400 mt-3">No data available</p>}
        </VizCard>

        <VizCard label="Policy Term"
          value={policy.policy_term_years != null ? `${policy.policy_term_years} years` : "—"}>
          <ProgressBar pct={policy.policy_term_years ? (policy.policy_term_years / 30) * 100 : 0} color="#2a78d6" />
        </VizCard>

        <VizCard label="Existing Coverage"
          value={formatCurrency(policy.existing_coverage_amount)}>
          <RingStat pct={existingPct} color="#1baf7a" caption="of total sum assured already in force" />
        </VizCard>
      </div>

      <div className="flex flex-col gap-4">
        <DataGridCard title="Financial Data" themeKey="Financial" fields={financialFields} />
        <DataGridCard title="Medical Snapshot" themeKey="Medical" fields={medicalFields} />
        <DataGridCard title="Lifestyle & Occupation" themeKey="Lifestyle" fields={lifestyleFields} />
      </div>
    </div>
  )
}

/* ── RISK CLASSIFICATION TAB — tile-grid style ────────────────────────── */

const SECTION_THEME = {
  "Lifestyle":            { accent: "bg-indigo-600", soft: "bg-indigo-50", text: "text-indigo-700", icon: "🧬", border: "border-indigo-400" },
  "Medical":              { accent: "bg-violet-600", soft: "bg-violet-50", text: "text-violet-700", icon: "🏥", border: "border-violet-400" },
  "Financial":            { accent: "bg-sky-700",    soft: "bg-sky-50",   text: "text-sky-800",     icon: "💰", border: "border-sky-500" },
  "Information Mismatch": { accent: "bg-slate-600",  soft: "bg-slate-100", text: "text-slate-700",  icon: "🔍", border: "border-slate-400" }
}

function RiskTile({ feature, onClick }) {
  const style = RISK_STYLES[feature.risk] ?? RISK_STYLES.LOW
  return (
    <button
      onClick={onClick}
      className="text-left w-full group rounded-xl border border-gray-200 bg-white p-4 flex flex-col gap-2 transition-all hover:shadow-md hover:border-gray-300 cursor-pointer"
    >
      <div className="flex items-start justify-between gap-2">
        <p className="text-sm font-bold text-gray-900">{feature.feature}</p>
        <span className={`text-[11px] font-bold tracking-wide px-2.5 py-1 rounded-full flex-shrink-0 ${style.badge}`}>
          {feature.risk}
        </span>
      </div>
      <p className="text-xs text-gray-500 leading-snug">{feature.value}</p>
      <span className="text-[11px] text-gray-400 font-semibold opacity-0 group-hover:opacity-100 transition-opacity">
        View reason →
      </span>
    </button>
  )
}

function RiskSection({ title, features, onSelect }) {
  const theme = SECTION_THEME[title] ?? SECTION_THEME.Financial
  const counts = features.reduce(
    (acc, f) => ({ ...acc, [f.risk]: (acc[f.risk] ?? 0) + 1 }),
    { LOW: 0, MEDIUM: 0, HIGH: 0 }
  )
  return (
    <div className="bg-white rounded-2xl border shadow-sm overflow-hidden">
      <div className={`flex items-center justify-between px-5 py-4 border-b ${theme.soft}`}>
        <div className="flex items-center gap-3">
          <span className={`w-9 h-9 rounded-xl ${theme.accent} flex items-center justify-center text-white text-base shadow-sm`}>
            {theme.icon}
          </span>
          <h3 className={`font-bold ${theme.text} text-base`}>{title}</h3>
        </div>
        <div className="flex gap-1.5">
          {counts.LOW > 0 && <span className="text-xs px-2.5 py-1 rounded-full bg-emerald-100 text-emerald-800 font-semibold">{counts.LOW} Low</span>}
          {counts.MEDIUM > 0 && <span className="text-xs px-2.5 py-1 rounded-full bg-yellow-100 text-yellow-800 font-semibold">{counts.MEDIUM} Med</span>}
          {counts.HIGH > 0 && <span className="text-xs px-2.5 py-1 rounded-full bg-red-100 text-red-800 font-semibold">{counts.HIGH} High</span>}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 p-5">
        {features.map((f, i) => <RiskTile key={i} feature={f} onClick={() => onSelect({ ...f, category: title })} />)}
      </div>
    </div>
  )
}

function ReasonModal({ item, onClose }) {
  if (!item) return null
  const style = RISK_STYLES[item.risk] ?? RISK_STYLES.LOW
  return (
    <div
      className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white rounded-2xl shadow-xl max-w-md w-full p-6"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-start justify-between mb-4">
          <div>
            <p className="text-xs text-gray-400 uppercase tracking-wide font-semibold">{item.category}</p>
            <h3 className="text-base font-bold text-gray-900">{item.feature}</h3>
          </div>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 text-xl leading-none px-1">×</button>
        </div>

        <span className={`inline-block text-xs font-bold px-2.5 py-1 rounded-full ${style.badge} mb-4`}>
          {item.risk} RISK
        </span>

        <div className="mb-4">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Finding</p>
          <p className="text-sm text-gray-800">{item.value}</p>
        </div>

        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wide mb-1">Why this classification?</p>
          <p className="text-sm text-gray-700 leading-relaxed">{item.reason}</p>
        </div>
      </div>
    </div>
  )
}

/* Semi-circular risk-level gauge — LOW/MEDIUM/HIGH zones, needle at the overall category */
function RiskGauge({ overall }) {
  const cx = 115, cy = 100, r = 68
  const zones = { LOW: 100 / 6, MEDIUM: 50, HIGH: (100 * 5) / 6 }
  const pct = zones[overall]
  const valueToAngle = p => -90 + (p / 100) * 180
  const needleTip = polarToCartesian(cx, cy, r - 24, valueToAngle(pct))
  const bands = [
    { upto: 100 / 3, color: "#10b981" },
    { upto: 200 / 3, color: "#facc15" },
    { upto: 100, color: "#dc2626" }
  ]
  const gapDeg = 1.5

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 230 135" className="w-full max-w-[260px]">
        {bands.map((b, i) => {
          const prevUpto = i === 0 ? 0 : bands[i - 1].upto
          let a0 = valueToAngle(prevUpto)
          let a1 = valueToAngle(b.upto)
          if (i !== 0) a0 += gapDeg
          if (i !== bands.length - 1) a1 -= gapDeg
          return (
            <path key={i} d={describeArc(cx, cy, r, a0, a1)}
              stroke={b.color} strokeWidth="16" fill="none" />
          )
        })}
        <line x1={cx} y1={cy} x2={needleTip.x} y2={needleTip.y}
          stroke="#1f2937" strokeWidth="3" strokeLinecap="round" />
        <circle cx={cx} cy={cy} r="5" fill="#1f2937" />
        <text x={cx - r - 15} y={cy + 26} fontSize="11" fontWeight="bold" fill="#6b7280" textAnchor="middle">LOW</text>
        <text x={cx} y={cy - r - 14} fontSize="11" fontWeight="bold" fill="#6b7280" textAnchor="middle">MEDIUM</text>
        <text x={cx + r + 15} y={cy + 26} fontSize="11" fontWeight="bold" fill="#6b7280" textAnchor="middle">HIGH</text>
      </svg>
      <span className="text-xs font-bold text-gray-400 uppercase tracking-widest">Risk</span>
    </div>
  )
}

function DecisionBanner({ counts, overall }) {
  const style = RISK_STYLES[overall]
  const total = counts.LOW + counts.MEDIUM + counts.HIGH

  return (
    <div className="rounded-2xl border border-gray-200 bg-white p-6 shadow-sm flex flex-col items-center text-center gap-5">
      <div>
        <p className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Overall Risk Classification</p>
        <span className={`inline-block text-xl font-extrabold px-3 py-1 rounded-lg tracking-tight ${style.badge}`}>
          {overall} RISK
        </span>
      </div>

      <div className="flex gap-8">
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">{counts.LOW}</div>
          <div className="text-xs text-gray-500 font-medium">Low</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">{counts.MEDIUM}</div>
          <div className="text-xs text-gray-500 font-medium">Medium</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-gray-800">{counts.HIGH}</div>
          <div className="text-xs text-gray-500 font-medium">High</div>
        </div>
        <div className="text-center border-l pl-8">
          <div className="text-2xl font-bold text-gray-800">{total}</div>
          <div className="text-xs text-gray-500 font-medium">Total</div>
        </div>
      </div>

      <RiskGauge overall={overall} />
    </div>
  )
}

function RiskTab({ risk_classification, counts, overall }) {
  const [selected, setSelected] = useState(null)
  return (
    <div className="flex flex-col gap-5">
      <RiskSection title="Information Mismatch" features={risk_classification.information_mismatch} onSelect={setSelected} />
      <RiskSection title="Lifestyle" features={risk_classification.lifestyle} onSelect={setSelected} />
      <RiskSection title="Financial" features={risk_classification.financial} onSelect={setSelected} />
      <RiskSection title="Medical" features={risk_classification.medical} onSelect={setSelected} />
      <DecisionBanner counts={counts} overall={overall} />
      <ReasonModal item={selected} onClose={() => setSelected(null)} />
    </div>
  )
}

/* ── MAIN DASHBOARD ────────────────────────────────────────────────────── */

function Dashboard({ onBack, data }) {
  const [activeTab, setActiveTab] = useState(0)
  const { profile, risk_classification } = data ?? mockCase

  const allFeatures = [
    ...risk_classification.financial,
    ...risk_classification.medical,
    ...risk_classification.lifestyle,
    ...risk_classification.information_mismatch
  ]
  const counts = allFeatures.reduce(
    (acc, f) => ({ ...acc, [f.risk]: (acc[f.risk] ?? 0) + 1 }),
    { LOW: 0, MEDIUM: 0, HIGH: 0 }
  )
  const overall = counts.HIGH > 0 ? "HIGH" : counts.MEDIUM > 0 ? "MEDIUM" : "LOW"

  const TABS = ["Profile", "Risk Classification"]

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-blue-900 text-white px-6 py-4 shadow-md flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-wide">Underwriting Case Dashboard</h1>
          <p className="text-blue-200 text-sm mt-0.5">
            Application No. {formatValue(profile.personal.application_number)}
          </p>
        </div>
        {onBack && (
          <button
            onClick={onBack}
            className="text-xs font-semibold bg-blue-800 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg transition-all"
          >
            ← Back to Upload
          </button>
        )}
      </div>

      {/* Tabs */}
      <div className="bg-white border-b shadow-sm px-6 py-3">
        <div className="flex gap-2">
          {TABS.map((tab, i) => (
            <button
              key={i}
              onClick={() => setActiveTab(i)}
              className={`px-4 py-2 rounded-lg font-medium text-sm transition-all
                ${activeTab === i ? "bg-blue-900 text-white shadow" : "bg-gray-100 text-gray-500 hover:bg-gray-200"}`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="p-6">
        {activeTab === 0 && <ProfileTab profile={profile} counts={counts} />}
        {activeTab === 1 && <RiskTab risk_classification={risk_classification} counts={counts} overall={overall} />}
      </div>
    </div>
  )
}

export default Dashboard
