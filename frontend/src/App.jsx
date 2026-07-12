import { useState, useRef } from "react"
import axios from "axios"
import Dashboard from "./Dashboard.jsx"

const CATEGORIES = ["Application", "Financial", "Medical"]
const API_URL = "http://localhost:8000"

function JsonViewer({ data, depth = 0 }) {
  if (typeof data === "object" && data !== null && !Array.isArray(data)) {
    return (
      <div className={depth > 0 ? "ml-4 border-l-2 border-blue-100 pl-3 mt-1" : ""}>
        {Object.entries(data).map(([key, value]) => (
          <div key={key} className="mb-2">
            <span className="text-blue-800 font-semibold text-xs uppercase tracking-wide">
              {key.replace(/_/g, " ")}
            </span>
            {typeof value === "object" && value !== null ? (
              <JsonViewer data={value} depth={depth + 1} />
            ) : (
              <span className={`ml-2 text-sm font-medium
                ${value === true  ? "text-green-600"
                  : value === false ? "text-red-500"
                  : typeof value === "number" ? "text-purple-700"
                  : value === null ? "text-gray-300"
                  : "text-gray-800"}`}>
                {value === null ? "—" : Array.isArray(value) ? value.join(", ") || "None" : String(value)}
              </span>
            )}
          </div>
        ))}
      </div>
    )
  }
  return null
}

function App() {
  const [view, setView]                   = useState("upload") // "upload" | "dashboard"
  const [activeTab, setActiveTab]         = useState(0)
  const [files, setFiles]                 = useState({ 0: [], 1: [], 2: [] })
  const [analysing, setAnalysing]         = useState(false)
  const [analysed, setAnalysed]           = useState(false)
  const [progress, setProgress]           = useState(0)
  const [activeJsonTab, setActiveJsonTab] = useState(0)
  const [jsonResults, setJsonResults]     = useState(null)
  const [dashboardData, setDashboardData] = useState(null)
  const fileInputRef                      = useRef(null)

  const handleFileUpload = async (e) => {
    const newFiles = Array.from(e.target.files).map(file => ({
      id:        Date.now() + Math.random(),
      name:      file.name,
      file:      file,
      status:    "extracting",
      extracted: null,
      method:    null
    }))

    setFiles(prev => ({
      ...prev,
      [activeTab]: [...prev[activeTab], ...newFiles]
    }))

    newFiles.forEach(async (f) => {
      try {
        const formData = new FormData()
        formData.append("file", f.file)

        const response = await axios.post(`${API_URL}/extract`, formData)

        setFiles(prev => ({
          ...prev,
          [activeTab]: prev[activeTab].map(item =>
            item.id === f.id
              ? {
                  ...item,
                  status:    "done",
                  extracted: response.data.extracted_text,
                  method:    response.data.extraction_method
                }
              : item
          )
        }))
      } catch (err) {
        setFiles(prev => ({
          ...prev,
          [activeTab]: prev[activeTab].map(item =>
            item.id === f.id
              ? { ...item, status: "error" }
              : item
          )
        }))
      }
    })

    e.target.value = ""
  }

  const handleRemove = (fileId) => {
    setFiles(prev => ({
      ...prev,
      [activeTab]: prev[activeTab].filter(f => f.id !== fileId)
    }))
    setAnalysed(false)
    setJsonResults(null)
    setDashboardData(null)
  }

  const handleAnalyse = async () => {
    setAnalysing(true)
    setAnalysed(false)
    setJsonResults(null)
    setDashboardData(null)
    setProgress(0)

    // The backend does the whole analysis in one blocking request, so there's
    // no real progress to poll — simulate a smooth climb that eases off
    // before 90% and only completes once the response actually arrives.
    const progressTimer = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) return prev
        const step = prev < 50 ? 6 : prev < 75 ? 3 : 1
        return Math.min(90, prev + step)
      })
    }, 400)

    try {
      const formData = new FormData()
      files[0].forEach(f => formData.append("application_files", f.file))
      files[1].forEach(f => formData.append("financial_files",   f.file))
      files[2].forEach(f => formData.append("medical_files",     f.file))

      const response = await axios.post(`${API_URL}/analyse`, formData)
      clearInterval(progressTimer)
      setProgress(100)
      setJsonResults(response.data)
      setAnalysed(true)
      setDashboardData({
        profile:            response.data.profile,
        risk_classification: response.data.risk_classification
      })
      setView("dashboard")
    } catch (err) {
      console.error("Analyse error:", err)
      alert("Analysis failed. Check backend.")
    } finally {
      clearInterval(progressTimer)
      setAnalysing(false)
    }
  }

  const allDone = CATEGORIES.every((_, i) =>
    files[i].length > 0 && files[i].every(f => f.status === "done")
  )

  const jsonKeys = ["application", "financial", "medical"]

  if (view === "dashboard") {
    return <Dashboard onBack={() => setView("upload")} data={dashboardData} />
  }

  return (
    <div className="min-h-screen bg-gray-50">

      {/* Header */}
      <div className="bg-blue-900 text-white px-6 py-4 shadow-md flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold tracking-wide">
            Agentic Underwriting Workbench
          </h1>
          <p className="text-blue-200 text-sm mt-0.5">
            ICICI Prudential Life Insurance
          </p>
        </div>
        <button
          onClick={() => setView("dashboard")}
          className="text-xs font-semibold bg-blue-800 hover:bg-blue-700 text-white px-3 py-1.5 rounded-lg transition-all"
        >
          📊 View Dashboard {dashboardData ? "" : "(Demo)"}
        </button>
      </div>

      {/* Stepper */}
      <div className="bg-white border-b shadow-sm px-6 py-4">
        <div className="flex items-center gap-0">
          {CATEGORIES.map((cat, i) => {
            const catFiles = files[i]
            const isDone   = catFiles.length > 0 && catFiles.every(f => f.status === "done")
            const isActive = activeTab === i
            return (
              <div key={i} className="flex items-center">
                <button
                  onClick={() => setActiveTab(i)}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium text-sm transition-all
                    ${isActive ? "bg-blue-900 text-white shadow"
                      : isDone  ? "bg-green-100 text-green-800"
                      : "bg-gray-100 text-gray-500 hover:bg-gray-200"}`}
                >
                  <span className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                    ${isActive ? "bg-white text-blue-900"
                      : isDone  ? "bg-green-600 text-white"
                      : "bg-gray-300 text-gray-600"}`}>
                    {isDone ? "✓" : i + 1}
                  </span>
                  {cat}
                  {catFiles.length > 0 && (
                    <span className="text-xs opacity-70">({catFiles.length})</span>
                  )}
                </button>
                {i < CATEGORIES.length - 1 && (
                  <div className="w-8 h-0.5 bg-gray-300 mx-1" />
                )}
              </div>
            )
          })}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex gap-4 p-6">

        {/* Left Panel */}
        <div className="w-80 flex flex-col gap-4 flex-shrink-0">
          <div
            onClick={() => fileInputRef.current.click()}
            className="border-2 border-dashed border-blue-300 rounded-xl p-6 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all bg-white"
          >
            <div className="text-3xl mb-2">📁</div>
            <p className="text-blue-800 font-medium text-sm">
              Upload {CATEGORIES[activeTab]} Documents
            </p>
            <p className="text-gray-400 text-xs mt-1">PDF, JPG, PNG supported</p>
            <p className="text-gray-400 text-xs">Multiple files allowed</p>
          </div>

          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.jpg,.jpeg,.png"
            onChange={handleFileUpload}
            className="hidden"
          />

          {files[activeTab].length > 0 && (
            <div className="flex flex-col gap-2">
              <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                Uploaded Files
              </p>
              {files[activeTab].map(f => (
                <div key={f.id}
                  className={`flex items-center justify-between p-3 rounded-lg border text-sm
                    ${f.status === "done"  ? "bg-green-50 border-green-200"
                      : f.status === "error" ? "bg-red-50 border-red-200"
                      : "bg-amber-50 border-amber-200"}`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    <span>
                      {f.status === "done"  ? "✅"
                        : f.status === "error" ? "❌"
                        : "⏳"}
                    </span>
                    <div className="flex flex-col min-w-0">
                      <span className="truncate text-gray-700 text-xs">{f.name}</span>
                      {f.method && (
                        <span className="text-xs text-gray-400">
                          {f.method === "pdfplumber" ? "📄 pdfplumber" : "👁️ GPT-4o Vision"}
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleRemove(f.id)}
                    className="ml-2 text-gray-400 hover:text-red-500 font-bold transition-colors"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div className="flex-1 flex flex-col gap-4">

          {/* Extracted Text */}
          <div className="bg-white rounded-xl border shadow-sm p-5">
            <h2 className="text-blue-900 font-bold text-base mb-4 border-b pb-2">
              📄 Extracted Text — {CATEGORIES[activeTab]}
            </h2>

            {files[activeTab].length === 0 && (
              <div className="flex flex-col items-center justify-center h-32 text-gray-400">
                <div className="text-3xl mb-2">📂</div>
                <p className="text-sm">Upload documents to see extracted text</p>
              </div>
            )}

            {files[activeTab].map(f => (
              <div key={f.id} className="mb-6">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-sm">📄</span>
                  <span className="font-semibold text-gray-700 text-sm">{f.name}</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full font-medium
                    ${f.status === "done"  ? "bg-green-100 text-green-700"
                      : f.status === "error" ? "bg-red-100 text-red-700"
                      : "bg-amber-100 text-amber-700"}`}>
                    {f.status === "done"  ? "Extracted"
                      : f.status === "error" ? "Failed"
                      : "Extracting..."}
                  </span>
                  {f.method && (
                    <span className="text-xs text-gray-400 bg-gray-100 px-2 py-0.5 rounded-full">
                      {f.method === "pdfplumber" ? "pdfplumber" : "GPT-4o Vision"}
                    </span>
                  )}
                </div>

                {f.status === "extracting" && (
                  <div className="flex items-center gap-2 text-amber-600 text-sm pl-2">
                    <div className="w-4 h-4 border-2 border-amber-400 border-t-transparent rounded-full animate-spin" />
                    Extracting content...
                  </div>
                )}

                {f.status === "error" && (
                  <p className="text-red-500 text-sm pl-2">Extraction failed. Try again.</p>
                )}

                {f.status === "done" && f.extracted && (
                  <div className="bg-gray-50 rounded-lg p-3 border max-h-48 overflow-y-auto">
                    <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono">
                      {f.extracted}
                    </pre>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* JSON + Conflicts Output */}
          {analysed && (
            <div className="bg-white rounded-xl border shadow-sm p-5">
              <h2 className="text-blue-900 font-bold text-base mb-4 border-b pb-2">
                🧠 Analysis Results
              </h2>

              {analysed && jsonResults && (
                <>
                  {/* Tab Buttons */}
                  <div className="flex gap-2 mb-4 flex-wrap">
                    {["Application", "Financial", "Medical"].map((tab, i) => (
                      <button
                        key={i}
                        onClick={() => setActiveJsonTab(i)}
                        className={`px-4 py-1.5 rounded-lg text-sm font-medium transition-all
                          ${activeJsonTab === i
                            ? "bg-blue-900 text-white"
                            : "bg-gray-100 text-gray-600 hover:bg-gray-200"}`}
                      >
                        {tab}
                      </button>
                    ))}
                  </div>

                  {/* JSON Viewer */}
                  <div className="bg-gray-50 rounded-lg p-4 border max-h-96 overflow-y-auto">
                    <JsonViewer data={jsonResults[jsonKeys[activeJsonTab]]} />
                  </div>

                  <div className="mt-3 flex items-center gap-2 text-green-700 text-sm">
                    <span>✅</span>
                    <span>3 structured JSONs extracted</span>
                  </div>
                </>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Bottom Bar */}
      <div className="sticky bottom-0 bg-white border-t shadow-lg px-6 py-3 flex items-center justify-between">
        <p className="text-xs text-gray-500">
          {allDone
            ? "All documents ready — click Analyse to proceed"
            : "Upload documents in all 3 categories to enable analysis"}
        </p>
        <button
          onClick={handleAnalyse}
          disabled={!allDone || analysing}
          className={`px-8 py-2.5 rounded-lg font-semibold text-sm transition-all
            ${allDone && !analysing
              ? "bg-blue-900 text-white hover:bg-blue-800 shadow-md cursor-pointer"
              : "bg-gray-200 text-gray-400 cursor-not-allowed"}`}
        >
          {analysing ? "Analysing..." : "🔍 Analyse Case"}
        </button>
      </div>

      {/* Full-screen Analysing Overlay */}
      {analysing && (
        <div className="fixed inset-0 z-50 bg-white/60 backdrop-blur-sm flex items-center justify-center">
          <div className="flex flex-col items-center justify-center bg-white rounded-2xl shadow-xl border px-10 py-10 w-full max-w-sm mx-4">
            <div className="w-12 h-12 border-4 border-blue-300 border-t-blue-700 rounded-full animate-spin mb-5" />
            <p className="font-semibold text-blue-900 text-base">Analysing case...</p>
            <p className="text-sm text-gray-400 mt-1 text-center">
              Running 3 parallel extraction calls
            </p>
            <div className="w-full mt-6">
              <div className="w-full h-2.5 rounded-full bg-blue-100 overflow-hidden">
                <div
                  className="h-full rounded-full bg-blue-700 transition-all duration-300 ease-out"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-center text-sm font-semibold text-blue-700 mt-2">
                {progress}%
              </p>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App