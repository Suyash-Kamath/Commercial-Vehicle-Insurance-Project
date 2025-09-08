import React, { useMemo, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [mainFile, setMainFile] = useState(null)
  const [newFile, setNewFile] = useState(null)
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState(null)
  const isReady = useMemo(() => !!mainFile && !!newFile, [mainFile, newFile])

  async function handleSubmit(e) {
    e.preventDefault()
    setError(null)
    if (!isReady || !mainFile || !newFile) return
    setDownloading(true)
    try {
      const form = new FormData()
      form.append('main_file', mainFile)
      form.append('new_file', newFile)
      const res = await fetch(`${API_URL}/compare`, { method: 'POST', body: form })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `Request failed (${res.status})`)
      }
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'updated_report.xlsx'
      document.body.appendChild(a)
      a.click()
      a.remove()
      URL.revokeObjectURL(url)
    } catch (err) {
      setError(err?.message || 'Something went wrong')
    } finally {
      setDownloading(false)
    }
  }

  return (
    <div className="container">
      <h1>Report Comparator</h1>
      <form onSubmit={handleSubmit}>
        <label>Main report (.xlsx)</label>
        <input
          type="file"
          accept=".xlsx"
          onChange={(e) => setMainFile(e.target.files?.[0] || null)}
        />
        <div className="note">Columns: Weight, IIB State, FGI State, FGI Zone, Status, OD Out flow, TP Outflow</div>

        <label>New report (.xlsx, .csv, or image)</label>
        <input
          type="file"
          accept=".xlsx,.csv,.png,.jpg,.jpeg"
          onChange={(e) => setNewFile(e.target.files?.[0] || null)}
        />
        <div className="note">If image, the server will use OCR to extract data.</div>

        <div className="row" style={{ marginTop: 12 }}>
          <button className="button" disabled={!isReady || downloading}>
            {downloading ? 'Generating...' : 'Compare & Download'}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </form>

      <div className="result">
        Backend: <a className="link" href={`${API_URL}/health`} target="_blank" rel="noreferrer">{API_URL}/health</a>
      </div>
    </div>
  )
}
