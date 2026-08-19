import { useState, useRef, useEffect } from 'react'
import { useQuery } from '@tanstack/react-query'
import { BOOK_TITLES, type Book } from './BookPicker'

type SearchResult = {
  rank: number
  score: number
  content: string
  start_index: number
  end_index: number
}

const PIPELINES = [
  { value: 1, label: 'TF-IDF' },
  { value: 2, label: 'Embeddings' },
  { value: 3, label: 'TF-IDF + Embeddings' },
  { value: 4, label: 'TF-IDF + Embeddings + Reranker' },
  { value: 5, label: 'TF-IDF + Embeddings + Reranker + NLI' },
  { value: 6, label: 'TF-IDF + Embeddings + LLM' },
  { value: 7, label: 'TF-IDF + Embeddings + Reranker + LLM' },
  { value: 8, label: 'TF-IDF + Embeddings(HyDE) + Reranker + LLM' },
  { value: 9, label: 'TF-IDF + Embeddings(HyDE) + Reranker' },
  { value: 10, label: 'TF-IDF + Embeddings(Multi-Query) + Reranker + LLM' },
]

function GearIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z" />
      <path d="M12 14a2 2 0 1 0 0-4 2 2 0 0 0 0 4Z" />
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
    </svg>
  )
}

function SearchIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
    </svg>
  )
}

export function SearchView({ book, onBack }: { book: Book; onBack: () => void }) {
  const [query, setQuery] = useState('')
  const [preset, setPreset] = useState('large')
  const [pipeline, setPipeline] = useState(4)
  const [showSettings, setShowSettings] = useState(false)
  const settingsRef = useRef<HTMLDivElement>(null)

  // Close settings panel when clicking outside
  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (settingsRef.current && !settingsRef.current.contains(e.target as Node)) {
        setShowSettings(false)
      }
    }
    document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [])

  const { data: presetsData } = useQuery<{ presets: string[] }>({
    queryKey: ['presets', book.id],
    queryFn: () => fetch(`/api/books/${book.id}/presets`).then(r => r.json()),
  })

  const wrapRef = useRef<HTMLDivElement>(null)
  const [borderDims, setBorderDims] = useState({ w: 0, h: 0 })

  const [isPending, setIsPending] = useState(false)
  const [progress, setProgress] = useState<string | null>(null)
  const [results, setResults] = useState<SearchResult[] | null>(null)

  useEffect(() => {
    if (isPending && wrapRef.current) {
      setBorderDims({ w: wrapRef.current.offsetWidth, h: wrapRef.current.offsetHeight })
    }
  }, [isPending])

  const bR = 16
  const borderPerim = borderDims.w > 0
    ? 2 * (borderDims.w - 2 * bR) + 2 * (borderDims.h - 2 * bR) + 2 * Math.PI * bR
    : 0

  async function runSearch() {
    if (!query || isPending) return
    setIsPending(true)
    setResults(null)
    setProgress('Initialisiere Pipeline…')

    const response = await fetch('/api/search/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ book_id: book.id, preset_name: preset, pipeline, query, top_k: 5 }),
    })

    const reader = response.body!.getReader()
    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const event = JSON.parse(line.slice(6))
        if (event.type === 'progress') setProgress(event.message)
        else if (event.type === 'result') { setResults(event.data); setProgress(null) }
        else if (event.type === 'error') { setProgress(`Fehler: ${event.message}`); setIsPending(false) }
      }
    }
    setIsPending(false)
  }

  const title = BOOK_TITLES[book.name] ?? book.name

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        <button
          onClick={onBack}
          className="flex items-center gap-1.5 text-slate-500 hover:text-slate-200 transition-colors text-sm"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m15 18-6-6 6-6" /></svg>
          Zurück
        </button>
        <div className="h-4 w-px bg-slate-700" />
        <h2 className="text-slate-200 font-semibold">{title}</h2>
      </div>

      {/* Search input */}
      <div ref={wrapRef} className={`relative rounded-2xl transition-all duration-300 ${isPending ? '' : 'border border-slate-700 focus-within:border-blue-500/60 focus-within:ring-2 focus-within:ring-blue-500/10'}`}>
        <div className="bg-slate-800 rounded-2xl">
          <textarea
            autoFocus
            rows={1}
            className="w-full bg-transparent text-slate-100 placeholder-slate-500 outline-none text-sm resize-none leading-relaxed px-4 pt-4 pb-2 block"
            placeholder={`${title} durchsuchen…`}
            value={query}
            onChange={e => {
              setQuery(e.target.value)
              e.target.style.height = 'auto'
              e.target.style.height = e.target.scrollHeight + 'px'
            }}
            onKeyDown={e => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                if (query) runSearch()
              }
            }}
          />
          <div className="flex items-center justify-between px-3 pb-3">
            {/* Settings gear */}
            <div className="relative" ref={settingsRef}>
              <button
                onClick={() => setShowSettings(s => !s)}
                title="Einstellungen"
                className={`p-1.5 rounded-lg transition-all duration-150 ${
                  showSettings
                    ? 'text-blue-400 bg-blue-500/20'
                    : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700'
                }`}
              >
                <GearIcon />
              </button>

              {showSettings && (
                <div className="absolute left-0 bottom-full mb-2 w-80 bg-slate-800 border border-slate-700 rounded-2xl shadow-2xl shadow-black/50 p-5 space-y-4 z-20">
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Einstellungen</p>
                  <div className="space-y-1">
                    <label className="block text-xs text-slate-500">Chunk-Preset</label>
                    <select
                      className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500/60"
                      value={preset}
                      onChange={e => setPreset(e.target.value)}
                    >
                      {presetsData?.presets.map(p => <option key={p} value={p}>{p}</option>)}
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="block text-xs text-slate-500">Pipeline</label>
                    <select
                      className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-blue-500/60"
                      value={pipeline}
                      onChange={e => setPipeline(Number(e.target.value))}
                    >
                      {PIPELINES.map(p => (
                        <option key={p.value} value={p.value}>{p.value} — {p.label}</option>
                      ))}
                    </select>
                  </div>
                </div>
              )}
            </div>

            <button
              onClick={() => runSearch()}
              disabled={!query || isPending}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-30 disabled:cursor-not-allowed text-white text-sm font-medium rounded-xl transition-all duration-150"
            >
              {isPending ? (
                <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>
              ) : (
                <SearchIcon />
              )}
              {isPending ? 'Suche…' : 'Suchen'}
            </button>
          </div>
        </div>
        {isPending && borderPerim > 0 && (
          <>
            <style>{`@keyframes borderBeam { to { stroke-dashoffset: ${-borderPerim}; } }`}</style>
            <svg
              className="absolute inset-0 pointer-events-none"
              width={borderDims.w}
              height={borderDims.h}
              style={{ overflow: 'visible' }}
            >
              <defs>
                <filter id="beamGlow" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>
              <rect
                x={1} y={1}
                width={borderDims.w - 2} height={borderDims.h - 2}
                rx={bR - 1} ry={bR - 1}
                fill="none"
                stroke="#60a5fa"
                strokeWidth="2"
                strokeLinecap="round"
                filter="url(#beamGlow)"
                strokeDasharray={`${borderPerim * 0.08} ${borderPerim * 0.92}`}
                style={{ strokeDashoffset: 0, animation: 'borderBeam 2s linear infinite' }}
              />
            </svg>
          </>
        )}
      </div>

      {/* Pipeline step text */}
      {isPending && progress && (
        <p className="text-xs text-slate-500 pl-1">{progress}</p>
      )}

      {/* Results */}
      {results && (
        <div className="space-y-3">
          {results.map(r => (
            <div
              key={r.rank}
              className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5 hover:border-slate-600/50 transition-colors"
            >
              <div className="flex items-center gap-3 mb-3">
                <span className="text-xs font-bold text-blue-400 bg-blue-950/60 border border-blue-800/40 px-2.5 py-0.5 rounded-full">
                  #{r.rank}
                </span>
                <span className="text-xs text-slate-500 font-mono">score {r.score.toFixed(4)}</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{r.content}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
