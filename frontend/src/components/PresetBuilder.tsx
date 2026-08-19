import { useState, useEffect, useRef } from 'react'
import { useQueryClient } from '@tanstack/react-query'

export const STANDARD_PRESETS = [
  { name: 'tiny',                min_words: 10,  max_words: 30,  overlap: 1 },
  { name: 'small',               min_words: 10,  max_words: 50,  overlap: 1 },
  { name: 'medium',              min_words: 30,  max_words: 100, overlap: 2 },
  { name: 'large',               min_words: 50,  max_words: 150, overlap: 2 },
  { name: 'xlarge',              min_words: 80,  max_words: 200, overlap: 3 },
  { name: 'medium_high_overlap', min_words: 30,  max_words: 100, overlap: 3 },
  { name: 'large_high_overlap',  min_words: 50,  max_words: 150, overlap: 4 },
]

export type IndexedPreset = { name: string; chunk_count: number }

export function CheckIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  )
}

export function SpinnerIcon({ size = 14 }: { size?: number }) {
  return (
    <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  )
}

// ── SSE progress display for a single preset indexing run ─────────────────

export function IndexProgress({ bookId, presetName, minWords, maxWords, overlap, onDone, label, autoStart }: {
  bookId: number; presetName: string; minWords: number; maxWords: number; overlap: number
  onDone: (chunkCount: number) => void; label?: string; autoStart?: boolean
}) {
  const [lines, setLines] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState(false)

  async function start() {
    setStarted(true)
    setLines([])
    setError(null)
    const res = await fetch(`/api/books/${bookId}/index`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ preset_name: presetName, min_words: minWords, max_words: maxWords, overlap }),
    })
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n')
      buf = parts.pop() ?? ''
      for (const line of parts) {
        if (!line.startsWith('data: ')) continue
        const evt = JSON.parse(line.slice(6))
        if (evt.type === 'progress') setLines(l => [...l, evt.message])
        else if (evt.type === 'done') { setLines(l => [...l, `✓ Fertig (${evt.chunk_count} Chunks)`]); onDone(evt.chunk_count) }
        else if (evt.type === 'error') { setError(evt.message); setStarted(false) }
      }
    }
  }

  const autoStarted = useRef(false)
  useEffect(() => {
    if (autoStart && !autoStarted.current) {
      autoStarted.current = true
      start()
    }
  }, [])

  if (!started) {
    return (
      <button onClick={start} className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-xl transition-colors">
        {label ?? 'Chunks & Embeddings erstellen'}
      </button>
    )
  }

  return (
    <div className="bg-slate-800/60 border border-slate-700/50 rounded-2xl p-4 space-y-2">
      {lines.map((l, i) => (
        <p key={i} className={`text-sm ${l.startsWith('✓') ? 'text-green-400' : 'text-slate-400'}`}>{l}</p>
      ))}
      {!error && lines.length > 0 && !lines.at(-1)?.startsWith('✓') && (
        <div className="flex items-center gap-2 text-slate-500 text-sm"><SpinnerIcon /> Läuft…</div>
      )}
      {error && <p className="text-red-400 text-sm">Fehler: {error}</p>}
    </div>
  )
}

// ── Preset configuration + indexing UI ───────────────────────────────────

export function AddPresetSection({ bookId, existingPresets, onPresetDone }: {
  bookId: number; existingPresets: string[]; onPresetDone: (preset: IndexedPreset) => void
}) {
  const [mode, setMode] = useState<'standard' | 'custom'>('standard')
  const [selectedStd, setSelectedStd] = useState(STANDARD_PRESETS[0].name)
  const [customName, setCustomName] = useState('')
  const [customMin, setCustomMin] = useState(30)
  const [customMax, setCustomMax] = useState(100)
  const [customOverlap, setCustomOverlap] = useState(2)
  const [indexing, setIndexing] = useState(false)
  const [donePreset, setDonePreset] = useState<IndexedPreset | null>(null)
  const queryClient = useQueryClient()

  const available = STANDARD_PRESETS.filter(p => !existingPresets.includes(p.name))

  const currentPreset = mode === 'standard'
    ? STANDARD_PRESETS.find(p => p.name === selectedStd)!
    : { name: customName, min_words: customMin, max_words: customMax, overlap: customOverlap }

  const canAdd = mode === 'standard'
    ? !!currentPreset && !existingPresets.includes(currentPreset.name)
    : !!customName && !existingPresets.includes(customName)

  function handleAddAnother() {
    setDonePreset(null)
    setIndexing(false)
    setCustomName('')
    queryClient.invalidateQueries({ queryKey: ['books'] })
    queryClient.invalidateQueries({ queryKey: ['presets', bookId] })
  }

  return (
    <div className="space-y-4">
      <div className="flex gap-2">
        {(['standard', 'custom'] as const).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${mode === m ? 'bg-blue-600 text-white' : 'bg-slate-700/60 text-slate-400 hover:text-slate-200'}`}>
            {m === 'standard' ? 'Standard-Preset' : 'Eigenes Preset'}
          </button>
        ))}
      </div>

      {mode === 'standard' ? (
        <select
          className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-blue-500/60"
          value={selectedStd}
          onChange={e => setSelectedStd(e.target.value)}
        >
          {available.length === 0
            ? <option value="">Alle Standard-Presets bereits erstellt</option>
            : available.map(p => <option key={p.name} value={p.name}>{p.name} ({p.min_words}–{p.max_words} Wörter, Overlap {p.overlap})</option>)
          }
        </select>
      ) : (
        <div className="grid grid-cols-2 gap-3">
          <div className="col-span-2">
            <label className="block text-xs text-slate-500 mb-1">Preset-Name</label>
            <input
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 placeholder-slate-500 rounded-xl px-3 py-2.5 text-sm font-mono outline-none focus:border-blue-500/60"
              placeholder="mein_preset"
              value={customName}
              onChange={e => setCustomName(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, ''))}
            />
          </div>
          {[
            { label: 'Min. Wörter', val: customMin, set: setCustomMin },
            { label: 'Max. Wörter', val: customMax, set: setCustomMax },
            { label: 'Overlap (Sätze)', val: customOverlap, set: setCustomOverlap },
          ].map(({ label, val, set }) => (
            <div key={label}>
              <label className="block text-xs text-slate-500 mb-1">{label}</label>
              <input type="number" min={1}
                className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-blue-500/60"
                value={val} onChange={e => set(Number(e.target.value))} />
            </div>
          ))}
        </div>
      )}

      {!indexing && !donePreset && (
        <button
          disabled={!canAdd || available.length === 0}
          onClick={() => setIndexing(true)}
          className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white text-sm font-medium rounded-xl transition-colors"
        >
          Chunks &amp; Embeddings erstellen
        </button>
      )}

      {indexing && !donePreset && (
        <IndexProgress
          bookId={bookId}
          presetName={currentPreset.name}
          minWords={currentPreset.min_words}
          maxWords={currentPreset.max_words}
          overlap={currentPreset.overlap}
          autoStart
          onDone={n => {
            const preset = { name: currentPreset.name, chunk_count: n }
            onPresetDone(preset)
            setDonePreset(preset)
          }}
        />
      )}

      {donePreset && (
        <div className="flex items-center justify-between bg-green-950/40 border border-green-800/40 rounded-xl px-4 py-3">
          <div className="flex items-center gap-2 text-green-400 text-sm"><CheckIcon /> &quot;{donePreset.name}&quot; erstellt ({donePreset.chunk_count} Chunks)</div>
          <button onClick={handleAddAnother} className="text-xs text-slate-400 hover:text-slate-200 ml-3 shrink-0">Weiteres</button>
        </div>
      )}
    </div>
  )
}
