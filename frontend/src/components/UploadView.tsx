import { useState, useRef, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'

// Standard presets matching chunk_presets.py
const STANDARD_PRESETS = [
  { name: 'tiny',                min_words: 10,  max_words: 30,  overlap: 1 },
  { name: 'small',               min_words: 10,  max_words: 50,  overlap: 1 },
  { name: 'medium',              min_words: 30,  max_words: 100, overlap: 2 },
  { name: 'xlarge',              min_words: 80,  max_words: 200, overlap: 3 },
  { name: 'medium_high_overlap', min_words: 30,  max_words: 100, overlap: 3 },
  { name: 'large_high_overlap',  min_words: 50,  max_words: 150, overlap: 4 },
]

type IndexedPreset = { name: string; chunk_count: number }
type Book = { id: number; name: string; title: string }
type FileData = { title: string; name: string; rawText: string }

function BackIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m15 18-6-6 6-6" />
    </svg>
  )
}

function CheckIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 6 9 17l-5-5" />
    </svg>
  )
}

function SpinnerIcon({ size = 14 }: { size?: number }) {
  return (
    <svg className="animate-spin" xmlns="http://www.w3.org/2000/svg" width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  )
}

// ── Phase 1: File + title form (no API call) ──────────────────────────────

function UploadForm({ onReady }: { onReady: (data: FileData) => void }) {
  const [file, setFile] = useState<File | null>(null)
  const [title, setTitle] = useState('')
  const [name, setName] = useState('')
  const [loading, setLoading] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const { data: existingBooks } = useQuery<{ name: string }[]>({
    queryKey: ['books'],
    queryFn: () => fetch('/api/books/').then(r => r.json()),
  })

  const nameConflict = !!name && existingBooks?.some(b => b.name === name)

  function handleFile(f: File) {
    setFile(f)
    if (!title) setTitle(f.name.replace(/\.txt$/i, ''))
    setName(f.name.replace(/\.txt$/i, '').toLowerCase().replace(/[^a-z0-9]/g, ''))
  }

  async function handleSubmit() {
    if (!file || !title || nameConflict) return
    setLoading(true)
    const rawText = await file.text()
    setLoading(false)
    onReady({ title, name, rawText })
  }

  return (
    <div className="space-y-6">
      {/* Drop zone */}
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={e => e.preventDefault()}
        onDrop={e => { e.preventDefault(); const f = e.dataTransfer.files[0]; if (f) handleFile(f) }}
        className="border-2 border-dashed border-slate-700 hover:border-blue-500/60 rounded-2xl p-10 text-center cursor-pointer transition-colors"
      >
        <input ref={inputRef} type="file" accept=".txt" className="hidden" onChange={e => { const f = e.target.files?.[0]; if (f) handleFile(f) }} />
        {file ? (
          <div className="space-y-1">
            <p className="text-slate-200 font-medium">{file.name}</p>
            <p className="text-slate-500 text-sm">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
        ) : (
          <div className="space-y-2">
            <svg className="mx-auto text-slate-600" xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            <p className="text-slate-400 text-sm">Klicken oder .txt-Datei hineinziehen</p>
          </div>
        )}
      </div>

      <div className="space-y-3">
        <div>
          <label className="block text-xs text-slate-500 mb-1">Titel</label>
          <input
            className="w-full bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-blue-500/60"
            placeholder="z.B. Die Verwandlung"
            value={title}
            onChange={e => setTitle(e.target.value)}
          />
        </div>
        <div>
          <label className="block text-xs text-slate-500 mb-1">Interner Name (Slug)</label>
          <div className={`flex items-center gap-2 bg-slate-800/60 border rounded-xl px-3 py-2.5 ${nameConflict ? 'border-red-700/70' : 'border-slate-700'}`}>
            <span className="flex-1 text-sm font-mono text-slate-400">{name || <span className="text-slate-600">— wird aus Dateiname abgeleitet —</span>}</span>
            {nameConflict && (
              <span className="text-xs text-red-400 shrink-0">bereits vorhanden</span>
            )}
          </div>
        </div>
      </div>

      <button
        disabled={!file || !title || loading || nameConflict}
        onClick={handleSubmit}
        className="w-full py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white text-sm font-medium rounded-xl transition-colors flex items-center justify-center gap-2"
      >
        {loading ? <><SpinnerIcon /> Lesen…</> : 'Weiter'}
      </button>
    </div>
  )
}

// ── Phase 2: atomically create book + large preset ─────────────────────────

function CreateAndIndexProgress({ fileData, onDone }: {
  fileData: FileData
  onDone: (book: Book, chunkCount: number) => void
}) {
  const [lines, setLines] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [started, setStarted] = useState(false)

  async function start() {
    setStarted(true)
    setLines([])
    setError(null)
    const res = await fetch('/api/books/create-and-index', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: fileData.title, name: fileData.name, raw_text: fileData.rawText,
        preset_name: 'large', min_words: 50, max_words: 150, overlap: 2,
      }),
    })
    const reader = res.body!.getReader()
    const decoder = new TextDecoder()
    let buf = ''
    let createdBook: Book | null = null
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const parts = buf.split('\n')
      buf = parts.pop() ?? ''
      for (const line of parts) {
        if (!line.startsWith('data: ')) continue
        const evt = JSON.parse(line.slice(6))
        if (evt.type === 'book_created') createdBook = { id: evt.id, name: evt.name, title: evt.title }
        else if (evt.type === 'progress') setLines(l => [...l, evt.message])
        else if (evt.type === 'done') {
          setLines(l => [...l, `✓ Fertig (${evt.chunk_count} Chunks)`])
          if (createdBook) onDone(createdBook, evt.chunk_count)
        } else if (evt.type === 'error') { setError(evt.message); setStarted(false) }
      }
    }
  }

  if (!started) {
    return (
      <button onClick={start} className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-xl transition-colors">
        Chunks &amp; Embeddings erstellen
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

// ── Preset indexing progress display ───────────────────────────────────────

function IndexProgress({ bookId, presetName, minWords, maxWords, overlap, onDone, label, autoStart }:
  { bookId: number; presetName: string; minWords: number; maxWords: number; overlap: number; onDone: (chunkCount: number) => void; label?: string; autoStart?: boolean }
) {
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

  // auto-start when used inside AddPresetSection
  useEffect(() => { if (autoStart) start() }, [])

  if (!started) {
    return (
      <button
        onClick={start}
        className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-xl transition-colors"
      >
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

// ── Phase 2: Add extra presets ─────────────────────────────────────────────

function AddPresetSection({ bookId, existingPresets, onPresetDone }: {
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
  }

  return (
    <div className="space-y-4">
      <p className="text-sm font-semibold text-slate-400 uppercase tracking-wider">Weiteres Preset hinzufügen</p>

      {/* Mode toggle */}
      <div className="flex gap-2">
        {(['standard', 'custom'] as const).map(m => (
          <button key={m} onClick={() => setMode(m)}
            className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${mode === m ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400 hover:text-slate-200'}`}>
            {m === 'standard' ? 'Standard-Preset' : 'Eigenes Preset'}
          </button>
        ))}
      </div>

      {mode === 'standard' ? (
        <select
          className="w-full bg-slate-800 border border-slate-700 text-slate-100 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-blue-500/60"
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
              className="w-full bg-slate-800 border border-slate-700 text-slate-100 placeholder-slate-500 rounded-xl px-3 py-2.5 text-sm font-mono outline-none focus:border-blue-500/60"
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
                className="w-full bg-slate-800 border border-slate-700 text-slate-100 rounded-xl px-3 py-2.5 text-sm outline-none focus:border-blue-500/60"
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
          <div className="flex items-center gap-2 text-green-400 text-sm"><CheckIcon /> Preset &quot;{donePreset.name}&quot; erstellt</div>
          <button onClick={handleAddAnother} className="text-xs text-slate-400 hover:text-slate-200">Weiteres hinzufügen</button>
        </div>
      )}
    </div>
  )
}

// ── Main view ──────────────────────────────────────────────────────────────

export function UploadView({ onBack }: { onBack: () => void }) {
  const [fileData, setFileData] = useState<FileData | null>(null)
  const [book, setBook] = useState<Book | null>(null)
  const [indexedPresets, setIndexedPresets] = useState<IndexedPreset[]>([])
  const queryClient = useQueryClient()

  const hasBasePreset = indexedPresets.length > 0

  function handleCreateDone(b: Book, chunkCount: number) {
    setBook(b)
    setIndexedPresets([{ name: 'large', chunk_count: chunkCount }])
    queryClient.invalidateQueries({ queryKey: ['books'] })
  }

  function handlePresetDone(preset: IndexedPreset) {
    setIndexedPresets(ps => [...ps, preset])
  }

  // Back behaviour depends on phase
  const headerBack = !book
    ? () => { if (fileData) setFileData(null); else onBack() }
    : null

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-4">
        {headerBack && (
          <>
            <button onClick={headerBack} className="flex items-center gap-1.5 text-slate-500 hover:text-slate-200 transition-colors text-sm">
              <BackIcon /> Zurück
            </button>
            <div className="h-4 w-px bg-slate-700" />
          </>
        )}
        <h2 className="text-slate-200 font-semibold">Buch hochladen</h2>
      </div>

      {/* Phase 1: collect file + metadata */}
      {!fileData && <UploadForm onReady={setFileData} />}

      {/* Phase 2: confirm + run create-and-index */}
      {fileData && !book && (
        <div className="space-y-4">
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5">
            <p className="text-slate-100 font-semibold">{fileData.title}</p>
            <p className="text-slate-500 text-xs font-mono mt-0.5">{fileData.name}</p>
            <p className="text-slate-600 text-xs mt-1">{(fileData.rawText.length / 1000).toFixed(0)} KB Text</p>
          </div>
          <CreateAndIndexProgress fileData={fileData} onDone={handleCreateDone} />
        </div>
      )}

      {/* Phase 3: book created — add more presets */}
      {book && (
        <div className="space-y-6">
          <div className="bg-slate-800/50 border border-slate-700/50 rounded-2xl p-5">
            <p className="text-slate-100 font-semibold">{book.title}</p>
            <p className="text-slate-500 text-xs font-mono mt-0.5">{book.name}</p>
          </div>

          <div className="space-y-2">
            {indexedPresets.map(p => (
              <div key={p.name} className="flex items-center gap-2 text-sm text-green-400">
                <CheckIcon /> <span className="font-mono">{p.name}</span>
                <span className="text-slate-500">({p.chunk_count} Chunks)</span>
              </div>
            ))}
          </div>

          {hasBasePreset && (
            <AddPresetSection
              bookId={book.id}
              existingPresets={indexedPresets.map(p => p.name)}
              onPresetDone={handlePresetDone}
            />
          )}

          <button
            onClick={onBack}
            className="w-full py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium rounded-xl transition-colors"
          >
            Fertig
          </button>
        </div>
      )}
    </div>
  )
}
