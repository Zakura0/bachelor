import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'

type Book = { id: number; name: string }

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

export function SearchForm() {
  // Formular-State: je ein useState pro Feld
  const [bookId, setBookId] = useState<number | null>(null)
  const [preset, setPreset] = useState('')
  const [pipeline, setPipeline] = useState(4)
  const [query, setQuery] = useState('')

  // Bücher laden — läuft automatisch beim ersten Render
  const { data: books } = useQuery<Book[]>({
    queryKey: ['books'],
    queryFn: () => fetch('/api/books/').then(r => r.json()),
  })

  // Presets nur laden wenn ein Buch gewählt ist (enabled: !!bookId)
  const { data: presetsData } = useQuery<{ presets: string[] }>({
    queryKey: ['presets', bookId],
    queryFn: () => fetch(`/api/books/${bookId}/presets`).then(r => r.json()),
    enabled: !!bookId,
  })

  // Suche — wird erst ausgelöst wenn der User den Button drückt
  const { mutate: runSearch, data: results, isPending } = useMutation<SearchResult[], Error>({
    mutationFn: () =>
      fetch('/api/search/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ book_id: bookId, preset_name: preset, pipeline, query, top_k: 5 }),
      }).then(r => r.json()),
  })

  const canSearch = !!bookId && !!preset && !!query

  function handleBookChange(id: number) {
    setBookId(id)
    setPreset('') // Preset zurücksetzen wenn anderes Buch gewählt
  }

  return (
    <div className="space-y-6">
      {/* Formular */}
      <div className="bg-slate-800 border border-slate-700 rounded-xl p-6 space-y-4">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Buch</label>
            <select
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm"
              value={bookId ?? ''}
              onChange={e => handleBookChange(Number(e.target.value))}
            >
              <option value="">— wählen —</option>
              {books?.map(b => <option key={b.id} value={b.id}>{b.name}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Preset</label>
            <select
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm disabled:opacity-40"
              value={preset}
              onChange={e => setPreset(e.target.value)}
              disabled={!bookId}
            >
              <option value="">— wählen —</option>
              {presetsData?.presets.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">Pipeline</label>
            <select
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-lg px-3 py-2 text-sm"
              value={pipeline}
              onChange={e => setPipeline(Number(e.target.value))}
            >
              {PIPELINES.map(p => <option key={p.value} value={p.value}>{p.value} — {p.label}</option>)}
            </select>
          </div>
        </div>

        <div className="flex gap-3">
          <input
            className="flex-1 bg-slate-700 border border-slate-600 text-slate-100 placeholder-slate-400 rounded-lg px-3 py-2 text-sm"
            type="text"
            placeholder="Suchanfrage eingeben..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && canSearch && runSearch()}
          />
          <button
            className="px-5 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg disabled:opacity-40 hover:bg-blue-500"
            onClick={() => runSearch()}
            disabled={!canSearch || isPending}
          >
            {isPending ? 'Suche...' : 'Suchen'}
          </button>
        </div>
      </div>

      {/* Ergebnisse */}
      {results && (
        <ul className="space-y-3">
          {results.map(r => (
            <li key={r.rank} className="bg-slate-800 border border-slate-700 rounded-xl p-4">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xs font-bold text-blue-400 bg-blue-950 px-2 py-0.5 rounded-full">
                  #{r.rank}
                </span>
                <span className="text-xs text-slate-400">Score: {r.score.toFixed(4)}</span>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">{r.content}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
