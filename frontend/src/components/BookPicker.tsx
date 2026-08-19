import { useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'

type Book = { id: number; name: string; title: string }

export type { Book }

// ── Delete modal ──────────────────────────────────────────────────────────

function DeleteModal({ books, onClose, onDeleted }: {
  books: Book[]
  onClose: () => void
  onDeleted: () => void
}) {
  const [selectedId, setSelectedId] = useState(books[0]?.id ?? -1)
  const [step, setStep] = useState<1 | 2>(1)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selected = books.find(b => b.id === selectedId)

  async function handleDelete() {
    if (!selected) return
    setLoading(true)
    const res = await fetch(`/api/books/${selected.id}`, { method: 'DELETE' })
    setLoading(false)
    if (!res.ok) { setError((await res.json()).detail ?? 'Fehler'); return }
    onDeleted()
  }

  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 w-full max-w-sm space-y-5" onClick={e => e.stopPropagation()}>
        {step === 1 ? (
          <>
            <p className="text-slate-100 font-semibold">Buch löschen</p>
            <select
              className="w-full bg-slate-700 border border-slate-600 text-slate-100 rounded-xl px-3 py-2.5 text-sm outline-none"
              value={selectedId}
              onChange={e => setSelectedId(Number(e.target.value))}
            >
              {books.map(b => <option key={b.id} value={b.id}>{b.title}</option>)}
            </select>
            <div className="flex gap-3 justify-end">
              <button onClick={onClose} className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors">Abbrechen</button>
              <button onClick={() => setStep(2)} disabled={!selected} className="px-4 py-2 bg-red-700 hover:bg-red-600 disabled:opacity-30 text-white text-sm font-medium rounded-xl transition-colors">Weiter</button>
            </div>
          </>
        ) : (
          <>
            <p className="text-slate-100 font-semibold">Bist du sicher?</p>
            <p className="text-slate-400 text-sm">
              <span className="text-slate-200 font-medium">{selected?.title}</span> und alle zugehörigen Chunks und Embeddings werden unwiderruflich gelöscht.
            </p>
            {error && <p className="text-red-400 text-sm">{error}</p>}
            <div className="flex gap-3 justify-end">
              <button onClick={() => setStep(1)} className="px-4 py-2 text-sm text-slate-400 hover:text-slate-200 transition-colors">Zurück</button>
              <button onClick={handleDelete} disabled={loading} className="px-4 py-2 bg-red-700 hover:bg-red-600 disabled:opacity-50 text-white text-sm font-medium rounded-xl transition-colors">
                {loading ? 'Löschen…' : 'Ja, löschen'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

// ── Book picker ───────────────────────────────────────────────────────────

export function BookPicker({ onSelect, onUpload }: { onSelect: (book: Book) => void; onUpload: () => void }) {
  const [showDelete, setShowDelete] = useState(false)
  const queryClient = useQueryClient()
  const { data: books, isLoading } = useQuery<Book[]>({
    queryKey: ['books'],
    queryFn: () => fetch('/api/books/').then(r => r.json()),
  })

  return (
    <div className="space-y-10">
      {showDelete && books && books.length > 0 && (
        <DeleteModal
          books={books}
          onClose={() => setShowDelete(false)}
          onDeleted={() => { setShowDelete(false); queryClient.invalidateQueries({ queryKey: ['books'] }) }}
        />
      )}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-100 tracking-tight">Summary-Source Alignment</h1>
          <p className="mt-3 text-slate-400 text-lg">Wähle ein Buch um die Suche zu starten.</p>
        </div>
        <div className="flex items-center gap-2 mt-1">
          <button
            onClick={onUpload}
            title="Buch hochladen"
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 hover:text-slate-100 hover:border-slate-500 transition-all text-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
            Buch
          </button>
          <button
            onClick={() => setShowDelete(true)}
            title="Buch löschen"
            disabled={!books || books.length === 0}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 hover:text-red-400 hover:border-red-800/60 disabled:opacity-30 transition-all text-sm"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/></svg>
            Buch
          </button>
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-32 bg-slate-800/40 rounded-2xl animate-pulse" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-4">
          {books?.map(book => (
            <button
              key={book.id}
              onClick={() => onSelect(book)}
              className="group text-left bg-slate-800/60 border border-slate-700/50 hover:border-blue-500/50 hover:bg-slate-700/50 rounded-2xl p-6 transition-all duration-200 hover:shadow-xl hover:shadow-blue-500/10 hover:-translate-y-0.5 cursor-pointer"
            >
              <div className="text-3xl mb-4">📖</div>
              <div className="font-semibold text-slate-100 group-hover:text-blue-300 transition-colors text-sm leading-snug">
                {book.title || book.name}
              </div>
              <div className="mt-1 text-xs text-slate-600 font-mono">{book.name}</div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
