import { useQuery } from '@tanstack/react-query'

type Book = { id: number; name: string; title: string }

export type { Book }

export function BookPicker({ onSelect, onUpload }: { onSelect: (book: Book) => void; onUpload: () => void }) {
  const { data: books, isLoading } = useQuery<Book[]>({
    queryKey: ['books'],
    queryFn: () => fetch('/api/books/').then(r => r.json()),
  })

  return (
    <div className="space-y-10">
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-4xl font-bold text-slate-100 tracking-tight">Summary-Source Alignment</h1>
          <p className="mt-3 text-slate-400 text-lg">Wähle ein Buch um die Suche zu starten.</p>
        </div>
        <button
          onClick={onUpload}
          title="Buch hochladen"
          className="mt-1 flex items-center gap-1.5 px-3 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-400 hover:text-slate-100 hover:border-slate-500 transition-all text-sm"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M5 12h14"/><path d="M12 5v14"/></svg>
          Buch
        </button>
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
