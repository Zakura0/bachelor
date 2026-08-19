import { useState } from 'react'
import { BookPicker, type Book } from './components/BookPicker'
import { SearchView } from './components/SearchView'
import { UploadView } from './components/UploadView'

type View = { kind: 'list' } | { kind: 'search'; book: Book } | { kind: 'upload' }

function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/60 z-50 flex items-end justify-center p-6" onClick={onClose}>
      <div
        className="w-full max-w-lg bg-slate-800 border border-slate-700 rounded-2xl p-8 space-y-5 mb-4"
        onClick={e => e.stopPropagation()}
      >
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-slate-100 font-semibold text-lg">Summary-Source Alignment</h2>
            <p className="text-slate-500 text-xs mt-0.5">Bachelorarbeit · Universität Hamburg</p>
          </div>
          <button onClick={onClose} className="text-slate-600 hover:text-slate-300 transition-colors text-xl leading-none">×</button>
        </div>
        <p className="text-slate-400 text-sm leading-relaxed">
          Diese Webapp ist Teil einer Bachelorarbeit zur <span className="text-slate-200">Evaluierung von Information-Retrieval-Pipelines</span> für die Aufgabe der Summary-Source-Alignment: Gegeben eine abstrakte Textzusammenfassung, finde die entsprechende Originalpassage im Quelltext.
        </p>
        <div className="grid grid-cols-2 gap-3 text-xs">
          {[
            ['Retrieval', 'TF-IDF · Embeddings (E5-large)'],
            ['Reranking', 'Cross-Encoder · LLM (Gemma 4)'],
            ['Fusion', 'Reciprocal Rank Fusion'],
            ['Verifikation', 'LLM (Gemma 4)'],
          ].map(([label, value]) => (
            <div key={label} className="bg-slate-700/40 rounded-xl px-3 py-2.5">
              <p className="text-slate-500">{label}</p>
              <p className="text-slate-300 mt-0.5">{value}</p>
            </div>
          ))}
        </div>
        <p className="text-slate-600 text-xs">© 2026 · <a href="https://github.com/zakura0" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-slate-400 transition-colors">Simon Kazemi<svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg></a></p>
      </div>
    </div>
  )
}

function App() {
  const [view, setView] = useState<View>({ kind: 'list' })
  const [showAbout, setShowAbout] = useState(false)

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 flex flex-col">
      <main className="flex-1 max-w-3xl w-full mx-auto px-8 py-16">
        {view.kind === 'list' && (
          <BookPicker
            onSelect={book => setView({ kind: 'search', book })}
            onUpload={() => setView({ kind: 'upload' })}
          />
        )}
        {view.kind === 'search' && (
          <SearchView book={view.book} onBack={() => setView({ kind: 'list' })} />
        )}
        {view.kind === 'upload' && (
          <UploadView onBack={() => setView({ kind: 'list' })} />
        )}
      </main>

      <footer className="flex items-center justify-center gap-4 pb-6 text-xs text-slate-700">
        <a href="https://github.com/zakura0" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 hover:text-slate-500 transition-colors">
          zakura0
          <svg xmlns="http://www.w3.org/2000/svg" width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
        </a>
        <span>·</span>
        <button onClick={() => setShowAbout(true)} className="hover:text-slate-500 transition-colors">Über diese App</button>
      </footer>

      {showAbout && <AboutModal onClose={() => setShowAbout(false)} />}
    </div>
  )
}

export default App
