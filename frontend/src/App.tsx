import { useState } from 'react'
import { BookPicker, type Book } from './components/BookPicker'
import { SearchView } from './components/SearchView'
import { UploadView } from './components/UploadView'

type View = { kind: 'list' } | { kind: 'search'; book: Book } | { kind: 'upload' }

function App() {
  const [view, setView] = useState<View>({ kind: 'list' })

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <main className="max-w-3xl mx-auto px-8 py-16">
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
    </div>
  )
}

export default App
