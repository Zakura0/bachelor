import { useState } from 'react'
import { BookPicker, type Book } from './components/BookPicker'
import { SearchView } from './components/SearchView'

function App() {
  const [selectedBook, setSelectedBook] = useState<Book | null>(null)

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <main className="max-w-3xl mx-auto px-8 py-16">
        {selectedBook ? (
          <SearchView book={selectedBook} onBack={() => setSelectedBook(null)} />
        ) : (
          <BookPicker onSelect={setSelectedBook} />
        )}
      </main>
    </div>
  )
}

export default App
