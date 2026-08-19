import { SearchForm } from './components/SearchForm'

function App() {
  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <header className="bg-slate-800 border-b border-slate-700 px-8 py-4">
        <h1 className="text-xl font-semibold text-slate-100">IR Bibliothek</h1>
      </header>
      <main className="max-w-4xl mx-auto px-8 py-8">
        <SearchForm />
      </main>
    </div>
  )
}

export default App
