import { BookList } from './components/BookList'

function App() {
  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <h1 className="text-2xl font-bold text-gray-800 mb-6">IR Bibliothek</h1>
      <BookList />
    </div>
  )
}

export default App
