import { useQuery } from '@tanstack/react-query'

type Book = {
  id: number
  name: string
}

export function BookList() {
  const { data: books, isLoading, error } = useQuery<Book[]>({
    queryKey: ['books'],
    queryFn: () => fetch('/api/books/').then(r => r.json()),
  })

  if (isLoading) return <p className="text-gray-500">Lade Bücher...</p>
  if (error) return <p className="text-red-500">Fehler beim Laden der Bücher.</p>

  return (
    <ul className="space-y-2">
      {books?.map(book => (
        <li key={book.id} className="px-4 py-2 bg-white border border-gray-200 rounded-lg">
          {book.name}
        </li>
      ))}
    </ul>
  )
}
