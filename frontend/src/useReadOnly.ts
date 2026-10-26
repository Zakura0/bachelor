import { useQuery } from '@tanstack/react-query'

export function useReadOnly(): boolean | undefined {
  const { data } = useQuery<{ read_only: boolean }>({
    queryKey: ['config'],
    queryFn: () => fetch('/api/config').then(r => r.json()),
    staleTime: Infinity,
  })
  // undefined while loading: callers show write controls only on an explicit false
  return data?.read_only
}
