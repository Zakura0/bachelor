import { useQuery } from '@tanstack/react-query'

export function useReadOnly(): boolean {
  const { data } = useQuery<{ read_only: boolean }>({
    queryKey: ['config'],
    queryFn: () => fetch('/api/config').then(r => r.json()),
    staleTime: Infinity,
  })
  // Until the config is loaded, assume read-only so write controls don't flash up
  return data?.read_only ?? true
}
