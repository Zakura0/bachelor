import { useEffect, useRef } from 'react'

export function TextHighlight({
  text,
  startIndex,
  endIndex,
}: {
  text: string
  startIndex: number
  endIndex: number
}) {
  const highlightRef = useRef<HTMLElement>(null)

  useEffect(() => {
    highlightRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [startIndex, endIndex])

  return (
    <div className="h-[65vh] overflow-y-auto rounded-2xl bg-slate-900/50 border border-slate-700/50 px-7 py-6">
      <p className="text-sm text-slate-400 leading-7 whitespace-pre-wrap">
        {text.slice(0, startIndex)}
        <mark
          ref={highlightRef}
          className="bg-yellow-400/25 text-yellow-100 not-italic rounded-sm"
          style={{ boxShadow: '0 0 0 2px rgba(250,204,21,0.15)' }}
        >
          {text.slice(startIndex, endIndex)}
        </mark>
        {text.slice(endIndex)}
      </p>
    </div>
  )
}
