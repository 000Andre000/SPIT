import { useEffect, useRef, useState } from 'react'
import './App.css'
import UGVPipelineGraph from './UGVPipelineGraph'

function App() {
  const [endpoint, setEndpoint] = useState('http://localhost:8000/predict/video')
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState<string | null>(null)
  const [resultBlobUrl, setResultBlobUrl] = useState<string | null>(null)
  const [status, setStatus] = useState('READY')
  const [fixedVideoState, setFixedVideoState] = useState('No file')
  const [loadingTitle, setLoadingTitle] = useState('Processing...')
  const [loadingSub, setLoadingSub] = useState('Please wait while the video is processed')
  const [showGlobalLoading, setShowGlobalLoading] = useState(false)
  const [showInlineLoading, setShowInlineLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [errorMessage, setErrorMessage] = useState('')
  const [showResultVideo, setShowResultVideo] = useState(false)

  const isRequestActiveRef = useRef(false)
  const fixedVideoRef = useRef<HTMLVideoElement | null>(null)
  const resultVideoRef = useRef<HTMLVideoElement | null>(null)
  const progressFillRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    return () => {
      if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl)
      if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl)
    }
  }, [uploadedPreviewUrl, resultBlobUrl])

  useEffect(() => {
    if (progressFillRef.current) {
      progressFillRef.current.style.width = `${progress}%`
    }
  }, [progress])

  const setFixedVideoSource = (url: string | null, stateText: string, tryPlay = false) => {
    const player = fixedVideoRef.current
    if (!player) return
    player.pause()
    player.src = url || ''
    player.load()
    setFixedVideoState(stateText)
    if (url && tryPlay) {
      player.play().catch(() => {})
    }
  }

  const handleFile = (file: File) => {
    if (!file.type.startsWith('video/')) {
      setErrorMessage('Only video files are accepted')
      return
    }

    if (uploadedPreviewUrl) URL.revokeObjectURL(uploadedPreviewUrl)
    if (resultBlobUrl) {
      URL.revokeObjectURL(resultBlobUrl)
      setResultBlobUrl(null)
    }

    const previewUrl = URL.createObjectURL(file)
    setUploadedPreviewUrl(previewUrl)
    setFixedVideoSource(previewUrl, 'Uploaded video', true)
    setSelectedFile(file)
    setStatus('READY')
    setProgress(0)
    setErrorMessage('')
    setShowResultVideo(false)
  }

  const onFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) handleFile(file)
  }

  const sendFile = async () => {
    if (!selectedFile || isProcessing) return

    const requestId = (window.crypto && 'randomUUID' in window.crypto)
      ? window.crypto.randomUUID()
      : `req_${Date.now()}_${Math.floor(Math.random() * 1000000)}`

    setIsProcessing(true)
    setStatus('UPLOADING')
    setProgress(0)
    setShowInlineLoading(true)
    setShowGlobalLoading(true)
    setLoadingTitle('Uploading...')
    setLoadingSub('Sending video to backend')
    setFixedVideoState('Uploading')
    setErrorMessage('')
    setShowResultVideo(false)
    isRequestActiveRef.current = true

    const formData = new FormData()
    formData.append('file', selectedFile)

    const progressUrl = (() => {
      try {
        const endpointUrl = new URL(endpoint, window.location.href)
        return `${endpointUrl.origin}/predict/progress/${encodeURIComponent(requestId)}`
      } catch {
        return ''
      }
    })()

    try {
      const xhr = await new Promise<XMLHttpRequest>((resolve, reject) => {
        const req = new XMLHttpRequest()
        let processingInterval: number | null = null
        let progressPoller: number | null = null
        let inProcessingState = false

        const stopProgressPolling = () => {
          if (progressPoller) {
            window.clearInterval(progressPoller)
            progressPoller = null
          }
        }

        const pollProgress = async () => {
          if (!isRequestActiveRef.current || !progressUrl) return
          try {
            const response = await fetch(progressUrl, { cache: 'no-store' })
            if (!isRequestActiveRef.current || !response.ok) return
            const data = await response.json() as {
              percent?: number
              frame?: number
              total_frames?: number
              message?: string
            }
            if (!isRequestActiveRef.current) return

            const percent = Number(data.percent || 0)
            const frame = Number(data.frame || 0)
            const total = Number(data.total_frames || 0)
            if (total > 0) {
              const text = `Frame ${frame}/${total} (${percent}%)`
              setLoadingSub(text)
              setLoadingTitle('Processing...')
            } else if (data.message) {
              setLoadingSub(data.message)
              setLoadingTitle('Processing...')
            }
            if (percent > 0) {
              setProgress(Math.min(95, 30 + Math.round(percent * 0.65)))
            }
          } catch {
          }
        }

        const enterProcessingState = () => {
          if (!isRequestActiveRef.current || inProcessingState) return
          inProcessingState = true
          setStatus('PROCESSING')
          setFixedVideoState('Processing')
          setLoadingTitle('Processing...')
          setLoadingSub('AI is segmenting video frames')
          setProgress(30)
          processingInterval = window.setInterval(() => {
            setProgress((current) => (current < 95 ? Math.min(95, current + Math.random() * 8 + 2) : current))
          }, 1200)
          pollProgress()
          progressPoller = window.setInterval(pollProgress, 800)
        }

        req.upload.addEventListener('progress', (event) => {
          if (event.lengthComputable) {
            setProgress(Math.round((event.loaded / event.total) * 30))
          }
        })
        req.upload.addEventListener('load', enterProcessingState)
        req.onreadystatechange = () => {
          if (req.readyState === 2) enterProcessingState()
        }

        req.onload = () => {
          isRequestActiveRef.current = false
          if (processingInterval) window.clearInterval(processingInterval)
          stopProgressPolling()
          resolve(req)
        }
        req.onerror = () => {
          isRequestActiveRef.current = false
          if (processingInterval) window.clearInterval(processingInterval)
          stopProgressPolling()
          reject(new Error(`Network error. Ensure FastAPI server is running at ${endpoint}`))
        }
        req.onabort = () => {
          isRequestActiveRef.current = false
          if (processingInterval) window.clearInterval(processingInterval)
          stopProgressPolling()
          reject(new Error('Request aborted'))
        }
        req.ontimeout = () => {
          isRequestActiveRef.current = false
          if (processingInterval) window.clearInterval(processingInterval)
          stopProgressPolling()
          reject(new Error('Request timed out while waiting for processing to finish'))
        }

        req.open('POST', endpoint)
        req.setRequestHeader('X-Request-ID', requestId)
        req.responseType = 'blob'
        req.timeout = 600000
        req.send(formData)
      })

      if (xhr.status < 200 || xhr.status >= 300) {
        throw new Error(`Server error: ${xhr.status} ${xhr.statusText || 'Request failed'}`)
      }

      const responseType = (xhr.getResponseHeader('content-type') || '').toLowerCase()
      const responseBlob = xhr.response
      if (!responseType.includes('video/')) {
        const text = await responseBlob.text()
        let message = text || 'Server returned an invalid response'
        try {
          const parsed = JSON.parse(text) as { detail?: string; error?: string }
          message = parsed.detail || parsed.error || message
        } catch {
        }
        throw new Error(message)
      }

      if (resultBlobUrl) URL.revokeObjectURL(resultBlobUrl)

      const processedUrl = URL.createObjectURL(responseBlob)
      setResultBlobUrl(processedUrl)
      setProgress(100)
      setStatus('READY')
      setShowInlineLoading(false)
      setShowGlobalLoading(false)

      const player = resultVideoRef.current
      if (player) {
        player.src = processedUrl
        player.load()
      }

      const previewReady = await new Promise<boolean>((resolve) => {
        if (!player) {
          resolve(false)
          return
        }
        let settled = false
        let readinessMonitor: number | null = null
        const cleanup = () => {
          player.removeEventListener('loadedmetadata', onReady)
          player.removeEventListener('loadeddata', onReady)
          player.removeEventListener('canplay', onReady)
          player.removeEventListener('canplaythrough', onReady)
          player.removeEventListener('error', onError)
          if (readinessMonitor) window.clearInterval(readinessMonitor)
        }
        const finish = (ok: boolean) => {
          if (settled) return
          settled = true
          cleanup()
          resolve(ok)
        }
        const onReady = () => finish(true)
        const onError = () => finish(false)
        readinessMonitor = window.setInterval(() => {
          if (player.readyState >= 2 && player.videoWidth > 0) finish(true)
        }, 250)
        player.addEventListener('loadedmetadata', onReady)
        player.addEventListener('loadeddata', onReady)
        player.addEventListener('canplay', onReady)
        player.addEventListener('canplaythrough', onReady)
        player.addEventListener('error', onError)
        window.setTimeout(() => finish(false), 10000)
      })

      setFixedVideoSource(processedUrl, previewReady ? 'Processed output' : 'Processed output (download)', true)
      setShowResultVideo(previewReady)
      if (previewReady && player) {
        player.play().catch(() => {})
      }
    } catch (error) {
      isRequestActiveRef.current = false
      const message = error instanceof Error ? error.message : 'Unknown error'
      setStatus('ERROR')
      setLoadingTitle('Error occurred')
      setLoadingSub(message)
      setErrorMessage(message)
      setShowGlobalLoading(false)
      setShowInlineLoading(false)
      setFixedVideoState('Error')
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="app-root">
      <header className="topbar">
        <div className="logo">UGV · <span>AI</span> PIPELINE TREE (React)</div>
      </header>

      <main className="main">
        <section className="left-pane">
          <div className="pipeline-panel">
            <UGVPipelineGraph />
          </div>
        </section>

        <section className="right-pane">
          <div className="panel-header">
            <div>SEG OVERLAY OUTPUT</div>
            <div className={`badge ${isProcessing ? 'processing' : ''}`}>{status}</div>
          </div>

          <div className="control-row">
            <label htmlFor="endpoint">API Endpoint</label>
            <input
              id="endpoint"
              value={endpoint}
              onChange={(event) => setEndpoint(event.target.value)}
              placeholder="http://localhost:8000/predict/video"
            />
          </div>

          <div className="control-row">
            <label htmlFor="file-input">Video File</label>
            <input id="file-input" type="file" accept="video/*" onChange={onFileInputChange} />
          </div>

          <div className="file-info">{selectedFile ? `${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)` : 'No file selected'}</div>

          <button className="predict-btn" onClick={sendFile} disabled={!selectedFile || isProcessing}>
            {isProcessing ? 'PROCESSING...' : 'PREDICT'}
          </button>

          <div className="progress-track"><div ref={progressFillRef} className="progress-fill" /></div>

          <div className="result-card">
            <div className="result-title">Processed Output</div>
            <div className="result-media-wrap">
              {showInlineLoading && <div className="inline-loading">{loadingSub}</div>}
              <video ref={resultVideoRef} controls className={`result-video ${showResultVideo ? '' : 'is-hidden'}`} />
            </div>
            {resultBlobUrl && (
              <a className="download-btn" href={resultBlobUrl} download="segmentation_result.mp4">
                DOWNLOAD RESULT
              </a>
            )}
          </div>

          {errorMessage && <div className="error-box">{errorMessage}</div>}
        </section>
      </main>

      <div className="fixed-video-dock">
        <div className="fixed-video-header">
          <div>Video Preview</div>
          <div>{fixedVideoState}</div>
        </div>
        <video ref={fixedVideoRef} className="fixed-video" controls playsInline muted />
      </div>

      {showGlobalLoading && (
        <div className="loading-dialog">
          <div className="loading-card">
            <div className="spinner" />
            <div className="loading-title">{loadingTitle}</div>
            <div className="loading-sub">{loadingSub}</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
