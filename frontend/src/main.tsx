import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import { App } from './app'

const root = document.querySelector('#root')
if (!root) throw new Error('root element missing')
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
