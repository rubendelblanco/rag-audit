import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: path.resolve(__dirname, '../src/rag_audit/web/dist'),
    emptyOutDir: true,
  },
  server: {
    // For standalone frontend dev, proxy the Python API
    proxy: {
      '/api': 'http://127.0.0.1:8844',
    },
  },
});