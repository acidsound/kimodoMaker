import { defineConfig } from 'vite';

export default defineConfig({
  server: {
    host: '127.0.0.1',
    allowedHosts: true,
    proxy: { '/api': { target: 'http://127.0.0.1:8765', changeOrigin: true } },
  },
  preview: {
    host: '127.0.0.1',
    port: 5173,
    allowedHosts: true,
  },
});
