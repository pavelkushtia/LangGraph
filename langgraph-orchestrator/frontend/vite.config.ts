import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0', // Bind to all network interfaces
    proxy: {
      '/api': {
        target: process.env.VITE_API_URL || 'http://192.168.1.81:3001',
        changeOrigin: true,
        secure: false,
      },
      '/socket.io': {
        target: process.env.VITE_WS_URL?.replace('ws://', 'http://').replace('wss://', 'https://') || 'http://192.168.1.81:3001',
        changeOrigin: true,
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          ui: ['lucide-react', 'recharts'],
          flow: ['@reactflow/core', '@reactflow/controls', '@reactflow/background'],
        },
      },
    },
  },
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'socket.io-client',
      'axios',
      'react-query',
      'zustand',
      '@reactflow/core',
      '@reactflow/controls',
      '@reactflow/background',
    ],
  },
});
