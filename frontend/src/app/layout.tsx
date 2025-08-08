/**
 * Root layout component for Next.js app
 * Sets up global providers and theme
 */

'use client';

import { Inter } from 'next/font/google';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useState } from 'react';

import { AuthProvider } from '@/contexts/AuthContext';
import { WebSocketProvider } from '@/contexts/WebSocketContext';
import Navigation from '@/components/Navigation';
import LoadingBar from '@/components/LoadingBar';

const inter = Inter({ subsets: ['latin'] });

// Create Material-UI theme
const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: inter.style.fontFamily,
    h1: {
      fontSize: '2.5rem',
      fontWeight: 600,
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
    },
    h3: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          borderRadius: 12,
        },
      },
    },
  },
});

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Create React Query client
  const [queryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 5 * 60 * 1000, // 5 minutes
            retry: 1,
            refetchOnWindowFocus: false,
          },
          mutations: {
            retry: 1,
          },
        },
      })
  );

  return (
    <html lang="en">
      <head>
        <title>DAPP - Data Analysis Platform</title>
        <meta name="description" content="Advanced data analysis platform with machine learning capabilities" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </head>
      <body className={inter.className}>
        <QueryClientProvider client={queryClient}>
          <ThemeProvider theme={theme}>
            <CssBaseline />
            <AuthProvider>
              <WebSocketProvider>
                <Box
                  sx={{
                    display: 'flex',
                    flexDirection: 'column',
                    minHeight: '100vh',
                    bgcolor: 'background.default',
                  }}
                >
                  <LoadingBar />
                  <Navigation />
                  <Box
                    component="main"
                    sx={{
                      flexGrow: 1,
                      p: 3,
                      mt: 8, // Account for AppBar height
                    }}
                  >
                    {children}
                  </Box>
                </Box>
              </WebSocketProvider>
            </AuthProvider>
          </ThemeProvider>
        </QueryClientProvider>
      </body>
    </html>
  );
}