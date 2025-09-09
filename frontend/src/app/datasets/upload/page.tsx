/**
 * Dataset upload page - with proper authentication
 */

'use client';

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  TextField,
  Button,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  CloudUploadOutlined,
} from '@mui/icons-material';
import { apiClient } from '@/lib/api-client';
import { useAuth } from '@/contexts/AuthContext';
import { useRouter } from 'next/navigation';

export default function UploadPage() {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const { user, isAuthenticated } = useAuth();
  const router = useRouter();

  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    return (
      <Box textAlign="center" py={8}>
        <Typography variant="h5" gutterBottom>
          Authentication Required
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={3}>
          Please log in to upload datasets.
        </Typography>
        <Button 
          variant="contained" 
          onClick={() => router.push('/login')}
        >
          Go to Login
        </Button>
      </Box>
    );
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage('');
    setProgress(0);

    try {
      console.log('Starting file upload:', file.name, 'Size:', file.size);
      
      // Use the apiClient which handles authentication
      const result = await apiClient.uploadFile(file, undefined, (progress) => {
        setProgress(progress);
        console.log('Upload progress:', progress + '%');
      });

      console.log('Upload successful:', result);
      setMessage(`Success! Dataset uploaded: ${result.filename || result.name} (ID: ${result.dataset_id || result.id})`);
      
      // Redirect to datasets page after successful upload
      setTimeout(() => {
        router.push('/datasets');
      }, 2000);
      
    } catch (error: any) {
      console.error('Upload failed:', error);
      const errorMessage = error.response?.data?.detail || error.message || String(error);
      setMessage(`Upload failed: ${errorMessage}`);
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <Box>
      {/* Header */}
      <Typography variant="h4" gutterBottom>
        📤 Upload Dataset
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Upload your data files to start analysis. Supported formats: Excel (.xlsx, .xls), 
        CSV (.csv), Text (.txt), TSV (.tsv)
      </Typography>

      {/* Simple upload form */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Choose File
          </Typography>
          
          <input
            type="file"
            accept=".xlsx,.xls,.csv,.txt,.tsv"
            onChange={handleFileUpload}
            disabled={uploading}
            style={{ marginBottom: '16px' }}
          />
          
          {uploading && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress variant="determinate" value={progress} />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Uploading... {progress}%
              </Typography>
            </Box>
          )}
          
          {message && (
            <Alert 
              severity={message.includes('Success') ? 'success' : 'error'}
              sx={{ mt: 2 }}
            >
              {message}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Backend connection test */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔧 Debug Info
          </Typography>
          <Typography variant="body2">
            Backend URL: http://localhost:8000
          </Typography>
          <Typography variant="body2">
            Upload endpoint: http://localhost:8000/data/upload
          </Typography>
          <Button
            variant="outlined"
            onClick={async () => {
              try {
                const response = await fetch('http://localhost:8000/health');
                const data = await response.json();
                setMessage(`Backend health: ${JSON.stringify(data)}`);
              } catch (error) {
                setMessage(`Backend connection failed: ${error}`);
              }
            }}
            sx={{ mt: 2 }}
          >
            Test Backend Connection
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
}