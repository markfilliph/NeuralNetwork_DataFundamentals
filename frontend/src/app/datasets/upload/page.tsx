/**
 * Dataset upload page - simplified version for debugging
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

export default function UploadPage() {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/data/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        setMessage(`Success! Dataset ID: ${result.id}`);
      } else {
        const error = await response.text();
        setMessage(`Upload failed: ${error}`);
      }
    } catch (error) {
      setMessage(`Upload failed: ${error}`);
    } finally {
      setUploading(false);
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
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Uploading...
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