/**
 * Simple test upload page - no authentication required
 */

'use client';

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Card,
  CardContent,
  Button,
  Alert,
  LinearProgress,
} from '@mui/material';

export default function TestUploadPage() {
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [file, setFile] = useState<File | null>(null);

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setMessage('');
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a file first');
      return;
    }

    setUploading(true);
    setMessage('');

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('description', 'Test upload from web dashboard');

      console.log('Uploading file:', file.name, file.size, file.type);

      const response = await fetch('http://localhost:8000/data/upload', {
        method: 'POST',
        body: formData,
        // Note: No authentication headers for testing
      });

      console.log('Response status:', response.status);
      console.log('Response headers:', response.headers);

      if (response.ok) {
        const result = await response.json();
        console.log('Upload result:', result);
        setMessage(`✅ Success! Dataset ID: ${result.id || result.dataset_id || 'Unknown'}`);
      } else {
        const errorText = await response.text();
        console.error('Upload error:', errorText);
        setMessage(`❌ Upload failed: ${errorText}`);
      }
    } catch (error) {
      console.error('Network error:', error);
      setMessage(`❌ Network error: ${error}`);
    } finally {
      setUploading(false);
    }
  };

  const testBackendConnection = async () => {
    try {
      console.log('Testing backend connection...');
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      console.log('Backend response:', data);
      setMessage(`✅ Backend connected: ${data.service} v${data.version}`);
    } catch (error) {
      console.error('Backend connection error:', error);
      setMessage(`❌ Backend connection failed: ${error}`);
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        🧪 Test Upload Page
      </Typography>
      
      <Typography variant="body1" color="text.secondary" paragraph>
        Simple file upload test without authentication.
      </Typography>

      {/* Backend test */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔗 Backend Connection
          </Typography>
          <Button
            variant="outlined"
            onClick={testBackendConnection}
            sx={{ mb: 2 }}
          >
            Test Backend Connection
          </Button>
        </CardContent>
      </Card>

      {/* File upload */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            📤 File Upload
          </Typography>
          
          <input
            type="file"
            accept=".xlsx,.xls,.csv,.txt,.tsv"
            onChange={handleFileSelect}
            disabled={uploading}
            style={{ 
              marginBottom: '16px',
              padding: '8px',
              border: '1px solid #ccc',
              borderRadius: '4px'
            }}
          />
          
          {file && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.100', borderRadius: 1 }}>
              <Typography variant="body2">
                <strong>Selected:</strong> {file.name}
              </Typography>
              <Typography variant="body2">
                <strong>Size:</strong> {(file.size / 1024).toFixed(2)} KB
              </Typography>
              <Typography variant="body2">
                <strong>Type:</strong> {file.type || 'Unknown'}
              </Typography>
            </Box>
          )}
          
          <Button
            variant="contained"
            onClick={handleUpload}
            disabled={!file || uploading}
            sx={{ mb: 2 }}
          >
            {uploading ? 'Uploading...' : 'Upload File'}
          </Button>
          
          {uploading && (
            <Box sx={{ mb: 2 }}>
              <LinearProgress />
              <Typography variant="body2" sx={{ mt: 1 }}>
                Uploading {file?.name}...
              </Typography>
            </Box>
          )}
          
          {message && (
            <Alert 
              severity={message.includes('✅') ? 'success' : 'error'}
              sx={{ mt: 2 }}
            >
              {message}
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Debug info */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            🔧 Debug Information
          </Typography>
          <Typography variant="body2" sx={{ fontFamily: 'monospace' }}>
            Backend: http://localhost:8000<br/>
            Upload endpoint: http://localhost:8000/data/upload<br/>
            Frontend: {window.location.href}<br/>
            Time: {new Date().toLocaleString()}
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
}