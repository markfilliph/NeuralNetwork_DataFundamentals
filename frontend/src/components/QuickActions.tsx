/**
 * Quick actions component for dashboard
 */

'use client';

import React from 'react';
import { Box, Button, Stack } from '@mui/material';
import { 
  UploadFileOutlined, 
  ModelTrainingOutlined, 
  AnalyticsOutlined,
  CodeOutlined 
} from '@mui/icons-material';
import { useRouter } from 'next/navigation';

export default function QuickActions() {
  const router = useRouter();

  return (
    <Stack spacing={2}>
      <Button
        variant="contained"
        startIcon={<UploadFileOutlined />}
        fullWidth
        onClick={() => router.push('/datasets/upload')}
      >
        Upload Dataset
      </Button>
      
      <Button
        variant="outlined"
        startIcon={<ModelTrainingOutlined />}
        fullWidth
        onClick={() => router.push('/models/train')}
      >
        Train Model
      </Button>
      
      <Button
        variant="outlined"
        startIcon={<AnalyticsOutlined />}
        fullWidth
        onClick={() => router.push('/analytics')}
      >
        View Analytics
      </Button>
      
      <Button
        variant="outlined"
        startIcon={<CodeOutlined />}
        fullWidth
        onClick={() => window.open('http://localhost:8888', '_blank')}
      >
        Open Jupyter
      </Button>
    </Stack>
  );
}