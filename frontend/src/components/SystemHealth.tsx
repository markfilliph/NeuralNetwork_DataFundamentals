/**
 * System health monitoring component
 */

'use client';

import React from 'react';
import { Card, CardContent, Typography, Box, Chip, LinearProgress } from '@mui/material';
import { CheckCircleOutlined, ErrorOutlined, WarningOutlined } from '@mui/icons-material';

interface SystemHealthProps {
  healthStatus?: {
    status: string;
    version: string;
    service: string;
  };
}

export default function SystemHealth({ healthStatus }: SystemHealthProps) {
  const isHealthy = healthStatus?.status === 'healthy';
  
  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          🔧 System Health
        </Typography>
        
        <Box display="flex" alignItems="center" gap={1} mb={2}>
          {isHealthy ? (
            <CheckCircleOutlined color="success" />
          ) : (
            <ErrorOutlined color="error" />
          )}
          <Chip
            label={healthStatus?.status || 'Unknown'}
            color={isHealthy ? 'success' : 'error'}
            size="small"
          />
        </Box>

        {healthStatus && (
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Service: {healthStatus.service}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Version: {healthStatus.version}
            </Typography>
          </Box>
        )}

        {/* Mock system metrics */}
        <Box mt={2}>
          <Typography variant="body2" gutterBottom>
            CPU Usage
          </Typography>
          <LinearProgress
            variant="determinate"
            value={25}
            sx={{ mb: 1 }}
          />
          
          <Typography variant="body2" gutterBottom>
            Memory Usage
          </Typography>
          <LinearProgress
            variant="determinate"
            value={40}
            sx={{ mb: 1 }}
          />
          
          <Typography variant="body2" gutterBottom>
            Storage Usage
          </Typography>
          <LinearProgress
            variant="determinate"
            value={15}
          />
        </Box>
      </CardContent>
    </Card>
  );
}