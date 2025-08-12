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
  // Handle case where healthStatus might be an error object
  const safeHealthStatus = healthStatus && typeof healthStatus === 'object' && 'status' in healthStatus ? healthStatus : null;
  const isHealthy = safeHealthStatus?.status === 'healthy';
  
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
            label={safeHealthStatus?.status || 'Unknown'}
            color={isHealthy ? 'success' : 'error'}
            size="small"
          />
        </Box>

        {safeHealthStatus && (
          <Box>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Service: {safeHealthStatus.service || 'Unknown'}
            </Typography>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Version: {safeHealthStatus.version || 'Unknown'}
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