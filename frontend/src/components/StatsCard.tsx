/**
 * Stats card component for dashboard metrics
 */

'use client';

import React from 'react';
import { Card, CardContent, Typography, Box } from '@mui/material';

interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'info';
}

export default function StatsCard({ 
  title, 
  value, 
  subtitle, 
  icon, 
  color = 'primary' 
}: StatsCardProps) {
  return (
    <Card>
      <CardContent>
        <Box display="flex" alignItems="center" justifyContent="space-between">
          <Box>
            <Typography color="text.secondary" gutterBottom>
              {title}
            </Typography>
            <Typography variant="h4" component="div" color={`${color}.main`}>
              {value}
            </Typography>
            {subtitle && (
              <Typography variant="body2" color="text.secondary">
                {subtitle}
              </Typography>
            )}
          </Box>
          {icon && (
            <Box sx={{ color: `${color}.main`, opacity: 0.7 }}>
              {React.cloneElement(icon as React.ReactElement, { fontSize: 'large' })}
            </Box>
          )}
        </Box>
      </CardContent>
    </Card>
  );
}