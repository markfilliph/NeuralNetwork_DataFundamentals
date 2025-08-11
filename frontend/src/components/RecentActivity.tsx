/**
 * Recent activity component for dashboard
 */

'use client';

import React from 'react';
import { List, ListItem, ListItemIcon, ListItemText, Typography, Box, Chip } from '@mui/material';
import { DatasetOutlined, ModelTrainingOutlined, TrendingUpOutlined } from '@mui/icons-material';
import { Dataset, Model } from '@/lib/api-client';

interface RecentActivityProps {
  datasets: Dataset[];
  models: Model[];
}

export default function RecentActivity({ datasets, models }: RecentActivityProps) {
  // Combine and sort by creation date
  const activities = [
    ...datasets.map(d => ({
      type: 'dataset',
      id: d.id,
      title: d.filename,
      subtitle: `Dataset uploaded`,
      time: new Date(d.created_at),
      status: d.status
    })),
    ...models.map(m => ({
      type: 'model',
      id: m.id,
      title: `Model for ${m.target_column}`,
      subtitle: `R² Score: ${m.metrics?.r2_score?.toFixed(3) || 'N/A'}`,
      time: new Date(m.created_at),
      status: m.status
    }))
  ].sort((a, b) => b.time.getTime() - a.time.getTime()).slice(0, 5);

  if (activities.length === 0) {
    return (
      <Box textAlign="center" py={4}>
        <Typography color="text.secondary">
          No recent activity. Upload a dataset or train a model to get started!
        </Typography>
      </Box>
    );
  }

  return (
    <List>
      {activities.map((activity) => (
        <ListItem key={`${activity.type}-${activity.id}`}>
          <ListItemIcon>
            {activity.type === 'dataset' ? (
              <DatasetOutlined color="primary" />
            ) : (
              <ModelTrainingOutlined color="secondary" />
            )}
          </ListItemIcon>
          <ListItemText
            primary={activity.title}
            secondary={
              <Box display="flex" alignItems="center" justifyContent="space-between" mt={0.5}>
                <Typography variant="body2" color="text.secondary">
                  {activity.subtitle}
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <Chip
                    size="small"
                    label={activity.status}
                    color={
                      activity.status === 'ready' || activity.status === 'completed'
                        ? 'success'
                        : activity.status === 'processing' || activity.status === 'training'
                        ? 'warning'
                        : 'error'
                    }
                  />
                  <Typography variant="caption" color="text.secondary">
                    {activity.time.toLocaleDateString()}
                  </Typography>
                </Box>
              </Box>
            }
          />
        </ListItem>
      ))}
    </List>
  );
}