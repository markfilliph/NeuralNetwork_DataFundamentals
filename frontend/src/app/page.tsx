/**
 * Dashboard home page - main entry point for authenticated users
 * Shows overview of datasets, models, and recent activity
 */

'use client';

import { Grid, Card, CardContent, Typography, Box, Chip, Button } from '@mui/material';
import { 
  DatasetOutlined, 
  ModelTrainingOutlined, 
  TrendingUpOutlined,
  UploadFileOutlined,
  AnalyticsOutlined 
} from '@mui/icons-material';
import { useQuery } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';

import { apiClient } from '@/lib/api-client';
import { useAuth } from '@/contexts/AuthContext';
import { useWebSocket } from '@/contexts/WebSocketContext';
import StatsCard from '@/components/StatsCard';
import RecentActivity from '@/components/RecentActivity';
import QuickActions from '@/components/QuickActions';
import SystemHealth from '@/components/SystemHealth';

export default function Dashboard() {
  const router = useRouter();
  const { user } = useAuth();
  const { connectionStatus, lastActivity } = useWebSocket();

  // Fetch dashboard data with error handling
  const { data: datasets, error: datasetsError } = useQuery({
    queryKey: ['datasets', { page: 1, per_page: 5 }],
    queryFn: () => apiClient.getDatasets(1, 5),
    retry: false,
    enabled: !!user, // Only fetch if authenticated
  });

  const { data: models, error: modelsError } = useQuery({
    queryKey: ['models', { page: 1, per_page: 5 }],
    queryFn: () => apiClient.getModels(1, 5),
    retry: false,
    enabled: !!user, // Only fetch if authenticated
  });

  const { data: healthStatus, error: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: () => apiClient.healthCheck(),
    refetchInterval: 30000, // Check every 30 seconds
    retry: 1, // Health check can be retried
  });

  // If no user, show login prompt
  if (!user) {
    return (
      <Box textAlign="center" py={8}>
        <Typography variant="h4" gutterBottom>
          Welcome to DAPP - Data Analysis Platform
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={3}>
          Please log in to access your dashboard and start analyzing data.
        </Typography>
        <Button 
          variant="contained" 
          size="large"
          onClick={() => router.push('/login')}
        >
          Go to Login
        </Button>
      </Box>
    );
  }

  return (
    <Box>
      {/* Welcome Section */}
      <Box mb={4}>
        <Typography variant="h3" gutterBottom>
          Welcome back, {user?.email?.split('@')[0] || 'User'}! 👋
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={2}>
          Ready to analyze your data? Here's what's happening in your platform.
        </Typography>
        
        {/* Connection Status */}
        <Chip
          label={`WebSocket: ${connectionStatus}`}
          color={connectionStatus === 'connected' ? 'success' : 'warning'}
          size="small"
          sx={{ mr: 1 }}
        />
        {lastActivity && (
          <Chip
            label={`Last activity: ${new Date(lastActivity).toLocaleTimeString()}`}
            size="small"
            variant="outlined"
          />
        )}
      </Box>

      {/* Stats Cards */}
      <Grid container spacing={3} mb={4}>
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Datasets"
            value={String(datasets?.total || 0)}
            icon={<DatasetOutlined />}
            color="primary"
            subtitle={`${datasets?.items?.filter?.(d => d.status === 'ready').length || 0} ready`}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Models"
            value={String(models?.total || 0)}
            icon={<ModelTrainingOutlined />}
            color="secondary"
            subtitle={`${models?.items?.filter?.(m => m.status === 'completed').length || 0} completed`}
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Avg R² Score"
            value={(() => {
              if (!models?.items?.length) return '0.000';
              const validModels = models.items.filter(m => m.metrics?.r2_score);
              if (!validModels.length) return '0.000';
              const average = validModels.reduce((sum, m) => sum + (m.metrics?.r2_score || 0), 0) / validModels.length;
              return average.toFixed(3);
            })()}
            icon={<TrendingUpOutlined />}
            color="success"
            subtitle="Model performance"
          />
        </Grid>
        
        <Grid item xs={12} sm={6} md={3}>
          <StatsCard
            title="Storage Used"
            value="15.2 MB"
            icon={<AnalyticsOutlined />}
            color="info"
            subtitle="of 10 GB limit"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Quick Actions */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                🚀 Quick Actions
              </Typography>
              <QuickActions />
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Activity */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                📈 Recent Activity
              </Typography>
              <RecentActivity datasets={datasets?.items || []} models={models?.items || []} />
            </CardContent>
          </Card>
        </Grid>

        {/* System Health */}
        <Grid item xs={12} md={6}>
          <SystemHealth healthStatus={healthStatus} />
        </Grid>

        {/* Recent Datasets */}
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
                <Typography variant="h6">
                  📊 Recent Datasets
                </Typography>
                <Button 
                  size="small" 
                  onClick={() => router.push('/datasets')}
                >
                  View All
                </Button>
              </Box>
              
              {datasets?.items?.length ? (
                <Box>
                  {datasets.items.slice(0, 3).map((dataset) => (
                    <Box
                      key={dataset.id || Math.random()}
                      sx={{
                        p: 2,
                        mb: 1,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        '&:hover': {
                          bgcolor: 'action.hover',
                          cursor: 'pointer',
                        },
                      }}
                      onClick={() => router.push(`/datasets/${dataset.id}`)}
                    >
                      <Typography variant="subtitle2" gutterBottom>
                        {dataset.filename || 'Unknown dataset'}
                      </Typography>
                      <Box display="flex" alignItems="center" justifyContent="space-between">
                        <Typography variant="caption" color="text.secondary">
                          {dataset.created_at ? new Date(dataset.created_at).toLocaleDateString() : 'Unknown date'}
                        </Typography>
                        <Chip
                          size="small"
                          label={dataset.status || 'unknown'}
                          color={dataset.status === 'ready' ? 'success' : 'warning'}
                        />
                      </Box>
                    </Box>
                  ))}
                </Box>
              ) : (
                <Box
                  sx={{
                    textAlign: 'center',
                    py: 4,
                    color: 'text.secondary',
                  }}
                >
                  <DatasetOutlined sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                  <Typography variant="body2">
                    No datasets yet. Upload your first dataset to get started!
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<UploadFileOutlined />}
                    sx={{ mt: 2 }}
                    onClick={() => router.push('/datasets/upload')}
                  >
                    Upload Dataset
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}