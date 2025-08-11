/**
 * Models page - view and manage trained models
 */

'use client';

import React, { useState } from 'react';
import {
  Box,
  Typography,
  Button,
  Card,
  CardContent,
  Grid,
  Chip,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  LinearProgress,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
} from '@mui/material';
import {
  ModelTrainingOutlined,
  DeleteOutlined,
  VisibilityOutlined,
  DownloadOutlined,
  RefreshOutlined,
  TrendingUpOutlined,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { apiClient, Model } from '@/lib/api-client';

export default function ModelsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [deleteDialog, setDeleteDialog] = useState<Model | null>(null);

  // Fetch models
  const { 
    data: modelsResponse, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['models'],
    queryFn: () => apiClient.getModels(1, 50),
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteModel(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['models'] });
      setDeleteDialog(null);
    },
  });

  const models = modelsResponse?.items || [];

  const handleDelete = (model: Model) => {
    setDeleteDialog(model);
  };

  const confirmDelete = () => {
    if (deleteDialog) {
      deleteMutation.mutate(deleteDialog.id);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'success';
      case 'training': return 'warning';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const formatScore = (score?: number) => {
    return score ? score.toFixed(4) : 'N/A';
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          🤖 Models
        </Typography>
        <Box display="flex" gap={1}>
          <Button
            startIcon={<RefreshOutlined />}
            onClick={() => refetch()}
            disabled={isLoading}
          >
            Refresh
          </Button>
          <Button
            variant="contained"
            startIcon={<ModelTrainingOutlined />}
            onClick={() => router.push('/models/train')}
          >
            Train Model
          </Button>
        </Box>
      </Box>

      {/* Loading */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load models. Please try again.
        </Alert>
      )}

      {/* No models */}
      {!isLoading && models.length === 0 && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 8 }}>
            <ModelTrainingOutlined sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No models trained yet
            </Typography>
            <Typography color="text.secondary" paragraph>
              Train your first machine learning model to get predictions and insights.
            </Typography>
            <Button
              variant="contained"
              startIcon={<ModelTrainingOutlined />}
              onClick={() => router.push('/models/train')}
            >
              Train Model
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Models grid */}
      {models.length > 0 && (
        <Grid container spacing={3}>
          {models.map((model) => (
            <Grid item xs={12} lg={6} key={model.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        Linear Regression Model
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Target: <strong>{model.target_column}</strong>
                      </Typography>
                    </Box>
                    <Chip
                      size="small"
                      label={model.status}
                      color={getStatusColor(model.status) as any}
                    />
                  </Box>

                  {/* Model metrics */}
                  {model.metrics && (
                    <Box mb={2}>
                      <Typography variant="subtitle2" gutterBottom>
                        📈 Performance Metrics
                      </Typography>
                      <TableContainer component={Paper} variant="outlined">
                        <Table size="small">
                          <TableBody>
                            <TableRow>
                              <TableCell><strong>R² Score</strong></TableCell>
                              <TableCell align="right">
                                <Box display="flex" alignItems="center" gap={1}>
                                  <TrendingUpOutlined 
                                    fontSize="small" 
                                    color={model.metrics.r2_score > 0.7 ? 'success' : 'warning'} 
                                  />
                                  {formatScore(model.metrics.r2_score)}
                                </Box>
                              </TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>MSE</strong></TableCell>
                              <TableCell align="right">{formatScore(model.metrics.mse)}</TableCell>
                            </TableRow>
                            <TableRow>
                              <TableCell><strong>MAE</strong></TableCell>
                              <TableCell align="right">{formatScore(model.metrics.mae)}</TableCell>
                            </TableRow>
                          </TableBody>
                        </Table>
                      </TableContainer>
                    </Box>
                  )}

                  {/* Features */}
                  <Box mb={2}>
                    <Typography variant="subtitle2" gutterBottom>
                      🔧 Features ({model.feature_columns.length})
                    </Typography>
                    <Box display="flex" flexWrap="wrap" gap={0.5}>
                      {model.feature_columns.slice(0, 3).map((feature) => (
                        <Chip 
                          key={feature} 
                          label={feature} 
                          size="small" 
                          variant="outlined" 
                        />
                      ))}
                      {model.feature_columns.length > 3 && (
                        <Chip 
                          label={`+${model.feature_columns.length - 3} more`} 
                          size="small" 
                          variant="outlined"
                          color="secondary"
                        />
                      )}
                    </Box>
                  </Box>

                  {/* Metadata */}
                  <Box mb={2}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Created:</strong> {new Date(model.created_at).toLocaleDateString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Model ID:</strong> {model.id.substring(0, 8)}...
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Dataset:</strong> {model.dataset_id.substring(0, 8)}...
                    </Typography>
                  </Box>

                  {/* Actions */}
                  <Box display="flex" justifyContent="space-between">
                    <Box>
                      <IconButton
                        size="small"
                        onClick={() => router.push(`/models/${model.id}`)}
                        title="View Details"
                      >
                        <VisibilityOutlined />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => {/* TODO: Implement download */}}
                        title="Download"
                        disabled={model.status !== 'completed'}
                      >
                        <DownloadOutlined />
                      </IconButton>
                    </Box>
                    
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDelete(model)}
                      title="Delete"
                    >
                      <DeleteOutlined />
                    </IconButton>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      )}

      {/* Delete confirmation dialog */}
      <Dialog open={!!deleteDialog} onClose={() => setDeleteDialog(null)}>
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete this model for "{deleteDialog?.target_column}"? 
            This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialog(null)}>Cancel</Button>
          <Button 
            color="error" 
            onClick={confirmDelete}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}