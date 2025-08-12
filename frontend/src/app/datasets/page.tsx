/**
 * Datasets page - view and manage all user datasets
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
} from '@mui/material';
import {
  UploadFileOutlined,
  DeleteOutlined,
  VisibilityOutlined,
  DownloadOutlined,
  RefreshOutlined,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useRouter } from 'next/navigation';
import { apiClient, Dataset } from '@/lib/api-client';
import { useAuth } from '@/contexts/AuthContext';

export default function DatasetsPage() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const { user, isAuthenticated } = useAuth();
  const [deleteDialog, setDeleteDialog] = useState<Dataset | null>(null);

  // Fetch datasets
  const { 
    data: datasetsResponse, 
    isLoading, 
    error,
    refetch 
  } = useQuery({
    queryKey: ['datasets'],
    queryFn: () => apiClient.getDatasets(1, 50),
    enabled: isAuthenticated, // Only fetch if authenticated
    retry: false,
  });

  // Delete mutation
  const deleteMutation = useMutation({
    mutationFn: (id: string) => apiClient.deleteDataset(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['datasets'] });
      setDeleteDialog(null);
    },
  });

  const datasets = datasetsResponse?.items || [];

  // If not authenticated, redirect to login
  if (!isAuthenticated) {
    return (
      <Box textAlign="center" py={8}>
        <Typography variant="h5" gutterBottom>
          Authentication Required
        </Typography>
        <Typography variant="body1" color="text.secondary" mb={3}>
          Please log in to view your datasets.
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

  const handleDelete = (dataset: Dataset) => {
    setDeleteDialog(dataset);
  };

  const confirmDelete = () => {
    if (deleteDialog) {
      deleteMutation.mutate(deleteDialog.id);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'ready': return 'success';
      case 'processing': return 'warning';
      case 'error': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box>
      {/* Header */}
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h4" gutterBottom>
          📊 Datasets
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
            startIcon={<UploadFileOutlined />}
            onClick={() => router.push('/datasets/upload')}
          >
            Upload Dataset
          </Button>
        </Box>
      </Box>

      {/* Loading */}
      {isLoading && <LinearProgress sx={{ mb: 2 }} />}

      {/* Error */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          Failed to load datasets: {typeof error === 'string' ? error : error.message || 'Please try again.'}
        </Alert>
      )}

      {/* No datasets */}
      {!isLoading && datasets.length === 0 && (
        <Card>
          <CardContent sx={{ textAlign: 'center', py: 8 }}>
            <UploadFileOutlined sx={{ fontSize: 64, color: 'grey.400', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              No datasets yet
            </Typography>
            <Typography color="text.secondary" paragraph>
              Upload your first dataset to get started with data analysis.
            </Typography>
            <Button
              variant="contained"
              startIcon={<UploadFileOutlined />}
              onClick={() => router.push('/datasets/upload')}
            >
              Upload Dataset
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Datasets grid */}
      {datasets.length > 0 && (
        <Grid container spacing={3}>
          {datasets.map((dataset) => (
            <Grid item xs={12} md={6} lg={4} key={dataset.id}>
              <Card>
                <CardContent>
                  <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                    <Typography variant="h6" noWrap>
                      {dataset.filename}
                    </Typography>
                    <Chip
                      size="small"
                      label={dataset.status}
                      color={getStatusColor(dataset.status) as any}
                    />
                  </Box>

                  {dataset.description && (
                    <Typography variant="body2" color="text.secondary" paragraph>
                      {dataset.description}
                    </Typography>
                  )}

                  <Box mb={2}>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Size:</strong> {formatFileSize(dataset.file_size)}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>Uploaded:</strong> {new Date(dataset.created_at).toLocaleDateString()}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>ID:</strong> {dataset.id.substring(0, 8)}...
                    </Typography>
                  </Box>

                  <Box display="flex" justifyContent="space-between">
                    <Box>
                      <IconButton
                        size="small"
                        onClick={() => router.push(`/datasets/${dataset.id}`)}
                        title="View Details"
                      >
                        <VisibilityOutlined />
                      </IconButton>
                      <IconButton
                        size="small"
                        onClick={() => {/* TODO: Implement download */}}
                        title="Download"
                      >
                        <DownloadOutlined />
                      </IconButton>
                    </Box>
                    
                    <IconButton
                      size="small"
                      color="error"
                      onClick={() => handleDelete(dataset)}
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
        <DialogTitle>Delete Dataset</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete "{deleteDialog?.filename}"? 
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