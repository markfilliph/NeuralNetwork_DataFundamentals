/**
 * Global loading bar component
 */

'use client';

import React from 'react';
import { LinearProgress, Box } from '@mui/material';
import { useRouter } from 'next/navigation';

export default function LoadingBar() {
  const [loading, setLoading] = React.useState(false);

  return (
    <Box sx={{ position: 'fixed', top: 0, left: 0, right: 0, zIndex: 9999 }}>
      {loading && (
        <LinearProgress 
          sx={{ 
            height: 2,
            backgroundColor: 'transparent',
            '& .MuiLinearProgress-bar': {
              backgroundColor: 'primary.main'
            }
          }} 
        />
      )}
    </Box>
  );
}